# =============================================================================
# YOLO 学習（バックグラウンド実行）
# =============================================================================
from __future__ import annotations

import os
import threading
from pathlib import Path

from .config import MLFLOW_URI, MODELS_DIR
from .dataset import resolve_train_data_arg
from .provenance import record_model_provenance
from .state import _get_train_shared


# ---------------------------------------------------------------------------
# MLflow 設定
# ---------------------------------------------------------------------------
def init_mlflow(project_name: str, run_name: str) -> bool:
    """MLflow サーバーへの接続確認と環境変数設定。
    Ultralytics の MLflow コールバックが自動でメトリクス・モデルを記録する。
    """
    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.tracking.MlflowClient().search_experiments()  # 接続テスト
        os.environ["MLFLOW_TRACKING_URI"]   = MLFLOW_URI
        os.environ["MLFLOW_EXPERIMENT_NAME"] = project_name
        os.environ["MLFLOW_RUN"]             = run_name
        print(f"[MLflow] 接続OK: {MLFLOW_URI} / {project_name} / {run_name}")
        return True
    except Exception as e:
        print(f"[MLflow] 接続エラー（実験追跡なし）: {e}")
        return False


# ---------------------------------------------------------------------------
# YOLO 学習ワーカー (別スレッドで実行)
# ---------------------------------------------------------------------------

class TrainingStopped(Exception):
    """UI からの停止要求で学習を打ち切ったことを表す。

    Ultralytics の `trainer.stop = True` で止めると「予定エポックを完走した」と
    記録され resume できなくなるため、例外でループを抜けてエポック末の
    last.pt をそのまま残す（= 続きから再開できる状態にする）。
    """


class _StdoutCapture:
    """sys.stdout を乗っ取り、YOLO の print 出力を _train_state["log"] に転送する。
    元の stdout にも同時に書くので docker logs でも確認できる。
    """
    def __init__(self, original, lock: threading.Lock, state: dict) -> None:
        self._orig  = original
        self._lock  = lock
        self._state = state
        self._buf   = ""

    def write(self, text: str) -> int:
        self._orig.write(text)
        self._buf += text
        # 改行単位で確定させる
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            line = line.rstrip()
            if line:
                with self._lock:
                    self._state["log"].append(line)
        return len(text)

    def flush(self) -> None:
        self._orig.flush()

    def fileno(self) -> int:
        return self._orig.fileno()


def _train_worker(
    data_yaml: str,
    model_name: str,
    epochs: int,
    batch_size: int,
    project_name: str,
    run_name: str,
    train_kwargs: dict,
):
    """バックグラウンドスレッドで YOLO 学習を実行する。
    sys.stdout を _StdoutCapture に差し替えて全 print 出力を UI に転送する。
    st.session_state はスレッド外から参照不可のため、_train_state 経由で通信する。
    train_kwargs は model.train() に **kwargs として渡す追加パラメータ。
    """
    import sys

    # 共有状態はここで取り出す。main.py / tab_train.py 側のモジュール変数を
    # 参照していたが、関数のグローバルは定義元（このファイル）で解決されるため
    # NameError になっていた（main.py の分割で混入）。
    _train_state, _train_log_lock = _get_train_shared()

    def _log(msg: str) -> None:
        with _train_log_lock:
            _train_state["log"].append(msg)

    def _on_epoch_end(trainer) -> None:
        cur   = trainer.epoch + 1
        total = trainer.epochs
        with _train_log_lock:
            _train_state["progress"] = int(cur / total * 95)
            _stop = _train_state.get("stop_requested", False)
        # エポック末の重みは保存済みなので、ここで抜ければ last.pt から再開できる
        if _stop:
            _log(f"[停止] {cur} エポック終了時点で学習を中断します。")
            raise TrainingStopped()

    def _on_fit_epoch_end(trainer) -> None:
        row: dict = {"epoch": trainer.epoch + 1}
        if hasattr(trainer, "metrics") and trainer.metrics:
            for k, v in trainer.metrics.items():
                try:
                    row[k] = float(v)
                except (TypeError, ValueError):
                    pass
        with _train_log_lock:
            _train_state["metrics_history"].append(row)

    _orig_stdout = sys.stdout
    sys.stdout   = _StdoutCapture(_orig_stdout, _train_log_lock, _train_state)

    try:
        mlflow_ok = init_mlflow(project_name, run_name)
        if mlflow_ok:
            _log(f"[MLflow] 実験追跡: {project_name} / {run_name}")
        else:
            _log("[MLflow] スキップ（実験追跡なし）")

        from ultralytics import YOLO

        model = YOLO(model_name)
        model.add_callback("on_train_epoch_end", _on_epoch_end)
        model.add_callback("on_fit_epoch_end", _on_fit_epoch_end)

        # 学習に入る前に「何で学習したか」を記録しておく
        # （途中で止めても、失敗しても残るように開始時点で書く）
        try:
            record_model_provenance(
                run_dir=MODELS_DIR / run_name,
                data_yaml=data_yaml,
                base_model=model_name,
                params={"epochs": epochs, "batch": batch_size, **train_kwargs},
                resumed=bool(train_kwargs.get("resume")),
            )
        except Exception as e:
            _log(f"[来歴] 記録をスキップしました: {e}")

        if train_kwargs.get("resume"):
            # 再開時は epochs / batch / data などを中断時の設定 (last.pt の args) から
            # 復元するため、こちらからは渡さない。
            # resume には bool ではなく last.pt のパスを渡すこと。True だと
            # Ultralytics が「最新の run」を自動探索してしまい、別の学習を再開する。
            _log(f"[再開] {model_name} から学習を再開します")
            _rk = {k: v for k, v in train_kwargs.items() if k != "resume"}
            results = model.train(resume=model_name, **_rk)
        else:
            results = model.train(
                # classify はディレクトリ、それ以外は data.yaml を渡す
                data=resolve_train_data_arg(data_yaml),
                epochs=epochs,
                batch=batch_size,
                project=str(MODELS_DIR),
                name=run_name,
                exist_ok=True,
                **train_kwargs,
            )

        best_model = Path(results.save_dir) / "weights" / "best.pt"
        with _train_log_lock:
            _train_state["model_path"] = str(best_model)
            _train_state["progress"]   = 100
        _log(f"[完了] best.pt: {best_model}")

        if mlflow_ok:
            try:
                import mlflow
                # Ultralytics callback がすでに run を close している場合に備えて、
                # 最後の run を取得して model を登録する
                runs = mlflow.search_runs(
                    experiment_names=[project_name],
                    filter_string=f"tags.mlflow.runName = '{run_name}'",
                    max_results=1,
                )
                if not runs.empty:
                    run_id = runs.iloc[0]["run_id"]
                    mv = mlflow.register_model(
                        f"runs:/{run_id}/weights",
                        project_name,
                    )
                    _log(f"[MLflow] モデル登録: {project_name} v{mv.version}")
            except Exception as e:
                _log(f"[MLflow] モデル登録スキップ: {e}")

    except TrainingStopped:
        # 停止はエラーではない。エポック末の重みが残っているので再開できる
        _best = MODELS_DIR / run_name / "weights" / "best.pt"
        with _train_log_lock:
            _train_state["progress"] = 100
            if _best.exists():
                _train_state["model_path"] = str(_best)
        _log("[停止] 学習を中断しました。"
             + (f"その時点までの best.pt: {_best}" if _best.exists() else "")
             + " 「中断した学習を再開する」から続きから再開できます。")

    except Exception as e:
        _log(f"[ERROR] {e}")
        with _train_log_lock:
            _train_state["error"] = str(e)

    finally:
        sys.stdout = _orig_stdout
        with _train_log_lock:
            _train_state["running"] = False
            _train_state["stop_requested"] = False
