# =============================================================================
# YOLO 学習（バックグラウンド実行）
# =============================================================================
from __future__ import annotations

import os
import re
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
# 学習に使うデバイス
#
#   `device=0` 決め打ちだと、GPU の無い構成（docker-compose.cpu.yml）で
#   「Invalid CUDA 'device=0' requested」になり学習が必ず失敗する。
# ---------------------------------------------------------------------------
def default_train_device():
    """GPU が使えれば 0、使えなければ "cpu" を返す"""
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return 0
    except Exception:
        pass
    return "cpu"


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


# ---------------------------------------------------------------------------
# 学習本体（子プロセスで動く）
#
#   学習は Streamlit と別のプロセスで回す。同じプロセスのスレッドで回していたときは
#     - sys.stdout の差し替えがプロセス全体に効き、評価や探索のログと混ざる
#     - GPU メモリ不足やクラッシュで UI ごと落ちる
#     - DataLoader の fork がマルチスレッドのプロセスから走る
#   という問題があった。子プロセスは標準出力に「ログ行」と「イベント行」を出し、
#   親（_train_worker）がそれを読んで共有状態へ流す。
# ---------------------------------------------------------------------------
EVENT_PREFIX = "@@TRAIN_EVENT@@ "
MAX_LOG_LINES = 3000          # 何時間も回るので際限なく溜めない
APP_DIR = Path(__file__).resolve().parents[1]


def run_training_job(cfg: dict, emit, stop_requested) -> None:
    """学習を 1 回まわす（子プロセス側）。

    emit(kind, **kw) … 親へイベントを送る（progress / metrics / model_path / error / stopped）
    stop_requested() … 停止が頼まれていれば True
    """
    data_yaml    = cfg["data_yaml"]
    model_name   = cfg["model_name"]
    epochs       = cfg["epochs"]
    batch_size   = cfg["batch_size"]
    project_name = cfg["project_name"]
    run_name     = cfg["run_name"]
    train_kwargs = dict(cfg.get("train_kwargs") or {})

    def _on_epoch_end(trainer) -> None:
        cur   = trainer.epoch + 1
        total = trainer.epochs
        emit("progress", value=int(cur / total * 95))

    def _on_fit_epoch_end(trainer) -> None:
        cur = trainer.epoch + 1
        row: dict = {"epoch": cur}
        if hasattr(trainer, "metrics") and trainer.metrics:
            for k, v in trainer.metrics.items():
                try:
                    row[k] = float(v)
                except (TypeError, ValueError):
                    pass
        emit("metrics", row=row)
        # 停止はここで見る。on_train_epoch_end は検証・重みの保存より**前**に
        # 呼ばれるため、そこで抜けるとそのエポックの last.pt が残らず
        # （1 エポック目なら再開そのものができない）。ここなら保存済み。
        if stop_requested():
            print(f"[停止] {cur} エポック終了時点で学習を中断します。", flush=True)
            raise TrainingStopped()

    try:
        mlflow_ok = init_mlflow(project_name, run_name)
        if mlflow_ok:
            print(f"[MLflow] 実験追跡: {project_name} / {run_name}", flush=True)
        else:
            print("[MLflow] スキップ（実験追跡なし）", flush=True)

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
            print(f"[来歴] 記録をスキップしました: {e}", flush=True)

        if train_kwargs.get("resume"):
            # 再開時は epochs / batch / data などを中断時の設定 (last.pt の args) から
            # 復元するため、こちらからは渡さない。
            # resume には bool ではなく last.pt のパスを渡すこと。True だと
            # Ultralytics が「最新の run」を自動探索してしまい、別の学習を再開する。
            print(f"[再開] {model_name} から学習を再開します", flush=True)
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
        emit("model_path", value=str(best_model))
        emit("progress", value=100)
        print(f"[完了] best.pt: {best_model}", flush=True)

        if mlflow_ok:
            try:
                import mlflow
                # Ultralytics callback がすでに run を close している場合に備えて、
                # 最後の run を取得して model を登録する
                _rn = run_name.replace("\\", "\\\\").replace("'", "\\'")
                runs = mlflow.search_runs(
                    experiment_names=[project_name],
                    filter_string=f"tags.mlflow.runName = '{_rn}'",
                    max_results=1,
                )
                if not runs.empty:
                    run_id = runs.iloc[0]["run_id"]
                    mv = mlflow.register_model(
                        f"runs:/{run_id}/weights",
                        project_name,
                    )
                    print(f"[MLflow] モデル登録: {project_name} v{mv.version}", flush=True)
            except Exception as e:
                print(f"[MLflow] モデル登録スキップ: {e}", flush=True)

    except TrainingStopped:
        # 停止はエラーではない。エポック末の重みが残っているので再開できる
        _best = MODELS_DIR / run_name / "weights" / "best.pt"
        emit("progress", value=100)
        if _best.exists():
            emit("model_path", value=str(_best))
        emit("stopped")
        print("[停止] 学習を中断しました。"
              + (f"その時点までの best.pt: {_best}" if _best.exists() else "")
              + " 「中断した学習を再開する」から続きから再開できます。", flush=True)

    except Exception as e:
        print(f"[ERROR] {e}", flush=True)
        emit("error", value=str(e))


# ---------------------------------------------------------------------------
# 親側: 子プロセスを起こして、出力を共有状態へ流す
# ---------------------------------------------------------------------------
_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def _apply_output_line(raw: bytes, state: dict, lock: threading.Lock) -> None:
    """子プロセスの出力 1 行を共有状態に反映する"""
    import json

    text = raw.decode("utf-8", errors="replace").rstrip("\n")
    # 進捗バーは \r で上書きしながら出るので、最後の姿だけ残す
    text = _ANSI.sub("", text.split("\r")[-1]).rstrip()
    if not text:
        return
    if text.startswith(EVENT_PREFIX):
        try:
            ev = json.loads(text[len(EVENT_PREFIX):])
        except ValueError:
            ev = None
        if isinstance(ev, dict):
            kind = ev.get("type")
            with lock:
                if kind == "progress":
                    state["progress"] = int(ev.get("value", 0))
                elif kind == "metrics":
                    state["metrics_history"].append(ev.get("row") or {})
                elif kind == "model_path":
                    state["model_path"] = ev.get("value")
                elif kind == "error":
                    state["error"] = ev.get("value") or "学習に失敗しました"
            return
    with lock:
        state["log"].append(text)
        if len(state["log"]) > MAX_LOG_LINES:
            del state["log"][:-MAX_LOG_LINES]


def _train_worker(
    data_yaml: str,
    model_name: str,
    epochs: int,
    batch_size: int,
    project_name: str,
    run_name: str,
    train_kwargs: dict,
):
    """バックグラウンドスレッドから学習の子プロセスを動かし、終わるまで見守る。
    st.session_state はスレッド外から参照不可のため、_train_state 経由で通信する。
    train_kwargs は model.train() に **kwargs として渡す追加パラメータ。
    """
    import json
    import subprocess
    import sys
    import tempfile
    import time

    _train_state, _train_log_lock = _get_train_shared()

    def _log(msg: str) -> None:
        with _train_log_lock:
            _train_state["log"].append(msg)

    job_dir = Path(tempfile.mkdtemp(prefix="train_job_"))
    stop_file = job_dir / "stop"
    cfg_path = job_dir / "config.json"
    proc = None
    try:
        cfg_path.write_text(json.dumps({
            "data_yaml": data_yaml, "model_name": model_name,
            "epochs": epochs, "batch_size": batch_size,
            "project_name": project_name, "run_name": run_name,
            "train_kwargs": train_kwargs, "stop_file": str(stop_file),
        }, ensure_ascii=False), encoding="utf-8")

        env = dict(os.environ, PYTHONUNBUFFERED="1")
        proc = subprocess.Popen(
            [sys.executable, "-u", "-m", "core.train_runner", str(cfg_path)],
            cwd=str(APP_DIR), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        with _train_log_lock:
            _train_state["pid"] = proc.pid

        # 停止の要求を子プロセスへ伝える（ファイルを置くと、エポック末で止まる）
        def _watch_stop() -> None:
            while proc.poll() is None:
                with _train_log_lock:
                    want = _train_state.get("stop_requested", False)
                if want and not stop_file.exists():
                    stop_file.touch()
                time.sleep(1.0)

        threading.Thread(target=_watch_stop, daemon=True).start()

        for raw in proc.stdout:
            # docker logs でも追えるように、そのまま流す
            try:
                sys.__stdout__.write(raw.decode("utf-8", errors="replace"))
            except Exception:
                pass
            _apply_output_line(raw, _train_state, _train_log_lock)

        code = proc.wait()
        with _train_log_lock:
            no_error = not _train_state["error"]
        if code != 0 and no_error:
            # 子プロセスが報告できずに落ちた（メモリ不足で強制終了された等）
            msg = (f"学習プロセスが異常終了しました（終了コード {code}）。"
                   + ("メモリ不足で強制終了された可能性があります。"
                      "batch や imgsz を下げるか、workers を減らしてください。"
                      if code in (-9, 137) else ""))
            _log(f"[ERROR] {msg}")
            with _train_log_lock:
                _train_state["error"] = msg

    except Exception as e:
        _log(f"[ERROR] {e}")
        with _train_log_lock:
            _train_state["error"] = str(e)

    finally:
        if proc is not None and proc.poll() is None:
            proc.kill()
        import shutil
        shutil.rmtree(job_dir, ignore_errors=True)
        with _train_log_lock:
            _train_state["running"] = False
            _train_state["stop_requested"] = False
            _train_state["pid"] = None


def start_training(
    data_yaml: str,
    model_name: str,
    epochs: int,
    batch_size: int,
    project_name: str,
    run_name: str,
    train_kwargs: dict,
) -> tuple[bool, str]:
    """学習をバックグラウンドで始める。(始めたか, 始めなかった理由) を返す。

    画面の「実行中か」はブラウザのタブごとの session_state に写した値なので、
    2 つのタブ（2 人）がほぼ同時に押すと両方すり抜ける。
    ここでロックの下で確かめてから印を付ける。
    同じ GPU を奪い合うので、ハイパーパラメータ探索とも同時には回さない。
    """
    from .state import _JOB_START_LOCK, _get_tune_shared

    state, lock = _get_train_shared()
    tune_state, tune_lock = _get_tune_shared()
    with _JOB_START_LOCK:
        with tune_lock:
            if tune_state["running"]:
                return False, "ハイパーパラメータ探索が動いています。終わってから学習してください。"
        with lock:
            if state["running"]:
                return False, "すでに学習が動いています。"
            state.update({"log": [], "progress": 0, "running": True, "error": None,
                          "model_path": None, "metrics_history": [],
                          "stop_requested": False})

    threading.Thread(
        target=_train_worker,
        args=(data_yaml, model_name, epochs, batch_size,
              project_name, run_name, train_kwargs),
        daemon=True,
    ).start()
    return True, ""
