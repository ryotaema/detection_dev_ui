# =============================================================================
# ハイパーパラメータ探索
#
#   Ultralytics の `model.tune()` は遺伝的アルゴリズムで設定を探す。
#   **1 イテレーション = 学習まるごと 1 回**なので、10 回回せば学習 10 回分の
#   時間がかかる。押してから「8 時間かかる」と気づくのが最悪なので、
#   始める前に見積もりを出すことをこのモジュールの主目的の 1 つにしている。
#
#   既定の探索空間は 26 項目あるが、そのまま振ると回数が足りない。
#   目的別のプリセットを用意し、絞って探せるようにする。
#
#   評価は data.yaml の val で行う（本家の振る舞いのまま）。
#   探索回数が多いと val に過学習した設定が選ばれうるので、
#   最後はテスト用データで確かめるよう UI 側で促す。
# =============================================================================
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from .config import MODELS_DIR

# 目的別の探索空間。範囲は Ultralytics の既定に合わせてある。
# 「何を振るか」を選べることが要点で、既定の 26 項目を全部振ると
# 回数がいくらあっても足りない。
SEARCH_PRESETS: dict[str, dict] = {
    "lr": {
        "label": "🎯 学習率まわり（推奨）",
        "desc": "効果が大きく項目が少ないので、少ない回数でも効きます",
        "space": {
            "lr0": (1e-5, 1e-2),
            "lrf": (0.01, 1.0),
            "momentum": (0.7, 0.98, 0.3),
            "warmup_epochs": (0.0, 5.0),
        },
    },
    "loss": {
        "label": "⚖️ 損失の重み",
        "desc": "位置ズレやクラス間の精度差が気になるときに",
        "space": {
            "box": (1.0, 20.0),
            "cls": (0.1, 4.0),
            "dfl": (0.4, 12.0),
        },
    },
    "aug": {
        "label": "🎨 データ拡張",
        "desc": "データが少なく、過学習ぎみのときに",
        "space": {
            "hsv_h": (0.0, 0.1),
            "hsv_s": (0.0, 0.9),
            "hsv_v": (0.0, 0.9),
            "degrees": (0.0, 45.0),
            "translate": (0.0, 0.9),
            "scale": (0.0, 0.95),
            "fliplr": (0.0, 1.0),
            "mosaic": (0.0, 1.0),
            "mixup": (0.0, 1.0),
        },
    },
    "all": {
        "label": "🔧 すべて（Ultralytics の既定）",
        "desc": "26 項目すべて。時間を潤沢に使えるときだけ",
        "space": None,      # None を渡すと本家の既定が使われる
    },
}


# ---------------------------------------------------------------------------
# 所要時間の見積もり
# ---------------------------------------------------------------------------
def estimate_epoch_seconds(dataset_dir: Optional[Path] = None) -> Optional[float]:
    """過去の学習から 1 エポックあたりの秒数を推定する。

    実測が無ければ None。推測で数字を出すより「分からない」と言うほうがよい。
    """
    best: Optional[float] = None
    if not MODELS_DIR.exists():
        return None

    for run in MODELS_DIR.iterdir():
        if not run.is_dir():
            continue
        csv = run / "results.csv"
        if not csv.exists():
            continue
        try:
            lines = [l for l in csv.read_text().splitlines() if l.strip()]
            if len(lines) < 3:
                continue
            header = [h.strip() for h in lines[0].split(",")]
            if "time" not in header:
                continue
            ti = header.index("time")
            last = float(lines[-1].split(",")[ti])
            epochs = len(lines) - 1
            if epochs > 0 and last > 0:
                per = last / epochs
                # 同じデータセットの実測があればそちらを優先する
                if dataset_dir is not None:
                    from .provenance import read_provenance
                    pv = read_provenance(run) or {}
                    if (pv.get("dataset") or {}).get("name") == Path(dataset_dir).name:
                        return per
                best = per if best is None else min(best, per)
        except Exception:
            continue
    return best


def estimate_tuning(iterations: int, epochs: int,
                    dataset_dir: Optional[Path] = None) -> dict:
    """探索にどれくらいかかるかを見積もる。

    始める前に出すためのもの。分からないときは分からないと返す。
    """
    per_epoch = estimate_epoch_seconds(dataset_dir)
    if per_epoch is None:
        return {"known": False, "per_epoch": None, "per_run": None,
                "total": None, "text": "過去の学習の記録がないため見積もれません"}

    per_run = per_epoch * max(1, epochs)
    total = per_run * max(1, iterations)
    return {
        "known": True,
        "per_epoch": per_epoch,
        "per_run": per_run,
        "total": total,
        "text": (f"1 回の学習 約 {_fmt_dur(per_run)} × {iterations} 回 "
                 f"= 約 {_fmt_dur(total)}"),
    }


def _fmt_dur(sec: float) -> str:
    sec = max(0, int(sec))
    if sec < 90:
        return f"{sec} 秒"
    if sec < 5400:
        return f"{sec / 60:.0f} 分"
    return f"{sec / 3600:.1f} 時間"


# ---------------------------------------------------------------------------
# 結果の読み取り
# ---------------------------------------------------------------------------
def read_tune_results(tune_dir: Path) -> list[dict]:
    """tune_results.csv を読む。1 行 = 1 イテレーション。"""
    csv = Path(tune_dir) / "tune_results.csv"
    if not csv.exists():
        return []
    try:
        lines = [l for l in csv.read_text().splitlines() if l.strip()]
        if len(lines) < 2:
            return []
        header = [h.strip() for h in lines[0].split(",")]
        rows = []
        for i, line in enumerate(lines[1:], 1):
            vals = line.split(",")
            if len(vals) != len(header):
                continue
            row: dict = {"iteration": i}
            for k, v in zip(header, vals):
                try:
                    row[k] = float(v)
                except ValueError:
                    row[k] = v
            rows.append(row)
        return rows
    except Exception:
        return []


def read_best_params(tune_dir: Path) -> dict:
    """best_hyperparameters.yaml を読む"""
    y = Path(tune_dir) / "best_hyperparameters.yaml"
    if not y.exists():
        return {}
    try:
        import yaml
        data = yaml.safe_load(y.read_text()) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def best_of(rows: list[dict]) -> Optional[dict]:
    """fitness が最も高い行"""
    scored = [r for r in rows if isinstance(r.get("fitness"), float)]
    return max(scored, key=lambda r: r["fitness"]) if scored else None


def find_tune_dirs(root: Optional[Path] = None) -> list[Path]:
    """過去の探索結果を新しい順に返す"""
    base = Path(root or MODELS_DIR)
    if not base.exists():
        return []
    dirs = [d for d in base.rglob("tune_results.csv")]
    return sorted((d.parent for d in dirs),
                  key=lambda p: p.stat().st_mtime, reverse=True)


# ---------------------------------------------------------------------------
# 探索ワーカー
# ---------------------------------------------------------------------------
class TuningStopped(Exception):
    """イテレーションの切り目で抜けるための合図"""


def _tune_worker(
    data_yaml: str,
    base_model: str,
    iterations: int,
    epochs: int,
    space: Optional[dict],
    run_name: str,
    extra: Optional[dict] = None,
) -> None:
    """バックグラウンドで探索を回す。

    停止は**イテレーションの切り目**で効く。学習の途中では止めない
    （その回の計算が無駄になるため）。学習の「⏹ 学習を停止」と同じ考え方。
    """
    from .state import _get_tune_shared

    state, lock = _get_tune_shared()

    def log(msg: str) -> None:
        with lock:
            state["log"].append(f"[{time.strftime('%H:%M:%S')}] {msg}")

    def stopped() -> bool:
        with lock:
            return state["stop_requested"]

    try:
        from ultralytics import YOLO

        tune_dir = MODELS_DIR / run_name
        with lock:
            state.update({"running": True, "error": None, "iteration": 0,
                          "total": iterations, "tune_dir": str(tune_dir),
                          "history": [], "best_fitness": None,
                          "best_params": None,
                          "started_at": time.time()})
        log(f"探索を開始します（{iterations} 回 / 各 {epochs} エポック）")
        log(f"振る項目: {', '.join(space) if space else 'Ultralytics の既定（26 項目）'}")

        model = YOLO(base_model)

        # tune() 自身には停止の口が無いので、コールバックで様子を見て
        # イテレーションの切り目に例外で抜ける
        def _on_fit_epoch_end(trainer):
            rows = read_tune_results(tune_dir)
            n = len(rows)
            with lock:
                if n != state["iteration"]:
                    state["iteration"] = n
                    state["history"] = rows
                    b = best_of(rows)
                    if b:
                        state["best_fitness"] = b.get("fitness")
            if n and n != getattr(_on_fit_epoch_end, "_last", -1):
                _on_fit_epoch_end._last = n
                b = best_of(rows)
                log(f"{n} / {iterations} 回目まで完了"
                    + (f"　最良 fitness {b['fitness']:.4f}" if b else ""))

        def _on_train_start(trainer):
            if stopped():
                log("停止の要求を受け付けました。この回で終了します。")
                raise TuningStopped()

        model.add_callback("on_fit_epoch_end", _on_fit_epoch_end)
        model.add_callback("on_train_start", _on_train_start)

        args = dict(
            data=data_yaml, epochs=int(epochs), iterations=int(iterations),
            optimizer="AdamW", plots=False, save=False, val=True,
            project=str(MODELS_DIR), name=run_name, exist_ok=True,
        )
        if space:
            args["space"] = space
        if extra:
            args.update(extra)

        try:
            model.tune(**args)
        except TuningStopped:
            log("停止しました。ここまでの結果は残っています。")
        except Exception as e:
            # tune() は内部で学習を回すので、停止の合図が包まれてくることがある
            if "TuningStopped" in repr(e):
                log("停止しました。ここまでの結果は残っています。")
            else:
                raise

        rows = read_tune_results(tune_dir)
        best = best_of(rows)
        with lock:
            state["history"] = rows
            state["iteration"] = len(rows)
            if best:
                state["best_fitness"] = best.get("fitness")
            state["best_params"] = read_best_params(tune_dir)
        log(f"終了しました（{len(rows)} 回）"
            + (f"　最良 fitness {best['fitness']:.4f}" if best else ""))

    except Exception as e:
        with lock:
            state["error"] = f"{type(e).__name__}: {e}"
        log(f"失敗しました: {e}")
    finally:
        with lock:
            state["running"] = False
            state["stop_requested"] = False


def start_tuning(data_yaml: str, base_model: str, iterations: int, epochs: int,
                 space: Optional[dict], run_name: str,
                 extra: Optional[dict] = None) -> bool:
    """探索をバックグラウンドで始める。既に走っていれば何もしない。"""
    import threading

    from .state import _get_tune_shared

    state, lock = _get_tune_shared()
    with lock:
        if state["running"]:
            return False
        state["log"] = []
        state["running"] = True
        state["stop_requested"] = False

    threading.Thread(
        target=_tune_worker,
        args=(data_yaml, base_model, iterations, epochs, space, run_name, extra),
        daemon=True,
    ).start()
    return True


def request_stop_tuning() -> None:
    from .state import _get_tune_shared

    state, lock = _get_tune_shared()
    with lock:
        state["stop_requested"] = True


def params_to_preset(best: dict, base: Optional[dict] = None) -> dict:
    """探索の結果を、Step3 の学習プリセットの形に直す。

    探索して終わりでは意味がないので、そのまま学習に使えるようにする。
    """
    out = dict(base or {})
    for k, v in (best or {}).items():
        if isinstance(v, (int, float)):
            out[k] = round(float(v), 6)
    return out
