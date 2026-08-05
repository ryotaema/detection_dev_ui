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
#
#   **Ultralytics の Tuner は各イテレーションの学習を subprocess で回す**
#   （8.4.48 の `Tuner.__call__`。dataloader のハングを避けるため）。
#   そのため `model.add_callback()` は**探索中は一切発火しない**。
#   進捗も停止もコールバックでは取れないので、
#     - 進捗 … 親プロセスで動く `Tuner._mutate()` を包んで拾う
#     - イテレーション内 … 学習が書く `results.csv` を読む
#     - 停止 … `_mutate()` の入口で例外を投げる（＝イテレーションの切り目）
#   という作りにしている。
#
#   結果は `tune_results.ndjson`（1 行 1 イテレーションの JSON）。
#   古い版の `tune_results.csv` も読めるようにしてある。
# =============================================================================
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from .config import MODELS_DIR

# 探索の出力先。models/ 直下に置くと、各イテレーションの学習が作る
# train / train2 … がモデル一覧に並んでしまうため、専用の場所に隔離する。
TUNE_PROJECT = MODELS_DIR / ".tuning"

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
    "custom": {
        "label": "✏️ 自分で選ぶ",
        "desc": "振りたい項目を自分で選びます。3〜5 個に絞るのが目安です",
        "space": None,
        "custom": True,     # UI 側で項目を選ばせる
    },
}

# 項目を増やすほど、同じ回数での到達点は落ちる。
# 合成関数での実測（30 シード平均・0〜1・TPE）:
#
#     項目数   10 回   20 回   40 回   80 回
#        4     0.531   0.650   0.834   0.956
#        9     0.125   0.155   0.263   0.463
#       26     0.015   0.018   0.021   0.024
#
# 26 項目は 80 回かけても 4 項目の 10 回に遠く及ばない。
# 1 回が学習まるごと 1 回であることを思えば、絞るのが前提になる。
MANY_PARAMS_WARN = 6


# ---------------------------------------------------------------------------
# 探索の手法
#
#   本家（Ultralytics）は遺伝的アルゴリズム。既定値から始めて、
#   上位の結果を交叉させ **乗算的に** 変異させる（値 × exp(N(0,σ))）。
#   これは 2 つの性質を生む:
#     - 既定値が良いところにある項目では、少ない回数でも堅実に効く
#     - **値が 0 の項目はほぼ動けない**（0 × 何か = 0）。
#       `degrees` / `mixup` は既定が 0.0 なので、実測では許容範囲 0〜45 に対して
#       0〜0.025 しか探れなかった。データ拡張を振るときに効かない
#
#   TPE（Optuna）は過去の結果から「良い値の分布」を作って次を選ぶ。
#   0 も端も関係なく探せる代わりに、序盤は当てずっぽうになる。
#
#   合成した目的関数で実測した平均（30 シード / 値は 0〜1）:
#
#       予算    空間          GA      TPE
#       10 回   データ拡張    0.492   0.460
#       20 回   データ拡張    0.535   0.739
#       10 回   学習率まわり  0.566   0.514
#       20 回   学習率まわり  0.666   0.752
#
#   → **10 回程度までは GA、20 回以上なら TPE**。UI でも回数に応じて薦める。
# ---------------------------------------------------------------------------
METHOD_GA = "ga"
METHOD_TPE = "tpe"

SEARCH_METHODS: dict[str, dict] = {
    METHOD_GA: {
        "label": "🧬 遺伝的アルゴリズム（Ultralytics 本家）",
        "desc": "既定値から少しずつ変えていきます。回数が少ないときに堅実です。",
        "caveat": "値が 0 の項目（degrees / mixup など）はほとんど動きません。",
    },
    METHOD_TPE: {
        "label": "📈 TPE（Optuna）",
        "desc": "過去の結果から次の候補を選びます。20 回以上ならこちらが有利です。",
        "caveat": "序盤の数回は当てずっぽうなので、回数が少ないと不利です。",
    },
}

# ここを境に薦める手法を変える（上の実測から）
TPE_RECOMMEND_FROM = 15


def optuna_available() -> bool:
    try:
        import optuna  # noqa: F401
        return True
    except ImportError:
        return False


def recommend_method(iterations: int) -> str:
    """回数から手法を薦める"""
    if iterations >= TPE_RECOMMEND_FROM and optuna_available():
        return METHOD_TPE
    return METHOD_GA


def default_space() -> dict:
    """Ultralytics の既定探索空間（26 項目）を実行時に取り出す。

    書き写すと本家の更新とずれるので、ソースから読む。
    「すべて」を選んだときにも項目を固定できるようにするために要る。
    """
    import inspect
    import re

    try:
        from ultralytics.engine.tuner import Tuner
        src = inspect.getsource(Tuner.__init__)
        m = re.search(r'self\.space = args\.pop\("space", None\) or \{(.*?)\n\s*\}',
                      src, re.S)
        if not m:
            return {}
        out = {}
        for k, vals in re.findall(r'"(\w+)":\s*\(([^)]*)\)', m.group(1)):
            nums = [float(x) for x in vals.split(",")]
            out[k] = tuple(nums)
        return out
    except Exception:
        return {}


def is_log_scale(lo: float, hi: float) -> bool:
    """対数で振るべき範囲か。

    学習率のように桁で効く項目を線形に振ると、大きいほうばかり試すことになる。
    """
    return lo > 0 and hi / lo >= 100


def apply_pins(space: Optional[dict], pinned: Optional[dict]) -> tuple[dict, dict]:
    """固定した項目を探索空間から外す。

    固定値は学習の引数としてそのまま渡す（`{**vars(self.args), **mutated_hyp}`
    の前段に入るため、探索対象から外すだけで固定される）。
    戻り値は (振る空間, 学習に渡す固定値)。
    """
    base = dict(space) if space else default_space()
    pins = {k: float(v) for k, v in (pinned or {}).items() if k in base}
    return ({k: v for k, v in base.items() if k not in pins}, pins)


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
        if not run.is_dir() or run.name.startswith("."):
            continue   # .tuning（探索の作業場）は学習の記録ではない
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


def baseline_hyp() -> dict:
    """Ultralytics の既定値。

    「いま試している設定」を出すとき、数字だけでは高いのか低いのか分からない。
    既定からどれだけ振れているかを添えるために使う。
    """
    try:
        from ultralytics.cfg import DEFAULT_CFG_DICT
        return {k: v for k, v in DEFAULT_CFG_DICT.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# 結果の読み取り
# ---------------------------------------------------------------------------
def read_tune_results(tune_dir: Path) -> list[dict]:
    """探索の結果を 1 行 1 イテレーションで返す。

    8.4.x は `tune_results.ndjson`（1 行 1 JSON）。
    古い版の `tune_results.csv` も読めるようにしてある。
    振ったパラメータは列として平らに広げる（表にそのまま出せるように）。
    """
    d = Path(tune_dir)

    nd = d / "tune_results.ndjson"
    if nd.exists():
        rows = []
        try:
            for line in nd.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                row = {"iteration": rec.get("iteration", len(rows) + 1),
                       "fitness": rec.get("fitness")}
                row.update(rec.get("hyperparameters") or {})
                # データセットごとの指標（1 つなら中身をそのまま持ち上げる）
                ds = rec.get("datasets") or {}
                if len(ds) == 1:
                    for k, v in list(ds.values())[0].items():
                        if k != "fitness" and isinstance(v, (int, float)):
                            row.setdefault(k, v)
                sd = rec.get("save_dirs") or {}
                if sd:
                    row["_save_dir"] = list(sd.values())[0]
                rows.append(row)
        except Exception:
            return []
        return rows

    csv = d / "tune_results.csv"
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
    dirs = {f.parent for pat in ("tune_results.ndjson", "tune_results.csv")
            for f in base.rglob(pat)}
    return sorted(dirs, key=lambda p: p.stat().st_mtime, reverse=True)


# ---------------------------------------------------------------------------
# 探索ワーカー
# ---------------------------------------------------------------------------
class TuningStopped(Exception):
    """イテレーションの切り目で抜けるための合図"""


def _read_live_epoch(project: Path) -> Optional[dict]:
    """いま回っている学習の results.csv を読む。

    学習は subprocess なので進捗はファイル越しにしか分からない。
    直近に更新された results.csv を「いま動いているもの」とみなす。
    """
    try:
        csvs = [c for c in Path(project).glob("*/results.csv")]
        if not csvs:
            return None
        csv = max(csvs, key=lambda c: c.stat().st_mtime)
        lines = [l for l in csv.read_text().splitlines() if l.strip()]
        if len(lines) < 2:
            return None
        header = [h.strip() for h in lines[0].split(",")]
        vals = lines[-1].split(",")
        row = {}
        for k, v in zip(header, vals):
            try:
                row[k] = float(v)
            except ValueError:
                row[k] = v
        return {"epoch": int(row.get("epoch") or len(lines) - 1), "row": row}
    except Exception:
        return None


def pick_metrics(row: dict) -> dict:
    """表に出すぶんだけ拾う（列が多いので絞る）"""
    out = {}
    for k, v in (row or {}).items():
        if not isinstance(v, (int, float)):
            continue
        kl = k.lower()
        if "map50-95" in kl or "map50(" in kl or kl.endswith("map50"):
            out[k.split("/")[-1]] = v
        elif "loss" in kl and "val" in kl:
            out[k.split("/")[-1]] = v
    return out


def _make_tpe_sampler(space: dict, iterations: int, defaults: dict):
    """Optuna の TPE を「次の候補を返すだけ」の関数にして返す。

    本家の探索の流れ（subprocess で学習 → ndjson に記録）はそのまま使い、
    **候補の選び方だけ**を差し替える。ask/tell で 1 回ずつやりとりする。
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            # 序盤の当てずっぽうを短くする（回数が少ないため）
            n_startup_trials=max(3, iterations // 5)),
    )
    # 1 回目は既定値から。本家 GA と揃えるため
    if defaults:
        study.enqueue_trial({k: v for k, v in defaults.items() if k in space})

    pending = {"trial": None}

    def suggest(last_fitness: Optional[float]) -> dict:
        if pending["trial"] is not None:
            study.tell(pending["trial"], last_fitness if last_fitness is not None else 0.0)
            pending["trial"] = None
        trial = study.ask()
        out = {}
        for k, bounds in space.items():
            lo, hi = float(bounds[0]), float(bounds[1])
            out[k] = trial.suggest_float(k, lo, hi, log=is_log_scale(lo, hi))
        pending["trial"] = trial
        return out

    return suggest


def _tune_worker(
    data_yaml: str,
    base_model: str,
    iterations: int,
    epochs: int,
    space: Optional[dict],
    run_name: str,
    extra: Optional[dict] = None,
    method: str = METHOD_GA,
    pinned: Optional[dict] = None,
) -> None:
    """バックグラウンドで探索を回す。

    Ultralytics の Tuner は各学習を subprocess で回すので、
    `model.add_callback()` はここでは効かない。
    親プロセスで動く `Tuner._mutate()` を包んで、進捗の取得と停止を行う。
    停止は**イテレーションの切り目**（走っている学習は最後までやる）。
    """
    import threading as _th

    from .state import _get_tune_shared

    state, lock = _get_tune_shared()

    def log(msg: str) -> None:
        with lock:
            state["log"].append(f"[{time.strftime('%H:%M:%S')}] {msg}")
            # 何時間も回るので、際限なく溜めない
            if len(state["log"]) > 400:
                del state["log"][:-400]

    def stopped() -> bool:
        with lock:
            return state["stop_requested"]

    watcher_stop = _th.Event()

    try:
        from ultralytics import YOLO
        from ultralytics.engine.tuner import Tuner

        tune_dir = TUNE_PROJECT / run_name
        with lock:
            state.update({"running": True, "error": None, "iteration": 0,
                          "total": iterations, "tune_dir": str(tune_dir),
                          "history": [], "best_fitness": None,
                          "best_params": None, "current_params": None,
                          "current_epoch": 0, "current_total_epochs": int(epochs),
                          "current_metrics": None, "iter_started_at": None,
                          "started_at": time.time()})
        space, pins = apply_pins(space, pinned)
        if not space:
            raise ValueError("振る項目がありません（すべて固定されています）")
        if method == METHOD_TPE and not optuna_available():
            log("Optuna が入っていないため、遺伝的アルゴリズムで探索します。")
            method = METHOD_GA

        log(f"探索を開始します（{iterations} 回 / 各 {epochs} エポック）")
        log(f"手法: {SEARCH_METHODS[method]['label']}")
        log(f"振る項目({len(space)}): {', '.join(space)}")
        if pins:
            log("固定: " + "　".join(f"{k}={v:g}" for k, v in pins.items()))

        # ── いま回している学習の様子をファイル越しに見る ────────────
        def _watch() -> None:
            while not watcher_stop.wait(3.0):
                live = _read_live_epoch(TUNE_PROJECT)
                if not live:
                    continue
                with lock:
                    state["current_epoch"] = live["epoch"]
                    state["current_metrics"] = pick_metrics(live["row"])

        watcher = _th.Thread(target=_watch, daemon=True)
        watcher.start()

        # ── 候補の選び方（TPE のときだけ差し替える）──────────────────
        _suggest = None
        if method == METHOD_TPE:
            from ultralytics.cfg import DEFAULT_CFG_DICT
            _suggest = _make_tpe_sampler(space, iterations, dict(DEFAULT_CFG_DICT))

        # ── _mutate を包む（ここだけが親プロセスで確実に通る）────────
        _orig_mutate = Tuner._mutate

        def _mutate(self, *a, **kw):
            done = read_tune_results(self.tune_dir)
            with lock:
                state["iteration"] = len(done)
                state["history"] = done
                b = best_of(done)
                if b:
                    state["best_fitness"] = b.get("fitness")

            if stopped():
                log("停止の要求を受け付けました。ここで終了します。")
                raise TuningStopped()

            if _suggest is not None:
                _last_fit = done[-1].get("fitness") if done else None
                hyp = _suggest(_last_fit)
                # 本家と同じく範囲に収める
                hyp = {k: round(min(max(v, space[k][0]), space[k][1]), 5)
                       for k, v in hyp.items()}
                if "close_mosaic" in hyp:
                    hyp["close_mosaic"] = round(hyp["close_mosaic"])
            else:
                hyp = _orig_mutate(self, *a, **kw)
            with lock:
                state["current_params"] = dict(hyp)
                state["current_epoch"] = 0
                state["current_metrics"] = None
                state["iter_started_at"] = time.time()
            log(f"{len(done) + 1} / {iterations} 回目を開始　"
                + "　".join(f"{k}={v:.4g}" for k, v in list(hyp.items())[:6]))
            return hyp

        Tuner._mutate = _mutate
        try:
            model = YOLO(base_model)
            # optimizer / imgsz / batch は**本番の学習と揃える**こと。
            # 揃えないと探索で見つけた値が本番で再現しない
            # （AdamW と SGD では適切な lr0 が一桁ずれる）。
            args = dict(
                data=data_yaml, epochs=int(epochs), iterations=int(iterations),
                plots=False, save=True, val=True,
                project=str(TUNE_PROJECT), name=run_name, exist_ok=True,
            )
            args["space"] = space
            if pins:
                args.update(pins)      # 固定した項目は学習の引数として渡す
            if extra:
                args.update(extra)

            try:
                model.tune(**args)
            except TuningStopped:
                log("停止しました。ここまでの結果は残っています。")
            except Exception as e:
                # tune() の中で包まれてくることがある
                if "TuningStopped" in repr(e):
                    log("停止しました。ここまでの結果は残っています。")
                else:
                    raise
        finally:
            Tuner._mutate = _orig_mutate

        rows = read_tune_results(tune_dir)
        best = best_of(rows)
        with lock:
            state["history"] = rows
            state["iteration"] = len(rows)
            state["current_params"] = None
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
        watcher_stop.set()
        with lock:
            state["running"] = False
            state["stop_requested"] = False


def start_tuning(data_yaml: str, base_model: str, iterations: int, epochs: int,
                 space: Optional[dict], run_name: str,
                 extra: Optional[dict] = None, method: str = METHOD_GA,
                 pinned: Optional[dict] = None) -> bool:
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
        args=(data_yaml, base_model, iterations, epochs, space, run_name,
              extra, method, pinned),
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
