# =============================================================================
# モデルの使い勝手（お気に入り・使用回数・新着）
#
#   モデルが増えると run 名だけでは選べなくなる。
#   さらに厄介なのが「せっかく学習したのに使い忘れる」こと。
#   一覧の下のほうに埋もれて、古いモデルを使い続けてしまう。
#
#   そこで 3 つの手がかりを持たせる:
#     - お気に入り … 人が明示的に印を付けたもの
#     - 使用回数   … 実際に使った回数と最後に使った日時（自動で記録）
#     - 新着       … 作られてから一度も使っていないもの（別枠で知らせる）
#
#   記録は models/.model_prefs.json に置く。
#   モデル本体（.pt）とは分けているので、モデルを消しても壊れない。
# =============================================================================
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import MODELS_DIR

PREFS_PATH = MODELS_DIR / ".model_prefs.json"

# 「新着」とみなす条件: 一度も使っていない、かつ作られてからこの日数以内
NEW_MODEL_DAYS = 30

SORT_OPTIONS = {
    "recommended": "おすすめ順（お気に入り → 精度 → 新しさ）",
    "recent":      "新しい順",
    "used":        "よく使う順",
    "last_used":   "最後に使った順",
    "map":         "精度順（mAP50-95）",
    "name":        "名前順",
}


def _key(model_path) -> str:
    """モデルを指す安定した名前。MODELS_DIR からの相対パス。"""
    p = Path(model_path)
    try:
        return str(p.resolve().relative_to(Path(MODELS_DIR).resolve()))
    except ValueError:
        return str(p)


def load_prefs() -> dict:
    """壊れていても空として扱う（画面を落とさない）"""
    if not PREFS_PATH.exists():
        return {}
    try:
        data = json.loads(PREFS_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_prefs(prefs: dict) -> bool:
    try:
        PREFS_PATH.parent.mkdir(parents=True, exist_ok=True)
        PREFS_PATH.write_text(
            json.dumps(prefs, ensure_ascii=False, indent=2), encoding="utf-8")
        return True
    except Exception:
        return False


def _entry(prefs: dict, model_path) -> dict:
    return prefs.get(_key(model_path)) or {}


def is_favorite(model_path, prefs: Optional[dict] = None) -> bool:
    return bool(_entry(prefs if prefs is not None else load_prefs(),
                       model_path).get("favorite"))


def toggle_favorite(model_path) -> bool:
    """お気に入りを切り替えて、切り替え後の状態を返す"""
    prefs = load_prefs()
    k = _key(model_path)
    e = prefs.setdefault(k, {})
    e["favorite"] = not e.get("favorite", False)
    save_prefs(prefs)
    return e["favorite"]


def record_use(model_path, action: str = "use") -> None:
    """使ったことを記録する。

    推論・評価・自動アノテーションへのデプロイなど、
    「実際に使った」場面から呼ぶ。数えることで、
    どれが現役かが自然に分かるようにする。
    """
    prefs = load_prefs()
    e = prefs.setdefault(_key(model_path), {})
    e["uses"] = int(e.get("uses", 0)) + 1
    e["last_used"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    hist = e.setdefault("actions", {})
    hist[action] = int(hist.get(action, 0)) + 1
    save_prefs(prefs)


def use_count(model_path, prefs: Optional[dict] = None) -> int:
    return int(_entry(prefs if prefs is not None else load_prefs(),
                      model_path).get("uses", 0))


def last_used(model_path, prefs: Optional[dict] = None) -> str:
    return str(_entry(prefs if prefs is not None else load_prefs(),
                      model_path).get("last_used", ""))


def is_unused(model_path, prefs: Optional[dict] = None) -> bool:
    return use_count(model_path, prefs) == 0


def age_days(model_path) -> float:
    try:
        import time
        return (time.time() - Path(model_path).stat().st_mtime) / 86400
    except Exception:
        return 1e9


def is_new(model_path, prefs: Optional[dict] = None,
           days: int = NEW_MODEL_DAYS) -> bool:
    """作ったのにまだ一度も使っていないもの。

    「学習したのに適用し忘れた」を拾うための判定なので、
    使用回数 0 であることが条件。古すぎるものは新着から外す。
    """
    return is_unused(model_path, prefs) and age_days(model_path) <= days


def best_map(model_path) -> Optional[float]:
    """評価済みなら最も良い mAP50-95（classify は top1）を返す"""
    from .evaluation import read_model_evals

    try:
        evals = read_model_evals(model_path) or {}
    except Exception:
        return None
    vals = []
    for r in evals.values():
        if not isinstance(r, dict):
            continue
        v = r.get("map50_95")
        if v is None:
            v = r.get("top1")
        if isinstance(v, (int, float)):
            vals.append(float(v))
    return max(vals) if vals else None


def describe(model_path, prefs: Optional[dict] = None) -> dict:
    """並べ替え・絞り込みに使う情報を 1 か所に集める"""
    from .provenance import read_status

    prefs = prefs if prefs is not None else load_prefs()
    p = Path(model_path)
    run = p.parent.parent if p.parent.name == "weights" else p.parent
    return {
        "path": str(p),
        "key": _key(p),
        "favorite": is_favorite(p, prefs),
        "uses": use_count(p, prefs),
        "last_used": last_used(p, prefs),
        "age_days": age_days(p),
        "is_new": is_new(p, prefs),
        "map": best_map(p),
        "status": read_status(run, "model"),
        "size_mb": (p.stat().st_size / 1024 / 1024) if p.exists() else 0.0,
    }


def sort_models(model_paths, how: str = "recommended",
                prefs: Optional[dict] = None) -> list[dict]:
    """並べ替えて、表示に使う情報つきで返す"""
    prefs = prefs if prefs is not None else load_prefs()
    items = [describe(p, prefs) for p in model_paths]

    if how == "recent":
        items.sort(key=lambda x: x["age_days"])
    elif how == "used":
        items.sort(key=lambda x: (-x["uses"], x["age_days"]))
    elif how == "last_used":
        # 使ったことのないものは後ろへ
        items.sort(key=lambda x: (x["last_used"] == "",
                                  x["last_used"] and -_ts(x["last_used"]) or 0))
    elif how == "map":
        items.sort(key=lambda x: (x["map"] is None, -(x["map"] or 0)))
    elif how == "name":
        items.sort(key=lambda x: x["key"])
    else:
        # おすすめ順: お気に入り → 精度 → 新しさ
        items.sort(key=lambda x: (not x["favorite"],
                                  x["map"] is None, -(x["map"] or 0),
                                  x["age_days"]))
    return items


def _ts(s: str) -> float:
    try:
        return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timestamp()
    except Exception:
        return 0.0


def unused_new_models(model_paths, prefs: Optional[dict] = None,
                      days: int = NEW_MODEL_DAYS) -> list[dict]:
    """作ったのにまだ使っていないモデル。

    一覧の下に埋もれて使い忘れるのを防ぐため、別枠で知らせる用。
    新しい順に返す。
    """
    prefs = prefs if prefs is not None else load_prefs()
    out = [describe(p, prefs) for p in model_paths]
    out = [x for x in out if x["uses"] == 0 and x["age_days"] <= days]
    out.sort(key=lambda x: x["age_days"])
    return out


def prune_prefs(existing_paths) -> int:
    """消えたモデルの記録を落とす。戻り値は落とした件数。"""
    prefs = load_prefs()
    alive = {_key(p) for p in existing_paths}
    dead = [k for k in prefs if k not in alive]
    for k in dead:
        del prefs[k]
    if dead:
        save_prefs(prefs)
    return len(dead)
