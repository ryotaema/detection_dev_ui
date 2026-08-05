# =============================================================================
# 精査の記録（どの画像を人が確認したか）
#
#   自動アノテーションを入れたデータは「BBOX は付いているが人の目が入っていない」
#   状態から始まる。そこから精査済みへ移すには、どこまで見たかが分かる必要がある。
#
#   これまで再アノテーションのフラグは `st.session_state` にしか無く、
#   **ブラウザを再読み込みすると消えていた**。5000 枚を見る作業では致命的なので、
#   データセットの中にファイルとして残す。
#
#   記録は `data/<dataset>/.review_state.json`。
#   データセットと一緒に移動・持ち出しできるのが利点。
# =============================================================================
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import DATA_DIR, IMG_EXTS

REVIEW_FILE = ".review_state.json"


def review_path(dataset_dir) -> Path:
    return Path(dataset_dir) / REVIEW_FILE


def load_review(dataset_dir) -> dict:
    """壊れていても空として扱う（作業を止めない）"""
    p = review_path(dataset_dir)
    if not p.exists():
        return {"reviewed": {}, "flagged": {}}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError
        return {
            "reviewed": dict(data.get("reviewed") or {}),
            "flagged": dict(data.get("flagged") or {}),
        }
    except Exception:
        return {"reviewed": {}, "flagged": {}}


def save_review(dataset_dir, state: dict) -> bool:
    try:
        p = review_path(dataset_dir)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(state, ensure_ascii=False, indent=2),
                     encoding="utf-8")
        return True
    except Exception:
        return False


def dataset_of_image(image_path) -> Optional[Path]:
    """その画像がどのデータセットに属するかを返す。

    `data/` の外にある画像（アップロードした一時ファイルなど）は None。
    記録の置き場が決まらないため。
    """
    try:
        rel = Path(image_path).resolve().relative_to(Path(DATA_DIR).resolve())
    except (ValueError, OSError):
        return None
    return (Path(DATA_DIR) / rel.parts[0]) if rel.parts else None


def mark(dataset_dir, image_name: str, *, flagged: bool) -> bool:
    """確認したことを記録する。

    flagged=True なら「要再アノテーション」、False なら「これでよい」。
    どちらの場合も**確認済み**として数える（見たことに変わりはない）。
    """
    state = load_review(dataset_dir)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    state["reviewed"][image_name] = now
    if flagged:
        state["flagged"][image_name] = now
    else:
        state["flagged"].pop(image_name, None)
    return save_review(dataset_dir, state)


def unmark(dataset_dir, image_name: str) -> bool:
    """確認の記録そのものを取り消す（見なかったことにする）"""
    state = load_review(dataset_dir)
    state["reviewed"].pop(image_name, None)
    state["flagged"].pop(image_name, None)
    return save_review(dataset_dir, state)


def is_flagged(dataset_dir, image_name: str) -> bool:
    return image_name in load_review(dataset_dir).get("flagged", {})


def is_reviewed(dataset_dir, image_name: str) -> bool:
    return image_name in load_review(dataset_dir).get("reviewed", {})


def flagged_images(dataset_dir) -> list[str]:
    return sorted(load_review(dataset_dir).get("flagged", {}))


def count_images(dataset_dir) -> int:
    d = Path(dataset_dir)
    if not d.exists():
        return 0
    return sum(1 for p in d.rglob("*")
               if p.is_file() and p.suffix.lower() in IMG_EXTS
               and "_backup_original" not in p.parts)


def review_progress(dataset_dir) -> dict:
    """精査がどこまで進んだかを返す。

    total は画像の総数。reviewed のうち flagged は「直しが要る」もの。
    """
    state = load_review(dataset_dir)
    total = count_images(dataset_dir)
    reviewed = len(state.get("reviewed", {}))
    flagged = len(state.get("flagged", {}))
    return {
        "total": total,
        "reviewed": reviewed,
        "flagged": flagged,
        "ok": max(0, reviewed - flagged),
        "remaining": max(0, total - reviewed),
        "ratio": (reviewed / total) if total else 0.0,
        "done": total > 0 and reviewed >= total,
    }


def progress_label(dataset_dir) -> str:
    """カードに出す 1 行（まだ何も見ていなければ空）"""
    p = review_progress(dataset_dir)
    if p["reviewed"] == 0:
        return ""
    s = f"👀 精査 {p['reviewed']} / {p['total']} 枚（{p['ratio'] * 100:.0f}%）"
    if p["flagged"]:
        s += f"　🚩 要修正 {p['flagged']} 枚"
    if p["done"]:
        s += "　✅ ひととおり確認済み"
    return s


def sync_from_names(dataset_dir, flagged_names, reviewed_names=None) -> bool:
    """画面側で持っている集合を、そのまま記録に反映する。

    セッションの集合と食い違ったままにしないための入口。
    """
    state = load_review(dataset_dir)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for n in (reviewed_names or flagged_names):
        state["reviewed"].setdefault(n, now)
    for n in flagged_names:
        state["reviewed"].setdefault(n, now)
        state["flagged"].setdefault(n, now)
    return save_review(dataset_dir, state)
