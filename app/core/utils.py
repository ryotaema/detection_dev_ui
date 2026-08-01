# =============================================================================
# 他モジュールから広く使う小さなユーティリティ
# =============================================================================
from __future__ import annotations

import os
from pathlib import Path


# ---------------------------------------------------------------------------
# 画像ディレクトリスキャン
# ---------------------------------------------------------------------------
def _find_image_dirs(base_dir: Path, max_depth: int = 4) -> list[Path]:
    """base_dir 以下で画像ファイルが1件以上あるディレクトリを返す。
    シンボリックリンク先も辿る。深さは max_depth で制限。
    """
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    result: list[Path] = []
    base_depth = len(base_dir.parts)
    for root, dirs, files in os.walk(str(base_dir), followlinks=True):
        root_path = Path(root)
        if len(root_path.parts) - base_depth > max_depth:
            dirs.clear()
            continue
        dirs.sort()
        if any(Path(f).suffix.lower() in img_exts for f in files):
            result.append(root_path)
    return result


# ---------------------------------------------------------------------------
# 要確認画像の自動抽出 (アノテーション対象の優先順位付け)
#
#   推論結果を分析し「モデルが自信を持てていない画像」を機械的に拾う。
#   人が全画像を目視して 🚩 を立てる作業を置き換えるためのもの。
# ---------------------------------------------------------------------------
def _iou(a: list[float], b: list[float]) -> float:
    """2つの xyxy ボックスの IoU"""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(ix2 - ix1, 0.0), max(iy2 - iy1, 0.0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(a[2] - a[0], 0.0) * max(a[3] - a[1], 0.0)
    area_b = max(b[2] - b[0], 0.0) * max(b[3] - b[1], 0.0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def slugify_function_name(run_name: str) -> str:
    """モデル run 名を Nuclio 関数名に使える形（英小文字・数字・ハイフン）へ整える"""
    import re

    s = re.sub(r"[^a-z0-9]+", "-", str(run_name).lower()).strip("-")
    return s or "model"


# ---------------------------------------------------------------------------
# 動画推論
# ---------------------------------------------------------------------------
def _box_iou(a: list, b: list) -> float:
    """2つのxyxy形式ボックスのIoUを計算する。"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0
