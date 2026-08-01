# =============================================================================
# データ拡張のプレビュー
#
#   学習時のパラメータ（mosaic / hsv / degrees ...）が画像に何をするのかを
#   学習前に目で確認できるようにする。値の意味が分からないまま回すのを減らすため。
#
#   注意: Ultralytics の内部実装をそのまま再現したものではなく、
#   「そのパラメータが何を変えるか」を掴むための近似的な再現。
#   実際の学習では確率的に適用され、ここでは効果が見えるよう必ず適用する。
# =============================================================================
from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Optional

from .config import IMG_EXTS

# UI に出す拡張の一覧: (パラメータ名, 表示名, 説明)
AUGMENT_ITEMS: list[tuple[str, str, str]] = [
    ("hsv_h",    "色相 (hsv_h)",     "色味をずらす。照明や品種の違いへの耐性がつく"),
    ("hsv_s",    "彩度 (hsv_s)",     "鮮やかさを変える。曇り/晴れの差に強くなる"),
    ("hsv_v",    "明度 (hsv_v)",     "明るさを変える。露出の違いに強くなる"),
    ("degrees",  "回転 (degrees)",   "画像を回す。カメラの傾きに強くなる"),
    ("translate", "平行移動 (translate)", "上下左右にずらす。位置の偏りを減らす"),
    ("scale",    "拡大縮小 (scale)", "大きさを変える。遠近の違いに強くなる"),
    ("shear",    "せん断 (shear)",   "斜めに歪ませる。撮影角度の違いに対応"),
    ("fliplr",   "左右反転 (fliplr)", "左右を入れ替える。最も効きやすい拡張"),
    ("flipud",   "上下反転 (flipud)", "上下を入れ替える。俯瞰撮影でなければ通常は 0"),
    ("mosaic",   "モザイク (mosaic)", "4枚を1枚に合成する。小さい物体の検出に効果的"),
    ("erasing",  "ランダム消去 (erasing)", "一部を隠す。隠れた物体に強くなる"),
]


def list_sample_images(dataset_dir: Path, limit: int = 8) -> list[Path]:
    """データセットからプレビュー用の画像を拾う"""
    ds = Path(dataset_dir)
    found: list[Path] = []
    for base in (ds / "images", ds):
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                found.append(p)
                if len(found) >= limit:
                    return found
    return found


def _apply_hsv(img, h_gain: float, s_gain: float, v_gain: float, rng: random.Random):
    import cv2
    import numpy as np

    if h_gain == 0 and s_gain == 0 and v_gain == 0:
        return img
    r = np.array([rng.uniform(-1, 1) * h_gain + 1,
                  rng.uniform(-1, 1) * s_gain + 1,
                  rng.uniform(-1, 1) * v_gain + 1])
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype
    x = np.arange(0, 256, dtype=r.dtype)
    lut_h = ((x * r[0]) % 180).astype(dtype)
    lut_s = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_v = np.clip(x * r[2], 0, 255).astype(dtype)
    merged = cv2.merge((cv2.LUT(hue, lut_h), cv2.LUT(sat, lut_s), cv2.LUT(val, lut_v)))
    return cv2.cvtColor(merged, cv2.COLOR_HSV2BGR)


def _apply_affine(img, degrees: float, translate: float, scale: float,
                  shear: float, rng: random.Random):
    import cv2
    import numpy as np

    h, w = img.shape[:2]
    if degrees == 0 and translate == 0 and scale == 0 and shear == 0:
        return img

    a = rng.uniform(-degrees, degrees) if degrees else 0.0
    s = 1 + rng.uniform(-scale, scale) if scale else 1.0
    R = cv2.getRotationMatrix2D((w / 2, h / 2), a, s)

    if shear:
        S = np.eye(3, dtype=np.float32)
        S[0, 1] = math.tan(math.radians(rng.uniform(-shear, shear)))
        S[1, 0] = math.tan(math.radians(rng.uniform(-shear, shear)))
        M = (np.vstack([R, [0, 0, 1]]).astype(np.float32) @ S)[:2]
    else:
        M = R

    if translate:
        M[0, 2] += rng.uniform(-translate, translate) * w
        M[1, 2] += rng.uniform(-translate, translate) * h

    return cv2.warpAffine(img, M, (w, h), borderValue=(114, 114, 114))


def _apply_erasing(img, prob: float, rng: random.Random):
    if prob <= 0:
        return img
    h, w = img.shape[:2]
    area = h * w * rng.uniform(0.02, 0.15)
    ratio = rng.uniform(0.3, 3.0)
    eh = min(int(round(math.sqrt(area * ratio))), h - 1)
    ew = min(int(round(math.sqrt(area / ratio))), w - 1)
    if eh <= 0 or ew <= 0:
        return img
    y = rng.randint(0, h - eh)
    x = rng.randint(0, w - ew)
    out = img.copy()
    out[y:y + eh, x:x + ew] = [rng.randint(0, 255) for _ in range(3)]
    return out


def _make_mosaic(images: list, size: int, rng: random.Random):
    """4枚を1枚に合成する（Ultralytics の mosaic を簡略化したもの）"""
    import cv2
    import numpy as np

    canvas = np.full((size, size, 3), 114, np.uint8)
    half = size // 2
    positions = [(0, 0), (half, 0), (0, half), (half, half)]
    for (x, y), im in zip(positions, images):
        tile = cv2.resize(im, (half, half))
        canvas[y:y + half, x:x + half] = tile
    return canvas


def build_augment_preview(
    image_paths: list[Path],
    params: dict,
    size: int = 480,
    seed: int = 0,
    n_variants: int = 3,
):
    """元画像と、拡張を適用した画像を返す。

    Returns: (元画像(RGB), [(ラベル, 画像(RGB)), ...])
    失敗した場合は (None, []) を返す。
    """
    import cv2

    paths = [Path(p) for p in image_paths if Path(p).exists()]
    if not paths:
        return None, []

    base = cv2.imread(str(paths[0]))
    if base is None:
        return None, []
    base = cv2.resize(base, (size, size))

    variants: list[tuple[str, object]] = []
    for i in range(max(1, n_variants)):
        rng = random.Random(seed + i)
        img = base.copy()

        # mosaic は確率なので、有効なら 4 枚合成した結果を土台にする
        if params.get("mosaic", 0) > 0 and len(paths) >= 2:
            tiles = []
            for k in range(4):
                p = paths[(i + k) % len(paths)]
                t = cv2.imread(str(p))
                tiles.append(cv2.resize(t, (size, size)) if t is not None else base)
            img = _make_mosaic(tiles, size, rng)

        img = _apply_hsv(img, params.get("hsv_h", 0), params.get("hsv_s", 0),
                         params.get("hsv_v", 0), rng)
        img = _apply_affine(img, params.get("degrees", 0), params.get("translate", 0),
                            params.get("scale", 0), params.get("shear", 0), rng)
        if params.get("fliplr", 0) and rng.random() < params["fliplr"]:
            img = cv2.flip(img, 1)
        if params.get("flipud", 0) and rng.random() < params["flipud"]:
            img = cv2.flip(img, 0)
        if params.get("erasing", 0) and rng.random() < params["erasing"]:
            img = _apply_erasing(img, params["erasing"], rng)

        variants.append((f"パターン {i + 1}", cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))

    return cv2.cvtColor(base, cv2.COLOR_BGR2RGB), variants


def describe_augment(params: dict) -> list[tuple[str, str, str]]:
    """有効になっている拡張の一覧を (表示名, 値, 説明) で返す"""
    out = []
    for key, label, desc in AUGMENT_ITEMS:
        v = params.get(key, 0)
        if v:
            out.append((label, str(v), desc))
    return out
