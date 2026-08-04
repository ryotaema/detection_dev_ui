# =============================================================================
# クロップ生成（2段階検出のための切り出し）
#
#   BBOX で対象を見つけ → その周辺を切り出し → セグメンテーションで細部を取る、
#   という 2 段構えのための前段。
#
#   ★ make_crop() は実機（crop_for_runtime）と**同一の実装を使う**こと ★
#     学習時と実機で切り出し規則がずれると精度が壊れる。
#     この関数は画像配列と bbox だけを受け取り、ファイルにも Streamlit にも
#     依存しない。実機側はこれを import して target_scale を渡すだけでよい。
#
#   アノテーション用と実機用の違いはラッパだけ:
#     アノテーション用 … 検出した対象ごとにループ + 画像とサイドカー json を保存
#     実機用           … 対象 1 個 + メモリ上のテンソル + 座標変換情報を返す
#
#   仕様: docs/spec_crop_for_annotation.md / docs/spec_crop_for_runtime.md
# =============================================================================
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import IMG_EXTS

SCHEMA_VERSION = "0.1"

SCALE_BASIS = {
    "long_side": "bbox の長辺",
    "diagonal":  "bbox の対角線",
}
PAD_MODES = {
    "reflect":  "画像を鏡写しにして埋める（境目が目立ちにくい）",
    "constant": "単色で埋める",
}


# ---------------------------------------------------------------------------
# コア（実機と共通。ここだけは絶対に分岐させない）
# ---------------------------------------------------------------------------
def make_crop(
    img,
    bbox_xyxy,
    scale: float = 2.0,
    scale_basis: str = "long_side",
    square: bool = True,
    pad_mode: str = "reflect",
    pad_value: int = 0,
    out_size: int = 1024,
    max_upscale: float = 1.5,
):
    """bbox の周りを切り出して out_size にそろえる。

    max_upscale <= 0 なら拡大を制限しない（必ず out_size で出す）。
    学習で使うサイズをそろえたいときはこちら。
    小さな対象は引き伸ばされるが、実効解像度が上がるわけではない。

    戻り値: (クロップ画像, crop_geometry)

    crop_geometry にはマスクを元画像へ戻すのに必要な値をすべて入れる。
    ここが欠けると、セグメンテーション結果を元座標へ復元できない。

    リサイズは出力段の 1 回だけ。中間リサイズを挟むと細部が劣化する。
    """
    import cv2
    import numpy as np

    h, w = img.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    bw, bh = max(1e-6, x2 - x1), max(1e-6, y2 - y1)

    # ① 基準長 → 切り出しサイズ
    if scale_basis == "diagonal":
        base = (bw ** 2 + bh ** 2) ** 0.5
    else:
        base = max(bw, bh)
    crop_w = crop_h = base * scale
    if not square:
        crop_w, crop_h = bw * scale, bh * scale

    # ② bbox の中心を中心に矩形を取る（理論矩形。画像外にはみ出しうる）
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    rx0 = cx - crop_w / 2.0
    ry0 = cy - crop_h / 2.0
    rx1 = rx0 + crop_w
    ry1 = ry0 + crop_h

    ix0, iy0 = int(round(rx0)), int(round(ry0))
    ix1, iy1 = int(round(rx1)), int(round(ry1))

    # ③ はみ出したぶんはパディングで補う（画像の端にある対象を欠けさせない）
    pad_left   = max(0, -ix0)
    pad_top    = max(0, -iy0)
    pad_right  = max(0, ix1 - w)
    pad_bottom = max(0, iy1 - h)

    sx0, sy0 = max(0, ix0), max(0, iy0)
    sx1, sy1 = min(w, ix1), min(h, iy1)
    patch = img[sy0:sy1, sx0:sx1]

    if pad_left or pad_top or pad_right or pad_bottom:
        border = (cv2.BORDER_REFLECT_101 if pad_mode == "reflect"
                  else cv2.BORDER_CONSTANT)
        # reflect は元領域より広い量を反射できないので、足りなければ単色に落とす
        ph, pw = patch.shape[:2]
        if border == cv2.BORDER_REFLECT_101 and (
                pad_left >= pw or pad_right >= pw
                or pad_top >= ph or pad_bottom >= ph):
            border = cv2.BORDER_CONSTANT
        patch = cv2.copyMakeBorder(
            patch, pad_top, pad_bottom, pad_left, pad_right, border,
            value=(int(pad_value),) * 3 if border == cv2.BORDER_CONSTANT else None)

    if patch.size == 0:
        patch = np.full((1, 1, img.shape[2] if img.ndim == 3 else 1),
                        int(pad_value), dtype=img.dtype)

    # ④ 出力サイズへ。小さな切り出しを無理に拡大しない（水増しを防ぐ）
    ph, pw = patch.shape[:2]
    src_side = max(ph, pw)
    want_ratio = out_size / max(1e-6, src_side)
    # max_upscale <= 0 は「上限なし」。学習で使うサイズをそろえたいときに使う
    unlimited = float(max_upscale) <= 0
    upscaled = (not unlimited) and want_ratio > max_upscale
    ratio = want_ratio if unlimited else min(want_ratio, float(max_upscale))
    final = max(1, int(round(src_side * ratio)))

    resized = cv2.resize(patch, (max(1, int(round(pw * ratio))),
                                 max(1, int(round(ph * ratio)))),
                         interpolation=(cv2.INTER_AREA if ratio < 1
                                        else cv2.INTER_LINEAR))

    geom = {
        "scale": float(scale),
        "scale_basis": scale_basis,
        "square": bool(square),
        # パディング前の理論矩形（元画像座標）。復元の基準になる
        "crop_rect_in_source": [int(ix0), int(iy0),
                                int(ix1 - ix0), int(iy1 - iy0)],
        "padding": {"top": pad_top, "bottom": pad_bottom,
                    "left": pad_left, "right": pad_right},
        "pad_mode": pad_mode,
        "out_size": int(out_size),
        "resize_ratio": float(ratio),
        "max_upscale": float(max_upscale),
        "max_upscale_applied": bool(upscaled),
        "output_size": [int(resized.shape[1]), int(resized.shape[0])],
    }
    return resized, geom


def crop_to_source(x: float, y: float, geom: dict) -> tuple:
    """クロップ座標 → 元画像座標。セグメンテーション結果を戻すのに使う。"""
    r = geom["resize_ratio"]
    x0, y0, _, _ = geom["crop_rect_in_source"]
    return (x0 + x / r, y0 + y / r)


def source_to_crop(x: float, y: float, geom: dict) -> tuple:
    """元画像座標 → クロップ座標"""
    r = geom["resize_ratio"]
    x0, y0, _, _ = geom["crop_rect_in_source"]
    return ((x - x0) * r, (y - y0) * r)


def target_rect_in_crop(geom: dict, target_scale: float) -> list:
    """学習用に切り直すための内側 target 矩形（クロップ座標系）。

    アノテーション用は大きめ（annotation_scale）に切っておき、
    学習・実機で使う倍率（target_scale）の範囲をここで示す。
    """
    _, _, cw, ch = geom["crop_rect_in_source"]
    r = geom["resize_ratio"]
    shrink = target_scale / max(1e-6, geom["scale"])
    tw, th = cw * shrink * r, ch * shrink * r
    ow, oh = geom["output_size"]
    return [round((ow - tw) / 2, 2), round((oh - th) / 2, 2),
            round(tw, 2), round(th, 2)]


# ---------------------------------------------------------------------------
# アノテーション用ラッパ
# ---------------------------------------------------------------------------
def _sha1(path: Path, limit: int = 8 * 1024 * 1024) -> str:
    """ファイルの照合用ハッシュ。大きい画像でも頭だけで足りる。"""
    h = hashlib.sha1()
    try:
        with open(path, "rb") as f:
            h.update(f.read(limit))
    except Exception:
        return ""
    return h.hexdigest()


def dedup_detections(dets: list[dict], center_dist: float) -> tuple:
    """bbox 中心が近すぎるものを間引く。0 なら何もしない。

    アノテーション用は原則 0（検出した対象をすべて出す）。
    密集しすぎて重複クロップが増えるときだけ使う。
    """
    if center_dist <= 0 or len(dets) < 2:
        return list(dets), []

    def _c(d):
        x1, y1, x2, y2 = d["bbox_xyxy"]
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def _diag(d):
        x1, y1, x2, y2 = d["bbox_xyxy"]
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

    # 信頼度の高い順に見て、近すぎるものを落とす（決定的な順序にする）
    order = sorted(dets, key=lambda d: (-d.get("confidence", 0.0),
                                        d["bbox_xyxy"]))
    kept, dropped = [], []
    for d in order:
        cx, cy = _c(d)
        near = False
        for k in kept:
            kx, ky = _c(k)
            thr = (_diag(d) + _diag(k)) / 2 * center_dist
            if ((cx - kx) ** 2 + (cy - ky) ** 2) ** 0.5 < thr:
                near = True
                break
        (dropped if near else kept).append(d)
    return kept, dropped


def generate_crops(
    detections_by_image: dict,
    out_dir: Path,
    annotation_scale: float = 2.0,
    target_scale: float = 1.5,
    scale_basis: str = "long_side",
    square: bool = True,
    pad_mode: str = "reflect",
    pad_value: int = 0,
    out_size: int = 1024,
    max_upscale: float = 1.5,
    dedup_center_dist: float = 0.0,
    out_format: str = "png",
    jpg_quality: int = 95,
    save_rejected: bool = False,
    model_info: Optional[dict] = None,
    name_prefix: str = "obj",
    on_progress=None,
) -> dict:
    """検出した対象ごとに 1 枚のクロップと、サイドカー json を書き出す。

    detections_by_image: {画像パス: [{"bbox_xyxy": [...], "confidence": f,
                                      "label": str, "class_id": int}, ...]}

    name_prefix: 出力ファイル名に使う語。対象に合わせて変えられる
                 （例: "fruit" にすると IMG_0031_fruit00.png になる）

    情報量とトレーサビリティを優先する（速度は求めない）。
    後から再生成・座標復元・モデル更新ができるよう、メタは多めに持たせる。
    """
    import cv2

    out = Path(out_dir)
    result = {
        "ok": False, "error": "", "out_dir": str(out),
        "images": 0, "crops": 0, "rejected": 0,
        "upscale_limited": 0, "skipped": [], "errors": [],
    }
    if out.exists() and any(out.iterdir()):
        result["error"] = f"出力先が既にあります: {out.name}（別の名前にしてください）"
        return result

    targets = [(Path(p), d) for p, d in detections_by_image.items() if d]
    if not targets:
        result["error"] = "対象となる検出がありませんでした"
        return result

    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "meta").mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest.jsonl"
    ext = "jpg" if out_format == "jpg" else "png"

    with open(manifest, "w", encoding="utf-8") as mf:
        for n, (img_path, dets) in enumerate(sorted(targets), 1):
            if on_progress:
                on_progress(n, len(targets))

            img = cv2.imread(str(img_path))
            if img is None:
                result["skipped"].append((str(img_path), "画像を読めません"))
                continue
            src_h, src_w = img.shape[:2]
            src_sha1 = _sha1(img_path)

            kept, dropped = dedup_detections(dets, dedup_center_dist)
            result["rejected"] += len(dropped)

            for i, det in enumerate(kept):
                try:
                    crop, geom = make_crop(
                        img, det["bbox_xyxy"], scale=annotation_scale,
                        scale_basis=scale_basis, square=square,
                        pad_mode=pad_mode, pad_value=pad_value,
                        out_size=out_size, max_upscale=max_upscale)
                except Exception as e:
                    result["errors"].append((str(img_path), str(e)))
                    continue

                if geom["max_upscale_applied"]:
                    result["upscale_limited"] += 1

                stem = f"{img_path.stem}_{name_prefix}{i:02d}"
                img_rel = f"images/{stem}.{ext}"
                meta_rel = f"meta/{stem}.json"

                params = ([cv2.IMWRITE_JPEG_QUALITY, int(jpg_quality)]
                          if ext == "jpg" else [])
                if not cv2.imwrite(str(out / img_rel), crop, params):
                    result["errors"].append((str(img_path), "クロップを書けません"))
                    continue

                x1, y1, x2, y2 = [float(v) for v in det["bbox_xyxy"]]
                meta = {
                    "schema_version": SCHEMA_VERSION,
                    "data_type": "object_crop",
                    "annotation_status": "raw",
                    "annotation_history": [{
                        "status": "raw", "tool": "detection_dev_ui/crop",
                        "version": SCHEMA_VERSION,
                        "timestamp": datetime.now().astimezone().isoformat(),
                    }],
                    "source_image": {
                        "path": str(img_path), "sha1": src_sha1,
                        "width": src_w, "height": src_h,
                    },
                    "bbox_model": dict(model_info or {}),
                    "target_object": {
                        "index": i,
                        "bbox_xywh": [x1, y1, x2 - x1, y2 - y1],
                        "confidence": float(det.get("confidence", 0.0)),
                        "class_id": det.get("class_id"),
                        "label": det.get("label", ""),
                    },
                    "crop_geometry": {
                        **geom,
                        "annotation_scale": float(annotation_scale),
                        "target_scale": float(target_scale),
                        # 学習用に切り直すための内側矩形
                        "target_rect_in_crop": target_rect_in_crop(
                            geom, target_scale),
                    },
                    # 同じクロップに写り込む他の対象（どれが主対象かの判別に使う）
                    "others_in_crop": [
                        {"bbox_xywh_in_source": [
                            o["bbox_xyxy"][0], o["bbox_xyxy"][1],
                            o["bbox_xyxy"][2] - o["bbox_xyxy"][0],
                            o["bbox_xyxy"][3] - o["bbox_xyxy"][1]],
                         "confidence": float(o.get("confidence", 0.0))}
                        for j, o in enumerate(kept) if j != i
                        and _overlaps(o["bbox_xyxy"], geom)
                    ],
                    "rejected_detections": (
                        [{"bbox_xyxy": d["bbox_xyxy"],
                          "confidence": float(d.get("confidence", 0.0)),
                          "reason": "dedup"} for d in dropped]
                        if save_rejected else []),
                }
                (out / meta_rel).write_text(
                    json.dumps(meta, ensure_ascii=False, indent=2),
                    encoding="utf-8")

                mf.write(json.dumps({
                    "crop_image": img_rel, "meta": meta_rel,
                    "data_type": "object_crop", "annotation_status": "raw",
                    "source_image": str(img_path),
                    "confidence": float(det.get("confidence", 0.0)),
                    "source_width": src_w, "source_height": src_h,
                }, ensure_ascii=False) + "\n")
                result["crops"] += 1

            result["images"] += 1

    result["ok"] = result["crops"] > 0 and not result["errors"]
    if result["crops"] == 0:
        result["error"] = "クロップを 1 枚も作れませんでした"
    elif result["errors"]:
        result["error"] = f"{len(result['errors'])} 件で問題が起きました"
    return result


def _overlaps(bbox, geom) -> bool:
    """その bbox がクロップ矩形に掛かっているか"""
    x0, y0, cw, ch = geom["crop_rect_in_source"]
    return not (bbox[2] < x0 or bbox[0] > x0 + cw
                or bbox[3] < y0 or bbox[1] > y0 + ch)


# ---------------------------------------------------------------------------
# 背景タイル
# ---------------------------------------------------------------------------
def generate_background_tiles(
    image_paths,
    out_dir: Path,
    tile_size: int = 1024,
    tile_overlap: float = 0.0,
    out_format: str = "png",
    on_progress=None,
) -> dict:
    """対象の検出されない画像をタイルに分割する。

    中身が薄いタイルを機械的に捨てることはしない。
    「何も無い」の判定は対象領域によって大きく変わり、誤ると必要な背景を失う。
    全タイルを出し、採否は人が決める。
    """
    import cv2

    out = Path(out_dir)
    result = {"ok": False, "error": "", "tiles": 0, "images": 0,
              "skipped": [], "out_dir": str(out)}

    paths = [Path(p) for p in image_paths]
    if not paths:
        result["error"] = "対象の画像がありません"
        return result

    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "meta").mkdir(parents=True, exist_ok=True)
    ext = "jpg" if out_format == "jpg" else "png"
    step = max(1, int(tile_size * (1.0 - min(0.5, max(0.0, tile_overlap)))))

    for n, p in enumerate(paths, 1):
        if on_progress:
            on_progress(n, len(paths))
        img = cv2.imread(str(p))
        if img is None:
            result["skipped"].append((str(p), "画像を読めません"))
            continue
        h, w = img.shape[:2]

        for r, y in enumerate(range(0, max(1, h - 1), step)):
            for c, x in enumerate(range(0, max(1, w - 1), step)):
                y2, x2 = min(h, y + tile_size), min(w, x + tile_size)
                tile = img[y:y2, x:x2]
                if tile.shape[0] < tile_size // 2 or tile.shape[1] < tile_size // 2:
                    continue          # 端の細切れは捨てる
                stem = f"{p.stem}_tile_r{r}_c{c}"
                cv2.imwrite(str(out / "images" / f"{stem}.{ext}"), tile)
                (out / "meta" / f"{stem}.json").write_text(json.dumps({
                    "schema_version": SCHEMA_VERSION,
                    "data_type": "background_tile",
                    "annotation_status": "raw",
                    "source_image": {"path": str(p), "width": w, "height": h},
                    "tile": {"row": r, "col": c, "x": x, "y": y,
                             "width": int(x2 - x), "height": int(y2 - y),
                             "tile_size": tile_size, "overlap": tile_overlap},
                }, ensure_ascii=False, indent=2), encoding="utf-8")
                result["tiles"] += 1
        result["images"] += 1

    result["ok"] = result["tiles"] > 0
    if not result["tiles"]:
        result["error"] = "タイルを 1 枚も作れませんでした"
    return result


def read_manifest(out_dir: Path) -> list[dict]:
    """manifest.jsonl を読む（一覧・集計用）"""
    mf = Path(out_dir) / "manifest.jsonl"
    if not mf.exists():
        return []
    rows = []
    for line in mf.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows
