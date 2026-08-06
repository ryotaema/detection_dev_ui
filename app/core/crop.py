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


# 切り出しの既定サイズ。
#
#   **UI と実機で必ず同じ値にすること。** 学習データを 512 で作ったのに
#   実機が 1024 で切り出すと、対象の写る大きさが変わって精度が落ちる。
#   以前は UI が 512、この関数の既定が 1024 で食い違っていた。
#
#   512 なのは実測から。ある実データ（640×480・bbox 長辺の中央値 92px）では
#   1024 にすると ×5.6 の引き伸ばしになり、情報量が増えないまま
#   ファイルだけ大きくなる。データセットによって対象の写る大きさが違うので、
#   UI は検出結果から実際の倍率を出す（この既定値を当てにさせない）。
DEFAULT_OUT_SIZE = 512


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
    out_size: int = DEFAULT_OUT_SIZE,
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


def build_model_info(weights_path, *, infer_input_size: int = 640,
                     conf_threshold: Optional[float] = None) -> dict:
    """クロップを作った BBOX モデルの素性をまとめる。

    **どの重みで作ったクロップかが残らないと、モデルを更新したときに
    「どれが古いモデル由来か」を追えなくなる。**
    重みは差し替わるので、パスだけでなくハッシュも持つ。
    """
    w = Path(weights_path)
    info = {
        "model_id": w.parent.parent.name if w.parent.name == "weights" else w.stem,
        "weights_path": str(w),
        "weights_sha1": _sha1(w),
        "infer_input_size": int(infer_input_size),
        "inferred_at": datetime.now().astimezone().isoformat(),
    }
    if conf_threshold is not None:
        info["conf_threshold"] = float(conf_threshold)
    try:
        info["trained_at"] = datetime.fromtimestamp(
            w.stat().st_mtime).astimezone().isoformat()
    except OSError:
        pass
    return info


def draw_debug_overlay(crop, geom: dict, target_bbox_xyxy, others=None):
    """確認用に、クロップへ枠を重ねた画像を作る。

    切り出しが意図どおりか（対象が中心にいるか、余白が合っているか）は
    数字を見ても分からない。仕様 §8 のサニティ確認用。
    """
    import cv2

    out = crop.copy()
    x1, y1, x2, y2 = [float(v) for v in target_bbox_xyxy]
    ax1, ay1 = source_to_crop(x1, y1, geom)
    ax2, ay2 = source_to_crop(x2, y2, geom)
    th = max(1, int(round(min(out.shape[:2]) / 250)))

    # 主対象の bbox（緑）
    cv2.rectangle(out, (int(ax1), int(ay1)), (int(ax2), int(ay2)),
                  (80, 220, 80), th)
    # 学習で使う内側の範囲（青）
    tr = geom.get("target_rect_in_crop")
    if tr:
        cv2.rectangle(out, (int(tr[0]), int(tr[1])),
                      (int(tr[0] + tr[2]), int(tr[1] + tr[3])),
                      (244, 207, 78), th)
    # 同じクロップに写り込む他の対象（灰）
    for o in (others or []):
        ox1, oy1 = source_to_crop(o[0], o[1], geom)
        ox2, oy2 = source_to_crop(o[2], o[3], geom)
        cv2.rectangle(out, (int(ox1), int(oy1)), (int(ox2), int(oy2)),
                      (150, 150, 150), max(1, th - 1))
    return out


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
    out_size: int = DEFAULT_OUT_SIZE,
    max_upscale: float = 1.5,
    dedup_center_dist: float = 0.0,
    out_format: str = "png",
    jpg_quality: int = 95,
    save_rejected: bool = False,
    model_info: Optional[dict] = None,
    name_prefix: str = "obj",
    debug_overlay: bool = False,
    debug_samples: int = 20,
    seed: int = 0,
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
        "upscale_limited": 0, "debug_images": 0, "skipped": [], "errors": [],
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

    # 確認画像は数枚だけ。全部出すと枚数ぶん時間と容量がかかる。
    # 先頭から順に出すと 1 枚目の元画像に偏るので、全体に散らす。
    # そのために総数を先に見積もって確率を決める。
    _dbg_n = 0
    _dbg_pick = None
    if debug_overlay:
        import random as _random
        _rng = _random.Random(seed)
        _est_total = sum(len(d) for d in detections_by_image.values() if d)
        _p = 1.0 if _est_total <= debug_samples else debug_samples / _est_total
        _dbg_pick = lambda: _rng.random() < _p      # noqa: E731

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

                # 確認用の重ね描き（切り出しが意図どおりかは数字では分からない）
                if debug_overlay and _dbg_pick is not None and _dbg_n < debug_samples:
                    if _dbg_pick():
                        try:
                            dbg = draw_debug_overlay(
                                crop, meta["crop_geometry"], det["bbox_xyxy"],
                                others=[o["bbox_xyxy"] for j, o in enumerate(kept)
                                        if j != i and _overlaps(o["bbox_xyxy"], geom)])
                            (out / "debug").mkdir(parents=True, exist_ok=True)
                            cv2.imwrite(str(out / "debug" / Path(img_rel).name),
                                        dbg, params)
                            _dbg_n += 1
                        except Exception as e:
                            result["errors"].append(f"確認画像: {e}")

                mf.write(json.dumps({
                    "crop_image": img_rel, "meta": meta_rel,
                    "data_type": "object_crop", "annotation_status": "raw",
                    "source_image": str(img_path),
                    "confidence": float(det.get("confidence", 0.0)),
                    "source_width": src_w, "source_height": src_h,
                }, ensure_ascii=False) + "\n")
                result["crops"] += 1

            result["images"] += 1

    result["debug_images"] = _dbg_n
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
def target_tile_count(n_object_crops: int, background_ratio: float) -> int:
    """背景タイルを何枚採るか。

    `background_ratio` は「最終的なデータセットに占める背景の割合」なので、
    対象クロップ N 枚に対して T/(N+T) = r となる T を返す。
    枚数そのものではなく割合で指定するのは、
    対象クロップの枚数が撮り足すたびに変わるため。
    """
    r = min(0.95, max(0.0, float(background_ratio)))
    if r <= 0 or n_object_crops <= 0:
        return 0
    return int(round(n_object_crops * r / (1.0 - r)))


BG_SAMPLING = {
    "random": "🎲 ランダム",
    "even":   "⚖️ 元画像ごとに均等",
    "all":    "📥 全部採用",
}


def _source_of(stem: str) -> str:
    """タイル名から元画像の stem を取り出す（`xxx_tile_r0_c1` → `xxx`）"""
    i = stem.rfind("_tile_r")
    return stem[:i] if i > 0 else stem


def sample_tiles(names: list, want: int, how: str = "random",
                 seed: int = 0) -> list:
    """背景タイルから採用ぶんを選ぶ。

    `even` は**元画像ごとに順番に採る**。ランダムだと、たまたま同じ場所を
    写した 1 枚から大量に採ってしまい、背景の種類が偏ることがある。
    どちらも seed 固定で毎回同じ選び方になる（再現性）。
    """
    import random as _random

    names = list(names)
    if how == "all" or want >= len(names):
        return sorted(names)
    if want <= 0:
        return []

    rng = _random.Random(seed)
    if how != "even":
        return sorted(rng.sample(names, want))

    groups: dict = {}
    for n in names:
        groups.setdefault(_source_of(n), []).append(n)
    for g in groups.values():
        rng.shuffle(g)

    picked, keys = [], sorted(groups)
    while len(picked) < want:
        moved = False
        for k in keys:
            if groups[k]:
                picked.append(groups[k].pop())
                moved = True
                if len(picked) >= want:
                    break
        if not moved:
            break
    return sorted(picked)


def read_tile_selection(bg_dir) -> dict:
    """いまの採否を {タイル名: 採用か} で返す"""
    rows = {}
    mf = Path(bg_dir) / "manifest.jsonl"
    if not mf.exists():
        return rows
    for line in mf.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        stem = Path(r.get("crop_image", "")).stem
        if stem:
            rows[stem] = bool(r.get("adopted"))
    return rows


def apply_selection(bg_dir, adopt: list) -> dict:
    """人が決めた採否を反映する（`images/` と `_unused/` を入れ替える）。

    抽選はあくまで下ごしらえで、**最終判断は人**。
    捨てずに `_unused/` へ動かすだけなので、何度でもやり直せる。
    """
    out = Path(bg_dir)
    cur = read_tile_selection(out)
    if not cur:
        return {"ok": False, "error": "manifest.jsonl がありません",
                "adopted": 0, "unused": 0, "moved": 0}

    want = set(adopt)
    unused = out / "_unused"
    moved = 0

    for stem, was in cur.items():
        now = stem in want
        if now == was:
            continue
        # ここに来るのは採否が変わったものだけ
        src_base, dst_base = (out, unused) if was else (unused, out)
        for sub, ext_glob in (("images", "*"), ("meta", ".json")):
            if sub == "images":
                cands = list((src_base / "images").glob(f"{stem}.*"))
            else:
                cands = [src_base / "meta" / f"{stem}.json"]
            for f in cands:
                if not f.exists():
                    continue
                (dst_base / sub).mkdir(parents=True, exist_ok=True)
                f.replace(dst_base / sub / f.name)
                if sub == "images":
                    moved += 1

    # manifest を書き直す（採否の正本はここ）
    ext = {}
    for stem in cur:
        for base in (out, unused):
            hit = list((base / "images").glob(f"{stem}.*"))
            if hit:
                ext[stem] = hit[0].suffix.lstrip(".")
                break
    with open(out / "manifest.jsonl", "w", encoding="utf-8") as mf:
        for stem in sorted(cur):
            ok = stem in want
            base = "" if ok else "_unused/"
            e = ext.get(stem, "png")
            mf.write(json.dumps({
                "crop_image": f"{base}images/{stem}.{e}",
                "meta": f"{base}meta/{stem}.json",
                "data_type": "background_tile",
                "annotation_status": "raw",
                "adopted": ok,
            }, ensure_ascii=False) + "\n")

    return {"ok": True, "error": "", "moved": moved,
            "adopted": sum(1 for s in cur if s in want),
            "unused": sum(1 for s in cur if s not in want)}


# ---------------------------------------------------------------------------
# コンタクトシート（仕様 §7.3）
#
#   タイルを縮小して 1 枚に並べる。画面を延々スクロールするより、
#   一覧で見て「これとこれを外す」と番号を控えるほうが速い場面がある
#   （紙に出す・別の端末で見る・複数人で確認する、など）。
#
#   **番号の対応表を必ず残す。** 番号だけ控えてもらっても、
#   作り直したときに順番が変われば意味がなくなる。
# ---------------------------------------------------------------------------
CONTACT_DIR = "contact_sheet"


def build_contact_sheet(bg_dir, names: Optional[list] = None, *,
                        cols: int = 6, rows: int = 8, thumb: int = 160,
                        out_format: str = "png") -> dict:
    """背景タイルを縮小して並べたシートを作る。

    採用中のものは緑、未採用は灰の枠で囲む（いまの状態が見えるように）。
    左上に通し番号を焼き込み、対応表を `index.json` に残す。
    """
    import cv2
    import numpy as np

    out = Path(bg_dir)
    result = {"ok": False, "error": "", "sheets": [], "index": {}, "total": 0}

    state = read_tile_selection(out)
    if not state:
        result["error"] = "背景タイルがありません"
        return result

    targets = sorted(names) if names is not None else sorted(state)
    targets = [t for t in targets if t in state]
    if not targets:
        result["error"] = "並べる対象がありません"
        return result

    sheet_dir = out / CONTACT_DIR
    if sheet_dir.exists():
        for f in sheet_dir.glob("*"):
            f.unlink(missing_ok=True)
    sheet_dir.mkdir(parents=True, exist_ok=True)

    label_h = max(16, thumb // 8)
    cell_w, cell_h = thumb, thumb + label_h
    per = max(1, cols * rows)
    ext = "jpg" if out_format == "jpg" else "png"
    index: dict = {}

    for page in range((len(targets) - 1) // per + 1):
        chunk = targets[page * per: (page + 1) * per]
        n_rows = (len(chunk) - 1) // cols + 1
        sheet = np.full((n_rows * cell_h, cols * cell_w, 3), 32, np.uint8)

        for i, stem in enumerate(chunk):
            num = page * per + i + 1
            index[str(num)] = stem
            adopted = state.get(stem, False)

            base = out if adopted else out / "_unused"
            hit = list((base / "images").glob(f"{stem}.*"))
            r, c = divmod(i, cols)
            x0, y0 = c * cell_w, r * cell_h

            if hit:
                im = cv2.imread(str(hit[0]))
                if im is not None:
                    im = cv2.resize(im, (thumb, thumb),
                                    interpolation=cv2.INTER_AREA)
                    sheet[y0:y0 + thumb, x0:x0 + thumb] = im

            # 採用中かどうかが一目で分かるように枠を付ける
            color = (80, 220, 80) if adopted else (110, 110, 110)
            cv2.rectangle(sheet, (x0 + 1, y0 + 1), (x0 + thumb - 2, y0 + thumb - 2),
                          color, 2 if adopted else 1)
            cv2.putText(sheet, str(num), (x0 + 6, y0 + thumb + label_h - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, thumb / 400, color,
                        max(1, thumb // 120), cv2.LINE_AA)

        f = sheet_dir / f"sheet_{page + 1:02d}.{ext}"
        cv2.imwrite(str(f), sheet)
        result["sheets"].append(str(f))

    (sheet_dir / "index.json").write_text(
        json.dumps({"created_at": datetime.now().astimezone().isoformat(),
                    "cols": cols, "rows": rows, "thumb": thumb,
                    "index": index}, ensure_ascii=False, indent=2),
        encoding="utf-8")

    result["index"] = index
    result["total"] = len(targets)
    result["ok"] = True
    return result


def read_contact_index(bg_dir) -> dict:
    """コンタクトシートの番号 → タイル名"""
    f = Path(bg_dir) / CONTACT_DIR / "index.json"
    if not f.exists():
        return {}
    try:
        return dict((json.loads(f.read_text(encoding="utf-8")).get("index") or {}))
    except Exception:
        return {}


def parse_selection_list(text: str, index: Optional[dict] = None) -> tuple:
    """控えてきた採否リストを解釈する。

    番号（`3`）・範囲（`5-12`）・ファイル名のどれでも受ける。
    区切りは空白・改行・カンマ・読点。人が手で書くものなので緩く読む。
    戻り値: (タイル名のリスト, 解釈できなかった語のリスト)
    """
    import re

    index = index or {}
    names, bad = [], []
    for tok in re.split(r"[\s,、]+", (text or "").strip()):
        if not tok:
            continue
        m = re.fullmatch(r"(\d+)\s*[-–~〜]\s*(\d+)", tok)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            for n in range(min(a, b), max(a, b) + 1):
                if str(n) in index:
                    names.append(index[str(n)])
                else:
                    bad.append(str(n))
            continue
        if tok.isdigit():
            (names.append(index[tok]) if tok in index else bad.append(tok))
            continue
        stem = Path(tok).stem
        if not index or stem in index.values():
            names.append(stem)
        else:
            bad.append(tok)
    # 重複は落とす（同じ番号を 2 回書いても困らないように）
    seen, uniq = set(), []
    for n in names:
        if n not in seen:
            seen.add(n)
            uniq.append(n)
    return uniq, bad


def generate_background_tiles(
    image_paths,
    out_dir: Path,
    tile_size: int = 1024,
    tile_overlap: float = 0.0,
    out_format: str = "png",
    jpg_quality: int = 95,
    background_ratio: float = 0.15,
    bg_sampling: str = "random",
    keep_unused_tiles: bool = True,
    n_object_crops: int = 0,
    seed: int = 0,
    on_progress=None,
) -> dict:
    """対象の検出されない画像をタイルに分割する。

    中身が薄いタイルを機械的に捨てることはしない。
    「何も無い」の判定は対象領域によって大きく変わり、誤ると必要な背景を失う。
    全タイルを出し、採否は人が決める。

    そのうえで、**枚数は `background_ratio` で調整する**。
    背景ばかりが増えると、対象の学習に使える割合が下がるため。
    採らなかったタイルは `_unused/` に置く（消さない。後から足せるように）。
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
    params = ([cv2.IMWRITE_JPEG_QUALITY, int(jpg_quality)] if ext == "jpg" else [])
    step = max(1, int(tile_size * (1.0 - min(0.5, max(0.0, tile_overlap)))))
    made: list[str] = []          # 作った順のタイル名（採否の振り分けに使う）

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
                cv2.imwrite(str(out / "images" / f"{stem}.{ext}"), tile, params)
                made.append(stem)
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

    # ── 採否を決める（枚数は割合から求める）──────────────────────────
    # 割合を計算する材料が無いとき（割合の指定なし・対象クロップ数が不明）は
    # **全部採用する**。黙って `_unused` に送ると、作ったのに 0 枚に見える。
    # 一方、材料があって計算結果が 0 枚になるのは正しい答えなので、そのまま従う
    # （対象が 1 件しかなければ、割合 0.15 に見合う背景は 0 枚）。
    _no_basis = (bg_sampling == "all" or background_ratio <= 0
                 or n_object_crops <= 0)
    want = (len(made) if _no_basis
            else min(len(made), target_tile_count(n_object_crops, background_ratio)))

    adopted = sample_tiles(made, want, how=bg_sampling, seed=seed)
    adopted_set = set(adopted)

    unused_dir = out / "_unused"
    for stem in made:
        if stem in adopted_set:
            continue
        src_img = out / "images" / f"{stem}.{ext}"
        if keep_unused_tiles:
            (unused_dir / "images").mkdir(parents=True, exist_ok=True)
            (unused_dir / "meta").mkdir(parents=True, exist_ok=True)
            src_img.replace(unused_dir / "images" / src_img.name)
            (out / "meta" / f"{stem}.json").replace(
                unused_dir / "meta" / f"{stem}.json")
        else:
            src_img.unlink(missing_ok=True)
            (out / "meta" / f"{stem}.json").unlink(missing_ok=True)

    with open(out / "manifest.jsonl", "w", encoding="utf-8") as mf:
        for stem in made:
            ok = stem in adopted_set
            base = "" if ok else "_unused/"
            mf.write(json.dumps({
                "crop_image": f"{base}images/{stem}.{ext}",
                "meta": f"{base}meta/{stem}.json",
                "data_type": "background_tile",
                "annotation_status": "raw",
                "adopted": ok,
            }, ensure_ascii=False) + "\n")

    result["tiles"] = len(adopted)
    result["generated"] = len(made)
    result["unused"] = len(made) - len(adopted)
    result["target"] = want
    result["ok"] = len(made) > 0
    if not made:
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


# ---------------------------------------------------------------------------
# 学習用への切り直し（アノテーション倍率 → target 倍率）
#
#   アノテーション用は大きめ（annotation_scale=2.0）に切る。
#   周りが見えていたほうが、どこまでが対象かを判断しやすいため。
#   しかし**学習は実機と同じ画角（target_scale=1.5）でなければならない**。
#   切り直さないまま学習すると、学習では対象が画面の 1/2、
#   実機では 2/3 を占めることになり、写る大きさがずれる。
#
#   切り出しは正方形で中心ぞろえなので、倍率の比だけで内側の矩形が決まる
#   （メタが無くても切り直せる）。ラベルも同じ変換で移す。
# ---------------------------------------------------------------------------
def _recut_label_line(line: str, rx: float, ry: float, rw: float, rh: float,
                      W: int, H: int) -> Optional[str]:
    """ラベル 1 行を、切り直した矩形の座標系へ移す。

    枠から出た部分は切り詰める。完全に外に出たものは None（＝落とす）。
    """
    parts = line.split()
    if len(parts) < 5:
        return None
    cls = parts[0]
    try:
        vals = [float(v) for v in parts[1:]]
    except ValueError:
        return None

    if len(vals) == 4:      # detect: cx cy w h
        cx, cy, bw, bh = vals
        x1, y1 = (cx - bw / 2) * W, (cy - bh / 2) * H
        x2, y2 = (cx + bw / 2) * W, (cy + bh / 2) * H
        nx1, ny1 = max(x1, rx), max(y1, ry)
        nx2, ny2 = min(x2, rx + rw), min(y2, ry + rh)
        if nx2 - nx1 <= 1 or ny2 - ny1 <= 1:
            return None
        ncx = ((nx1 + nx2) / 2 - rx) / rw
        ncy = ((ny1 + ny2) / 2 - ry) / rh
        return (f"{cls} {ncx:.6f} {ncy:.6f} "
                f"{(nx2 - nx1) / rw:.6f} {(ny2 - ny1) / rh:.6f}")

    if len(vals) < 6 or len(vals) % 2:
        return None

    pts = [(vals[i] * W, vals[i + 1] * H) for i in range(0, len(vals), 2)]
    inside = [p for p in pts
              if rx <= p[0] <= rx + rw and ry <= p[1] <= ry + rh]
    if not inside:
        return None         # まるごと枠の外

    moved = []
    for x, y in pts:
        nx = min(max(x, rx), rx + rw)
        ny = min(max(y, ry), ry + rh)
        moved.append(((nx - rx) / rw, (ny - ry) / rh))
    # 潰れた（面積が無い）ものは落とす
    xs = [p[0] for p in moved]
    ys = [p[1] for p in moved]
    if max(xs) - min(xs) < 1e-3 or max(ys) - min(ys) < 1e-3:
        return None
    return cls + " " + " ".join(f"{x:.6f} {y:.6f}" for x, y in moved)


def recut_dataset(src_dir, out_dir, from_scale: float, to_scale: float,
                  out_size: Optional[int] = None, on_progress=None) -> dict:
    """アノテーション倍率で切ったデータセットを、target 倍率へ切り直す。

    画像は中心を保ったまま内側を切り、ラベルも同じ変換で移す。
    `out_size` を渡すと最後に一度だけリサイズする（実機の出力と揃えたいとき）。
    """
    import cv2

    src, out = Path(src_dir), Path(out_dir)
    result = {"ok": False, "error": "", "images": 0, "labels": 0,
              "dropped": 0, "skipped": [], "out_dir": str(out)}

    if not src.exists():
        result["error"] = f"元のデータセットがありません: {src}"
        return result
    if out.exists() and any(out.iterdir()):
        result["error"] = f"出力先が既にあります: {out.name}（別の名前にしてください）"
        return result
    ratio = float(to_scale) / max(1e-6, float(from_scale))
    if not (0 < ratio < 1):
        result["error"] = (f"target 倍率 ({to_scale}) は "
                           f"アノテーション倍率 ({from_scale}) より小さくしてください")
        return result

    imgs = [p for p in src.rglob("images/*/*") if p.suffix.lower() in IMG_EXTS]
    if not imgs:
        result["error"] = "画像が見つかりません"
        return result

    for n, ip in enumerate(imgs, 1):
        if on_progress:
            on_progress(n, len(imgs))
        img = cv2.imread(str(ip))
        if img is None:
            result["skipped"].append((str(ip), "画像を読めません"))
            continue
        H, W = img.shape[:2]
        rw, rh = W * ratio, H * ratio
        rx, ry = (W - rw) / 2, (H - rh) / 2
        ix, iy = int(round(rx)), int(round(ry))
        ix2, iy2 = int(round(rx + rw)), int(round(ry + rh))
        cut = img[iy:iy2, ix:ix2]
        if cut.size == 0:
            result["skipped"].append((str(ip), "切り直せません"))
            continue
        if out_size:
            cut = cv2.resize(cut, (int(out_size), int(out_size)),
                             interpolation=(cv2.INTER_AREA
                                            if cut.shape[0] > out_size
                                            else cv2.INTER_LINEAR))

        rel = ip.relative_to(src)
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst), cut)
        result["images"] += 1

        lp = Path(str(ip).replace("/images/", "/labels/")).with_suffix(".txt")
        if not lp.exists():
            continue
        lines = []
        for line in lp.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            moved = _recut_label_line(line, rx, ry, rw, rh, W, H)
            if moved:
                lines.append(moved)
            else:
                result["dropped"] += 1
        lo = out / lp.relative_to(src)
        lo.parent.mkdir(parents=True, exist_ok=True)
        lo.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        result["labels"] += len(lines)

    for extra in ("data.yaml", "classes.txt"):
        f = src / extra
        if f.exists():
            (out / extra).write_text(f.read_text(encoding="utf-8"),
                                     encoding="utf-8")

    result["ok"] = result["images"] > 0
    if not result["images"]:
        result["error"] = "1 枚も切り直せませんでした"
    return result


# ---------------------------------------------------------------------------
# 処理ログ（仕様 §8）と、実機へ渡す設定
# ---------------------------------------------------------------------------
def write_crop_log(out_dir, result: dict, params: dict,
                   bg_result: Optional[dict] = None) -> Optional[Path]:
    """何をどう作ったかを出力先に残す（仕様 §8）。

    画面の表示は次の操作で消える。あとから
    「このデータはどの設定で作ったか」を確かめられるようにする。
    """
    out = Path(out_dir)
    try:
        out.mkdir(parents=True, exist_ok=True)
        lines = [
            f"# クロップ生成ログ  {datetime.now().astimezone().isoformat()}",
            "",
            "## 設定",
        ]
        lines += [f"  {k:20} {v}" for k, v in params.items()]
        lines += [
            "",
            "## 結果",
            f"  入力画像           {result.get('images', 0)}",
            f"  生成クロップ       {result.get('crops', 0)}",
            f"  重複除去で落とした {result.get('rejected', 0)}",
            f"  拡大上限が効いた   {result.get('upscale_limited', 0)}",
            f"  確認画像           {result.get('debug_images', 0)}",
        ]
        if bg_result:
            lines += [
                "",
                "## 背景タイル",
                f"  生成               {bg_result.get('generated', 0)}",
                f"  採用               {bg_result.get('tiles', 0)}",
                f"  未採用             {bg_result.get('unused', 0)}",
            ]
        skipped = list(result.get("skipped") or [])
        errors = list(result.get("errors") or [])
        if skipped or errors:
            lines += ["", "## 問題（処理は止めていない）"]
            for p, why in skipped[:200]:
                lines.append(f"  skip  {p} — {why}")
            for e in errors[:200]:
                lines.append(f"  error {e}")

        f = out / "crop_log.txt"
        f.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return f
    except Exception:
        return None


CROP_CONFIG = "crop_config.json"


def write_crop_config(out_dir, params: dict) -> Optional[Path]:
    """実機がそのまま読める形で、切り出しの規則を書き出す。

    **学習データを作った設定と実機の設定がずれると精度が壊れる。**
    人が値を書き写す運用にすると、いつか必ずずれるので、
    ファイルにして渡せるようにしておく。
    実機は `target_scale` で 1 回切る（アノテーション用の 2 段倍率は使わない）。
    """
    try:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        cfg = {
            "schema_version": SCHEMA_VERSION,
            "created_at": datetime.now().astimezone().isoformat(),
            "note": "実機は make_crop(frame, bbox, **runtime) で 1 回切る",
            "runtime": {
                "scale": float(params.get("target_scale", 1.5)),
                "scale_basis": params.get("scale_basis", "long_side"),
                "square": bool(params.get("square", True)),
                "pad_mode": params.get("pad_mode", "reflect"),
                "pad_value": int(params.get("pad_value", 0)),
                "out_size": int(params.get("out_size", DEFAULT_OUT_SIZE)),
                "max_upscale": float(params.get("max_upscale", 1.5)),
            },
            "annotation": {
                "scale": float(params.get("annotation_scale", 2.0)),
            },
        }
        f = out / CROP_CONFIG
        f.write_text(json.dumps(cfg, ensure_ascii=False, indent=2),
                     encoding="utf-8")
        return f
    except Exception:
        return None


def load_crop_config(path) -> dict:
    """`crop_config.json` から実機用の引数を読む。

    使い方:
        cfg = load_crop_config("data/crops_xxx")
        crop, geom = make_crop(frame, bbox_xyxy, **cfg)
    """
    p = Path(path)
    if p.is_dir():
        p = p / CROP_CONFIG
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    rt = data.get("runtime") or {}
    allowed = {"scale", "scale_basis", "square", "pad_mode",
               "pad_value", "out_size", "max_upscale"}
    return {k: v for k, v in rt.items() if k in allowed}
