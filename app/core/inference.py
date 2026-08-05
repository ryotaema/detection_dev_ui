# =============================================================================
# 推論と結果の描画・書き出し
# =============================================================================
from __future__ import annotations

import json
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import streamlit as st

from .config import PREDICTIONS_DIR
from .utils import _box_iou


# ---------------------------------------------------------------------------
# 推論結果プレビュー描画
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _draw_predictions(json_path: Path):
    """JSONを読み込みバウンディングボックスを描画した RGB 画像配列を返す。失敗時は None。"""
    import cv2

    _COLORS = [
        (78, 207, 244), (244, 168, 78), (126, 207, 78),
        (207, 78, 126), (168, 78, 244), (78, 168, 207),
    ]
    try:
        with open(json_path) as f:
            pred = json.load(f)
        img_path = pred.get("image_path", "")
        if not img_path or not Path(img_path).exists():
            return None
        img = cv2.imread(img_path)
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes = pred.get("boxes", [])
        label_set = list(dict.fromkeys(b["label"] for b in boxes))
        for box in boxes:
            xyxy = box.get("bbox_xyxy", [])
            if len(xyxy) != 4:
                continue
            x1, y1, x2, y2 = [int(v) for v in xyxy]
            label = box["label"]
            conf  = box.get("confidence", 0.0)
            color = _COLORS[label_set.index(label) % len(_COLORS)]
            # セグメンテーション結果があれば輪郭を重ねて塗る
            _mxy = box.get("mask_xy")
            if _mxy and len(_mxy) >= 3:
                import numpy as _np_seg
                _pts = _np_seg.array(_mxy, dtype=_np_seg.int32).reshape(-1, 1, 2)
                _overlay = img.copy()
                cv2.fillPoly(_overlay, [_pts], color)
                cv2.addWeighted(_overlay, 0.35, img, 0.65, 0, img)
                cv2.polylines(img, [_pts], True, color, 2)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            text = f"{label} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(img, text, (x1 + 2, y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return img, len(boxes), json_path.stem
    except Exception:
        return None


def export_prediction_images(
    out_dir: Path,
    img_format: str = "PNG",
    quality: int = 95,
    target_files: Optional[list[Path]] = None,
    progress_cb=None,
) -> tuple[int, int]:
    """predictions/*.json を描画済み画像として out_dir に書き出す。
    target_files が None のときは predictions/ の全 JSON を対象とする。
    progress_cb(current, total, filename) を渡すと処理ごとに呼ばれる。
    Returns (成功数, スキップ数)
    """
    from PIL import Image as PILImage

    out_dir.mkdir(parents=True, exist_ok=True)
    ext = ".jpg" if img_format == "JPEG" else ".png"
    json_files = target_files if target_files else sorted(PREDICTIONS_DIR.glob("*.json"))
    total = len(json_files)
    success = 0
    skipped = 0
    for i, jf in enumerate(json_files):
        if progress_cb:
            progress_cb(i, total, jf.name)
        result = _draw_predictions(jf)
        if result is None:
            skipped += 1
            continue
        img_arr, _, stem = result
        out_path = out_dir / f"{stem}{ext}"
        pil_img = PILImage.fromarray(img_arr)
        if img_format == "JPEG":
            pil_img.save(out_path, "JPEG", quality=quality)
        else:
            pil_img.save(out_path, "PNG")
        success += 1
    return success, skipped


# ---------------------------------------------------------------------------
# 再アノテーション用 ZIP 生成
# ---------------------------------------------------------------------------
def build_reannotation_zip(json_paths: list[Path]) -> tuple[bytes, int, int]:
    """フラグ済み predictions JSON から CVAT for images 1.1 XML + YOLO txt + 元画像を
    まとめた ZIP バイト列を返す。
    Returns: (zip_bytes, 成功数, スキップ数)
    """
    import io as _io
    import xml.etree.ElementTree as ET
    import cv2

    success, skipped = 0, 0

    # ── 全JSONからクラス名を収集してインデックスを確定 ──────────────────────
    all_labels: list[str] = []
    pred_cache: dict[str, dict] = {}
    for jf in json_paths:
        try:
            with open(jf) as f:
                pred = json.load(f)
            pred_cache[str(jf)] = pred
            for b in pred.get("boxes", []):
                lbl = b.get("label", "")
                if lbl and lbl not in all_labels:
                    all_labels.append(lbl)
        except Exception:
            pass
    label2id = {lbl: i for i, lbl in enumerate(all_labels)}

    # ── CVAT for images 1.1 XML ルート構築 ──────────────────────────────────
    root_el = ET.Element("annotations")
    ET.SubElement(root_el, "version").text = "1.1"
    meta_el = ET.SubElement(root_el, "meta")
    task_el = ET.SubElement(meta_el, "task")
    labels_el = ET.SubElement(task_el, "labels")
    for lbl in all_labels:
        lbl_el = ET.SubElement(labels_el, "label")
        ET.SubElement(lbl_el, "name").text = lbl
        ET.SubElement(lbl_el, "attributes")

    zip_buf = _io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:

        for img_id, jf in enumerate(json_paths):
            pred = pred_cache.get(str(jf))
            if pred is None:
                skipped += 1
                continue

            orig_path = Path(pred.get("image_path", ""))
            if not orig_path.exists():
                skipped += 1
                continue

            # 画像サイズ取得
            img_cv = cv2.imread(str(orig_path))
            if img_cv is None:
                skipped += 1
                continue
            img_h, img_w = img_cv.shape[:2]

            fname = orig_path.name
            boxes = pred.get("boxes", [])

            # ── 元画像をそのまま images/ に追加 ─────────────────────────────
            zf.write(orig_path, f"images/{fname}")

            # ── YOLO txt ラベルファイル ──────────────────────────────────────
            txt_lines: list[str] = []
            for b in boxes:
                lbl = b.get("label", "")
                if lbl not in label2id:
                    continue
                cls_id = label2id[lbl]
                xywhn = b.get("bbox_xywhn")
                if xywhn and len(xywhn) == 4:
                    cx, cy, bw, bh = xywhn
                else:
                    # bbox_xywhn がない場合（動画推論など）は xyxy から計算
                    xyxy = b.get("bbox_xyxy", [])
                    if len(xyxy) != 4:
                        continue
                    x1, y1, x2, y2 = xyxy
                    cx = (x1 + x2) / 2 / img_w
                    cy = (y1 + y2) / 2 / img_h
                    bw = (x2 - x1) / img_w
                    bh = (y2 - y1) / img_h
                txt_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            stem = Path(fname).stem
            zf.writestr(f"labels/{stem}.txt", "\n".join(txt_lines))

            # ── CVAT XML の <image> 要素 ──────────────────────────────────────
            img_el = ET.SubElement(root_el, "image",
                                   id=str(img_id), name=fname,
                                   width=str(img_w), height=str(img_h))
            for b in boxes:
                lbl = b.get("label", "")
                xyxy = b.get("bbox_xyxy", [])
                if len(xyxy) != 4 or lbl not in label2id:
                    continue
                x1, y1, x2, y2 = xyxy
                box_el = ET.SubElement(img_el, "box",
                                       label=lbl,
                                       xtl=f"{x1:.2f}", ytl=f"{y1:.2f}",
                                       xbr=f"{x2:.2f}", ybr=f"{y2:.2f}",
                                       occluded="0")
                conf = b.get("confidence")
                if conf is not None:
                    attr_el = ET.SubElement(box_el, "attribute", name="confidence")
                    attr_el.text = f"{conf:.4f}"

            success += 1

        # ── classes.txt ─────────────────────────────────────────────────────
        zf.writestr("classes.txt", "\n".join(all_labels))

        # ── annotations.xml ─────────────────────────────────────────────────
        ET.indent(root_el)
        xml_str = ET.tostring(root_el, encoding="unicode", xml_declaration=False)
        xml_str = '<?xml version="1.0" encoding="utf-8"?>\n' + xml_str
        zf.writestr("annotations.xml", xml_str)

    zip_buf.seek(0)
    return zip_buf.getvalue(), success, skipped


# ---------------------------------------------------------------------------
# YOLO 推論
# ---------------------------------------------------------------------------
def prediction_json_path(out_dir: Path, image_path: str) -> Path:
    """推論結果 JSON の保存先を返す。

    画像名だけで付けると、別のデータセットにある同名画像（`1000color.png` のような
    連番名は特に）を推論したときに前の結果を黙って上書きしてしまう。
    元画像の絶対パスから短いハッシュを付けて衝突を防ぐ。
    同じ画像を再推論した場合は同じ名前になるので、意図した上書きは従来どおり効く。
    """
    import hashlib

    p = Path(image_path)
    digest = hashlib.sha1(str(p.resolve()).encode("utf-8")).hexdigest()[:8]
    return Path(out_dir) / f"{p.stem}__{digest}.json"


def prediction_display_name(json_path: Path) -> str:
    """一覧表示用の名前。JSON に元画像パスがあればその画像名を使う
    （ファイル名に付けたハッシュを利用者に見せないため）。
    """
    try:
        pred = json.loads(Path(json_path).read_text())
        img = pred.get("image_path")
        if img:
            return Path(img).name
    except Exception:
        pass
    return Path(json_path).name


def run_inference(
    model_path: str,
    image_dir: Path,
    out_dir: Path,
    conf_threshold: float = 0.25,
) -> list[Path]:
    """指定モデルで画像フォルダを推論し、結果JSONを out_dir に保存して返す"""
    try:
        from ultralytics import YOLO

        model = YOLO(model_path)
        _task = getattr(model, "task", "detect") or "detect"
        results_list = model.predict(
            source=str(image_dir),
            conf=conf_threshold,
            save=False,
        )
        saved_jsons = []
        for res in results_list:
            img_path = res.path
            _h, _w = res.orig_shape
            # セグメンテーションモデルならインスタンスごとの輪郭が入る
            _masks = getattr(res, "masks", None)
            _mask_xy  = list(getattr(_masks, "xy", []) or [])  if _masks is not None else []
            _mask_xyn = list(getattr(_masks, "xyn", []) or []) if _masks is not None else []

            boxes = []
            if res.boxes:
                for _bi, box in enumerate(res.boxes):
                    xyxy   = box.xyxy[0].tolist()
                    xywhn  = box.xywhn[0].tolist()
                    cls_id = int(box.cls[0])
                    conf   = float(box.conf[0])
                    label  = res.names[cls_id]
                    item = {
                        "label": label,
                        "confidence": round(conf, 4),
                        "bbox_xyxy": [round(v, 2) for v in xyxy],
                        "bbox_xywhn": [round(v, 6) for v in xywhn],
                    }
                    # マスクは輪郭ポリゴンとして保存する（絶対座標と正規化の両方）
                    if _bi < len(_mask_xy):
                        item["mask_xy"] = [[round(float(x), 2), round(float(y), 2)]
                                           for x, y in _mask_xy[_bi]]
                    if _bi < len(_mask_xyn):
                        item["mask_xyn"] = [[round(float(x), 6), round(float(y), 6)]
                                            for x, y in _mask_xyn[_bi]]
                    boxes.append(item)

            out_json = prediction_json_path(out_dir, img_path)
            with open(out_json, "w") as f:
                json.dump({
                    "image_path": img_path,
                    "task": _task,                 # detect / segment / pose
                    "image_size": [int(_w), int(_h)],
                    "boxes": boxes,
                }, f, indent=2, ensure_ascii=False)
            saved_jsons.append(out_json)

        return saved_jsons
    except Exception as e:
        st.error(f"推論エラー: {e}")
        return []


def run_video_inference(
    model_path: str,
    video_path: Path,
    out_dir: Path,
    conf_threshold: float = 0.25,
    enable_tracking: bool = False,
    tracker: str = "bytetrack.yaml",
    temporal_smoothing: bool = False,
    smooth_frames: int = 5,
    progress_cb=None,
) -> Optional[dict]:
    """動画ファイルをフレームごとに推論し、アノテーション済み動画とサマリーJSONを保存する。
    enable_tracking=True のとき model.track() でオブジェクトトラッキングを行う。
    temporal_smoothing=True のとき直近 smooth_frames フレームの検出を補完描画する。
    progress_cb(frame_idx, total_frames) を渡すと進捗通知に使われる。
    Returns: {"video_path": Path, "json_path": Path, "total_frames": int, "frame_stats": list} or None
    """
    import cv2
    import subprocess
    from ultralytics import YOLO

    # ゴーストボックス描画色（グレー）
    _GHOST_COLOR = (160, 160, 160)

    def _draw_ghost(frame, xyxy, label, conf, tid):
        x1, y1, x2, y2 = [int(v) for v in xyxy]
        cv2.rectangle(frame, (x1, y1), (x2, y2), _GHOST_COLOR, 1)
        text = f"{label} {conf:.2f}"
        if tid is not None:
            text += f" #{tid}"
        cv2.putText(frame, text, (x1 + 2, max(y1 - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, _GHOST_COLOR, 1)

    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        model = YOLO(model_path)

        cap = cv2.VideoCapture(str(video_path))
        fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # OpenCV で一時ファイルに書き出し → ffmpeg で H.264 に再エンコード
        tmp_path       = out_dir / f"{video_path.stem}_tmp.mp4"
        out_video_path = out_dir / f"{video_path.stem}_annotated.mp4"
        out_json_path  = out_dir / f"{video_path.stem}_summary.json"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(tmp_path), fourcc, fps, (width, height))

        frame_stats: list[dict] = []

        # テンポラル平滑化用メモリ
        # track ON  → {track_id: {xyxy, label, conf, tid, last_frame}}
        # track OFF → [{xyxy, label, conf, tid, last_frame}, ...]
        _track_mem: dict[int, dict] = {}
        _spatial_mem: list[dict] = []

        common_kwargs = dict(
            source=str(video_path),
            conf=conf_threshold,
            save=False,
            stream=True,
            verbose=False,
        )
        if enable_tracking:
            results = model.track(persist=True, tracker=tracker, **common_kwargs)
        else:
            results = model.predict(**common_kwargs)

        for frame_idx, res in enumerate(results):
            ids = res.boxes.id if (res.boxes is not None) else None

            # ── 現フレームの検出結果を収集 ──────────────────────────
            current_boxes = []
            if res.boxes is not None:
                for i, box in enumerate(res.boxes):
                    tid = int(ids[i]) if ids is not None else None
                    current_boxes.append({
                        "xyxy":  box.xyxy[0].tolist(),
                        "label": res.names[int(box.cls[0])],
                        "conf":  float(box.conf[0]),
                        "tid":   tid,
                    })

            # ── テンポラル平滑化: メモリ更新 ─────────────────────────
            if temporal_smoothing:
                if enable_tracking:
                    for b in current_boxes:
                        if b["tid"] is not None:
                            _track_mem[b["tid"]] = {**b, "last_frame": frame_idx}
                else:
                    used = set()
                    for b in current_boxes:
                        best_i, best_iou = -1, 0.3
                        for mi, m in enumerate(_spatial_mem):
                            if mi in used or m["label"] != b["label"]:
                                continue
                            iou = _box_iou(b["xyxy"], m["xyxy"])
                            if iou > best_iou:
                                best_iou, best_i = iou, mi
                        if best_i >= 0:
                            _spatial_mem[best_i] = {**b, "last_frame": frame_idx}
                            used.add(best_i)
                        else:
                            _spatial_mem.append({**b, "last_frame": frame_idx})
                    # 古いエントリを削除
                    _spatial_mem[:] = [
                        m for m in _spatial_mem
                        if frame_idx - m["last_frame"] <= smooth_frames
                    ]

            # ── ゴーストボックスを決定 ───────────────────────────────
            ghost_boxes: list[dict] = []
            if temporal_smoothing and smooth_frames > 0:
                if enable_tracking:
                    cur_tids = {b["tid"] for b in current_boxes if b["tid"] is not None}
                    for tid, mem in _track_mem.items():
                        age = frame_idx - mem["last_frame"]
                        if 0 < age <= smooth_frames and tid not in cur_tids:
                            ghost_boxes.append(mem)
                else:
                    for m in _spatial_mem:
                        age = frame_idx - m["last_frame"]
                        if 0 < age <= smooth_frames:
                            ghost_boxes.append(m)

            # ── 描画 ─────────────────────────────────────────────────
            annotated = res.plot()  # 現フレームの検出 + トラックID を自動描画
            for g in ghost_boxes:
                _draw_ghost(annotated, g["xyxy"], g["label"], g["conf"], g["tid"])
            writer.write(annotated)

            # ── JSON 用データ収集 ────────────────────────────────────
            boxes = []
            for b in current_boxes:
                entry = {
                    "label":      b["label"],
                    "confidence": round(b["conf"], 4),
                    "bbox_xyxy":  [round(v, 2) for v in b["xyxy"]],
                }
                if b["tid"] is not None:
                    entry["track_id"] = b["tid"]
                boxes.append(entry)
            frame_stats.append({"frame": frame_idx, "detections": len(boxes), "boxes": boxes})

            if progress_cb:
                progress_cb(frame_idx, total_frames)

        writer.release()

        # ffmpeg で H.264 / AAC に再エンコードしてブラウザ互換 MP4 を生成
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(tmp_path),
                "-vcodec", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-pix_fmt", "yuv420p",   # Streamlit / ブラウザ互換
                "-movflags", "+faststart",
                "-an",                   # 入力動画に音声がない場合のエラー回避
                str(out_video_path),
            ],
            check=True,
            capture_output=True,
        )
        tmp_path.unlink(missing_ok=True)

        summary = {
            "video_path": str(video_path),
            "output_video": str(out_video_path),
            "fps": fps,
            "total_frames": total_frames,
            "frame_stats": frame_stats,
        }
        with open(out_json_path, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        return {
            "video_path": out_video_path,
            "json_path": out_json_path,
            "total_frames": total_frames,
            "frame_stats": frame_stats,
        }
    except Exception as e:
        st.error(f"動画推論エラー: {e}")
        return None


# ---------------------------------------------------------------------------
# 拡大確認（再アノテーションのフラグ付けを楽にする）
#
#   一覧のプレビューは 1 枚が画面の 1/3 程度にしかならず、
#   「この検出が合っているか」を判断できない。
#   ここでは 2 つの見方を用意する:
#     - 画像全体を大きく（線の太さを画像サイズに合わせて調整する）
#     - 検出ボックスの周辺だけを切り出す（当たり外れはこちらのほうが速い）
# ---------------------------------------------------------------------------
def prediction_box_summaries(json_path: Path) -> list[dict]:
    """拡大対象を選ぶための、検出ごとの要約を返す"""
    try:
        with open(json_path) as f:
            pred = json.load(f)
    except Exception:
        return []

    out = []
    for i, b in enumerate(pred.get("boxes") or []):
        xyxy = b.get("bbox_xyxy") or []
        if len(xyxy) != 4:
            continue
        w = max(0.0, float(xyxy[2]) - float(xyxy[0]))
        h = max(0.0, float(xyxy[3]) - float(xyxy[1]))
        out.append({
            "index": i,
            "label": b.get("label", "?"),
            "confidence": float(b.get("confidence", 0.0)),
            "bbox_xyxy": [float(v) for v in xyxy],
            "size": (int(w), int(h)),
        })
    return out


def _scaled_draw(img, boxes, only_index: Optional[int] = None):
    """画像の大きさに合わせた太さで枠を描く。

    従来の描画は線幅 2px・文字 0.5 の固定で、1600px の画像では細すぎて見えない。
    拡大して見る用途では画像サイズに比例させる。
    """
    import cv2

    _COLORS = [
        (78, 207, 244), (244, 168, 78), (126, 207, 78),
        (207, 78, 126), (168, 78, 244), (78, 168, 207),
    ]
    h, w = img.shape[:2]
    thick = max(2, int(round(min(w, h) / 400)))
    fscale = max(0.5, min(w, h) / 900)

    label_set = list(dict.fromkeys(b.get("label", "?") for b in boxes))
    for i, b in enumerate(boxes):
        xyxy = b.get("bbox_xyxy") or []
        if len(xyxy) != 4:
            continue
        # 注目している検出だけ強調し、他は細く描く
        focused = (only_index is None or i == only_index)
        t = thick if focused else max(1, thick // 2)
        color = _COLORS[label_set.index(b.get("label", "?")) % len(_COLORS)]

        x1, y1, x2, y2 = [int(v) for v in xyxy]
        _mxy = b.get("mask_xy")
        if _mxy and len(_mxy) >= 3 and focused:
            import numpy as _np
            pts = _np.array(_mxy, dtype=_np.int32).reshape(-1, 1, 2)
            ov = img.copy()
            cv2.fillPoly(ov, [pts], color)
            cv2.addWeighted(ov, 0.30, img, 0.70, 0, img)
            cv2.polylines(img, [pts], True, color, t)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, t)

        if focused:
            text = f"{b.get('label','?')} {b.get('confidence', 0.0):.2f}"
            (tw, th), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, fscale, max(1, t - 1))
            cv2.rectangle(img, (x1, max(0, y1 - th - 8)),
                          (x1 + tw + 6, y1), color, -1)
            cv2.putText(img, text, (x1 + 3, max(th, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, fscale,
                        (255, 255, 255), max(1, t - 1))
    return img


def prediction_detail(
    json_path: Path,
    box_index: Optional[int] = None,
    margin: float = 0.8,
    max_side: int = 1400,
):
    """拡大確認用の画像を作る。

    box_index=None … 画像全体（枠は画像サイズに合わせた太さで描く）
    box_index=n    … その検出の周辺を切り出す。margin はボックス長辺に対する余白の割合

    戻り値: {"image": RGB配列, "n_boxes": int, "stem": str, "crop_rect": tuple|None}
    失敗時は None。
    """
    import cv2

    try:
        with open(json_path) as f:
            pred = json.load(f)
        img_path = pred.get("image_path", "")
        if not img_path or not Path(img_path).exists():
            return None
        img = cv2.imread(img_path)
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception:
        return None

    boxes = pred.get("boxes") or []
    crop_rect = None

    if box_index is not None and 0 <= box_index < len(boxes):
        xyxy = boxes[box_index].get("bbox_xyxy") or []
        if len(xyxy) == 4:
            h, w = img.shape[:2]
            x1, y1, x2, y2 = [float(v) for v in xyxy]
            base = max(x2 - x1, y2 - y1)
            pad = base * margin
            cx1 = max(0, int(x1 - pad))
            cy1 = max(0, int(y1 - pad))
            cx2 = min(w, int(x2 + pad))
            cy2 = min(h, int(y2 + pad))
            if cx2 - cx1 > 4 and cy2 - cy1 > 4:
                # 切り出す前に描画する（座標をずらさずに済む）
                img = _scaled_draw(img, boxes, only_index=box_index)
                img = img[cy1:cy2, cx1:cx2]
                crop_rect = (cx1, cy1, cx2, cy2)

    if crop_rect is None:
        img = _scaled_draw(img, boxes, only_index=box_index)

    # 大きすぎるとブラウザが重くなるので上限をかける（縮小のみ・拡大はしない）
    h, w = img.shape[:2]
    if max(h, w) > max_side:
        r = max_side / max(h, w)
        img = cv2.resize(img, (int(w * r), int(h * r)),
                         interpolation=cv2.INTER_AREA)

    return {"image": img, "n_boxes": len(boxes),
            "stem": Path(json_path).stem, "crop_rect": crop_rect}
