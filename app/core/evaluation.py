# =============================================================================
# モデル評価と正解ラベルとの差分分析
# =============================================================================
from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import IMG_EXTS, MODELS_DIR
from .dataset import _yolo_txt_to_xyxy, resolve_train_data_arg
from .state import _get_eval_shared
from .utils import _iou, slugify_function_name


def analyze_predictions(
    json_paths: list[Path],
    conf_low: float = 0.5,
    tiny_area: float = 0.001,
    conflict_iou: float = 0.5,
) -> list[dict]:
    """推論結果 JSON を分析し、要確認と判断した理由を付けて返す。

    判定理由:
      - 検出ゼロ            … 写っているのに拾えていない可能性（見逃し）
      - 低信頼度            … conf_low 未満の検出を含む（誤検出/曖昧）
      - クラス競合          … ほぼ同じ位置に別クラスが重なっている（モデルが迷っている）
      - 極小ボックス        … 画像面積比 tiny_area 未満（ノイズ検出の疑い）
    """
    results = []
    for jf in json_paths:
        try:
            pred = json.loads(Path(jf).read_text())
        except Exception:
            continue

        boxes = pred.get("boxes", []) or []
        confs = [float(b.get("confidence", 0.0)) for b in boxes]
        reasons: list[str] = []

        if not boxes:
            reasons.append("検出ゼロ")
        else:
            if min(confs) < conf_low:
                reasons.append(f"低信頼度({min(confs):.2f})")

            # クラス競合: IoU が高いのにラベルが異なる組み合わせ
            for i in range(len(boxes)):
                for k in range(i + 1, len(boxes)):
                    if boxes[i].get("label") == boxes[k].get("label"):
                        continue
                    bi, bk = boxes[i].get("bbox_xyxy"), boxes[k].get("bbox_xyxy")
                    if bi and bk and _iou(bi, bk) >= conflict_iou:
                        reasons.append("クラス競合")
                        break
                if "クラス競合" in reasons:
                    break

            # 極小ボックス（正規化 w*h で判定）
            for b in boxes:
                xywhn = b.get("bbox_xywhn")
                if xywhn and len(xywhn) == 4 and (xywhn[2] * xywhn[3]) < tiny_area:
                    reasons.append("極小ボックス")
                    break

        results.append({
            "json": Path(jf),
            "name": Path(jf).name,
            "image_path": pred.get("image_path", ""),
            "n_boxes": len(boxes),
            "min_conf": min(confs) if confs else None,
            "mean_conf": (sum(confs) / len(confs)) if confs else None,
            "reasons": reasons,
            "flagged": bool(reasons),
        })
    return results


def compare_with_ground_truth(
    model_path: Path,
    data_yaml: str,
    split: str = "val",
    conf: float = 0.25,
    iou_match: float = 0.5,
    max_images: int = 500,
) -> dict:
    """モデルの予測と GT を突き合わせ、画像ごとの TP/FP/FN を返す。

    マッチングは「同一クラスかつ IoU >= iou_match」を信頼度の高い予測から貪欲に行う。
    """
    res: dict = {
        "ok": False, "error": None,
        "model": str(model_path), "split": split, "conf": conf, "iou_match": iou_match,
        "n_images": 0, "tp": 0, "fp": 0, "fn": 0,
        "precision": None, "recall": None,
        "per_image": [], "by_class": {},
    }
    try:
        import yaml as _yml
        from ultralytics import YOLO

        cfg = _yml.safe_load(Path(data_yaml).read_text()) or {}
        names = cfg.get("names") or []
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names)]

        root = Path(cfg.get("path") or Path(data_yaml).parent)
        rel = cfg.get(split)
        if not rel:
            res["error"] = f"data.yaml に {split} の定義がありません"
            return res
        img_dir = (root / rel) if not str(rel).startswith("/") else Path(rel)
        if not img_dir.exists():
            res["error"] = f"画像ディレクトリが見つかりません: {img_dir}"
            return res

        images = sorted(p for p in img_dir.iterdir()
                        if p.is_file() and p.suffix.lower() in IMG_EXTS)
        if max_images and len(images) > max_images:
            images = images[:max_images]
        if not images:
            res["error"] = f"{img_dir} に画像がありません"
            return res

        model = YOLO(str(model_path))
        by_class: dict[str, dict] = {}

        # 画像リストを渡すと Results.path が仮名 (image0.jpg 等) になるため、
        # 入力順が保たれることを利用して元のパスと zip で対応付ける
        _results = model.predict(source=[str(p) for p in images], conf=conf,
                                 stream=True, verbose=False)
        for img_path, r in zip(images, _results):
            h, w = r.orig_shape

            _m = getattr(r, "masks", None)
            _mxy = list(getattr(_m, "xy", []) or []) if _m is not None else []

            preds = []
            if r.boxes is not None:
                for _bi, b in enumerate(r.boxes):
                    item = {
                        "label": r.names[int(b.cls[0])],
                        "confidence": float(b.conf[0]),
                        "bbox_xyxy": [float(v) for v in b.xyxy[0].tolist()],
                        "bbox_xywhn": [float(v) for v in b.xywhn[0].tolist()],
                    }
                    if _bi < len(_mxy):
                        item["mask_xy"] = [[float(x), float(y)] for x, y in _mxy[_bi]]
                    preds.append(item)
            preds.sort(key=lambda d: -d["confidence"])

            # images/... → labels/... の対応（YOLO の標準レイアウト）
            lbl_path = Path(str(img_path)
                            .replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}")
                            ).with_suffix(".txt")
            gts = _yolo_txt_to_xyxy(lbl_path, w, h, names)

            matched = [False] * len(gts)
            tp = fp = 0
            for p in preds:
                best_i, best_iou = -1, 0.0
                for gi, g in enumerate(gts):
                    if matched[gi] or g["label"] != p["label"]:
                        continue
                    v = _iou(p["bbox_xyxy"], g["bbox_xyxy"])
                    if v > best_iou:
                        best_i, best_iou = gi, v
                if best_i >= 0 and best_iou >= iou_match:
                    matched[best_i] = True
                    tp += 1
                    by_class.setdefault(p["label"], {"tp": 0, "fp": 0, "fn": 0})["tp"] += 1
                else:
                    fp += 1
                    by_class.setdefault(p["label"], {"tp": 0, "fp": 0, "fn": 0})["fp"] += 1

            fn = 0
            for gi, g in enumerate(gts):
                if not matched[gi]:
                    fn += 1
                    by_class.setdefault(g["label"], {"tp": 0, "fp": 0, "fn": 0})["fn"] += 1

            res["tp"] += tp
            res["fp"] += fp
            res["fn"] += fn
            res["per_image"].append({
                "image": str(img_path),
                "name": img_path.name,
                "width": w, "height": h,
                "n_gt": len(gts), "n_pred": len(preds),
                "tp": tp, "fp": fp, "fn": fn,
                "gt_boxes": gts, "pred_boxes": preds,
            })

        n_pred_total = res["tp"] + res["fp"]
        n_gt_total   = res["tp"] + res["fn"]
        res.update({
            "ok": True,
            "n_images": len(res["per_image"]),
            "precision": (res["tp"] / n_pred_total) if n_pred_total else None,
            "recall":    (res["tp"] / n_gt_total) if n_gt_total else None,
            "by_class":  by_class,
        })
    except Exception as e:
        res["error"] = f"{type(e).__name__}: {e}"
    return res


# ---------------------------------------------------------------------------
# モデル評価 (model.val)
#
#   学習時の results.csv は「そのモデルが自分の val で出した値」でしかないため、
#   別環境で学習したモデルとは比較できない。ここでは任意のデータセットに対して
#   同一条件で val を回し、モデル同士を同じ土俵で比べられるようにする。
# ---------------------------------------------------------------------------
def model_eval_path(model_path: Path) -> Path:
    """評価結果の保存先（rglob('*.pt') に載らないドット始まりの名前）"""
    return model_path.parent / f".{model_path.name}.eval.json"


def read_model_evals(model_path: Path) -> dict:
    """{データセットキー: 評価結果} を返す"""
    p = model_eval_path(model_path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def save_model_eval(model_path: Path, dataset_key: str, result: dict) -> None:
    evals = read_model_evals(model_path)
    evals[dataset_key] = result
    try:
        model_eval_path(model_path).write_text(
            json.dumps(evals, ensure_ascii=False, indent=2))
    except Exception:
        pass


def evaluate_model(
    model_path: Path,
    data_yaml: str,
    split: str = "val",
    imgsz: int = 640,
    batch: int = 8,
    conf: float = 0.001,
    iou: float = 0.6,
    plots: bool = True,
) -> dict:
    """1モデルを1データセットで評価する。

    conf の既定 0.001 は Ultralytics の val と同じ。mAP は全信頼度域の
    Precision-Recall 曲線から計算するため、低い値を使うのが正しい。
    """
    res: dict = {
        "ok": False, "error": None,
        "model": str(model_path), "data_yaml": data_yaml, "split": split,
        "imgsz": imgsz, "conf": conf, "iou": iou,
        "task": None,
        "map50": None, "map50_95": None, "precision": None, "recall": None,
        "mask_map50": None, "mask_map50_95": None,   # セグメンテーションモデルのみ
        "top1": None, "top5": None,                  # 画像分類モデルのみ
        "fitness": None, "per_class": [], "speed_ms": None, "n_images": None,
        "plots_dir": None,
        "evaluated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    try:
        from ultralytics import YOLO

        # 成果物（PR曲線・混同行列）はモデルの隣に置く
        plots_root = model_path.parent / ".eval_plots"
        run_name = slugify_function_name(f"{Path(data_yaml).parent.name}-{split}")
        out_dir = plots_root / run_name

        model = YOLO(str(model_path))
        r = model.val(
            data=resolve_train_data_arg(data_yaml), split=split, imgsz=imgsz, batch=batch,
            conf=conf, iou=iou, plots=plots, verbose=False,
            project=str(plots_root), name=run_name, exist_ok=True,
        )

        # ── 画像分類は mAP ではなく top1 / top5 accuracy ──────────────────
        if getattr(model, "task", None) == "classify":
            speed = getattr(r, "speed", {}) or {}
            res.update({
                "ok": True,
                "task": "classify",
                "top1": float(getattr(r, "top1", 0.0) or 0.0),
                "top5": float(getattr(r, "top5", 0.0) or 0.0),
                "fitness": float(getattr(r, "fitness", 0.0) or 0.0),
                "speed_ms": {k: round(float(v), 2) for k, v in speed.items()},
                "plots_dir": str(out_dir) if out_dir.exists() else None,
            })
            if plots:
                res["plots_dir"] = str(out_dir) if out_dir.exists() else None
            return res

        # OBB モデルでは指標が r.box ではなく r.obb に入る
        box = getattr(r, "obb", None) or r.box
        # セグメンテーションモデルでは r.seg にマスク基準の指標が入る
        seg = getattr(r, "seg", None)
        names = r.names if isinstance(r.names, dict) else dict(enumerate(r.names or []))

        per_class = []
        try:
            for i, c in enumerate(box.ap_class_index):
                cid = int(c)
                row = {
                    "class": names.get(cid, f"id={cid}"),
                    "ap50":    float(box.ap50[i]),
                    "ap50_95": float(box.ap[i].mean()),
                    "precision": float(box.p[i]),
                    "recall":    float(box.r[i]),
                }
                if seg is not None:
                    try:
                        row["mask_ap50"]    = float(seg.ap50[i])
                        row["mask_ap50_95"] = float(seg.ap[i].mean())
                    except Exception:
                        pass
                per_class.append(row)
        except Exception:
            pass

        speed = getattr(r, "speed", {}) or {}
        res.update({
            "ok": True,
            "task": getattr(model, "task", None),
            "map50":    float(box.map50),
            "map50_95": float(box.map),
            "precision": float(box.mp),
            "recall":    float(box.mr),
            "fitness":   float(getattr(r, "fitness", 0.0) or 0.0),
            "per_class": per_class,
            "speed_ms":  {k: round(float(v), 2) for k, v in speed.items()},
            "plots_dir": str(out_dir) if out_dir.exists() else None,
        })
        if seg is not None:
            try:
                res["mask_map50"]    = float(seg.map50)
                res["mask_map50_95"] = float(seg.map)
            except Exception:
                pass
    except Exception as e:
        res["error"] = f"{type(e).__name__}: {e}"
    return res


def collect_model_evals(dataset_key: Optional[str] = None) -> list[dict]:
    """全モデルの保存済み評価結果を集める（モデル横断の比較表に使う）"""
    rows = []
    if not MODELS_DIR.exists():
        return rows
    for mp in MODELS_DIR.rglob("*.pt"):
        for key, r in read_model_evals(mp).items():
            if dataset_key and key != dataset_key:
                continue
            if not r.get("ok"):
                continue
            rows.append({"model_path": mp, "dataset_key": key, **r})
    return rows


def _eval_worker(model_paths: list[str], data_yaml: str, split: str,
                 imgsz: int, batch: int, conf: float, iou: float) -> None:
    """複数モデルを順に評価する（バックグラウンドスレッド）"""
    state, lock = _get_eval_shared()

    def _log(msg: str) -> None:
        with lock:
            state["log"].append(msg)

    dataset_key = f"{Path(data_yaml).parent.name}:{split}"
    try:
        for i, mp_str in enumerate(model_paths):
            mp = Path(mp_str)
            with lock:
                state["current"] = mp.name
                state["done"] = i
            _log(f"[{i + 1}/{len(model_paths)}] 評価中: {mp.relative_to(MODELS_DIR)}")

            r = evaluate_model(mp, data_yaml, split=split, imgsz=imgsz,
                               batch=batch, conf=conf, iou=iou)
            if r["ok"]:
                save_model_eval(mp, dataset_key, r)
                _log(f"    mAP50={r['map50']:.4f}  mAP50-95={r['map50_95']:.4f}  "
                     f"P={r['precision']:.3f}  R={r['recall']:.3f}")
            else:
                _log(f"    ❌ 失敗: {r['error']}")

            with lock:
                state["results"].append({"model": str(mp), **r})
                state["done"] = i + 1
        _log("")
        _log("✅ 評価が完了しました")
    except Exception as e:
        with lock:
            state["error"] = f"{type(e).__name__}: {e}"
    finally:
        with lock:
            state["running"] = False
            state["finished"] = True
            state["current"] = ""


def start_evaluation(model_paths: list[str], data_yaml: str, split: str,
                     imgsz: int, batch: int, conf: float, iou: float) -> None:
    state, lock = _get_eval_shared()
    with lock:
        if state["running"]:
            return
        state.update({"log": [], "running": True, "error": None, "finished": False,
                      "total": len(model_paths), "done": 0, "current": "", "results": []})
    threading.Thread(
        target=_eval_worker,
        args=(model_paths, data_yaml, split, imgsz, batch, conf, iou),
        daemon=True,
    ).start()
