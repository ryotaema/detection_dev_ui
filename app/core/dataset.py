# =============================================================================
# データセットの生成・検査・修正・書き出し
# =============================================================================
from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path
from typing import Optional

import streamlit as st

from .config import IMG_EXTS
from .provenance import record_dataset_provenance


def generate_yolo_dataset(
    raw_dir: Path,
    xml_info: dict,
    selected_labels: list[str],
    task_type: str,
    out_dir: Path,
    val_ratio: float = 0.2,
    cvat_tasks: Optional[list[dict]] = None,
) -> Optional[Path]:
    """選択ラベル × タスク種別で YOLO 形式データセットを生成する。

    - 画像は raw_dir 内から探してシンボリックリンクを張る（大容量でもコピー不要）
    - train/val は annotated サンプルを 80/20 でランダム分割
    - data.yaml は絶対パス + names リスト形式で生成
    """
    import xml.etree.ElementTree as ET
    import random
    import yaml

    label2id = {lbl: i for i, lbl in enumerate(selected_labels)}
    xml_path = Path(xml_info["xml_path"])

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 画像ディレクトリを探す（ZIPの構造に依存するため複数候補）
    img_roots: list[Path] = []
    for d in raw_dir.rglob("*"):
        if d.is_dir() and d.name in ("images", "train", "val"):
            img_roots.append(d)
    if not img_roots:
        img_roots = [raw_dir]

    def _find_image(name: str) -> Optional[Path]:
        for base in img_roots:
            p = base / name
            if p.exists():
                return p
        for p in raw_dir.rglob(Path(name).name):
            if p.is_file():
                return p
        return None

    def _box_to_detect(box, w: int, h: int) -> str:
        xtl, ytl = float(box.get("xtl", 0)), float(box.get("ytl", 0))
        xbr, ybr = float(box.get("xbr", 0)), float(box.get("ybr", 0))
        cx = (xtl + xbr) / 2 / w
        cy = (ytl + ybr) / 2 / h
        bw = (xbr - xtl) / w
        bh = (ybr - ytl) / h
        return f"{cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"

    def _box_to_obb(box, w: int, h: int) -> str:
        """CVAT の box を OBB 形式（4隅の正規化座標）に変換する。
        CVAT の box は rotation 属性（度・中心周り）を持つことがある。
        """
        import math

        xtl, ytl = float(box.get("xtl", 0)), float(box.get("ytl", 0))
        xbr, ybr = float(box.get("xbr", 0)), float(box.get("ybr", 0))
        rot = float(box.get("rotation", 0) or 0)
        corners = [(xtl, ytl), (xbr, ytl), (xbr, ybr), (xtl, ybr)]
        if rot:
            cx, cy = (xtl + xbr) / 2, (ytl + ybr) / 2
            a = math.radians(rot)
            ca, sa = math.cos(a), math.sin(a)
            corners = [
                (cx + (x - cx) * ca - (y - cy) * sa,
                 cy + (x - cx) * sa + (y - cy) * ca)
                for x, y in corners
            ]
        return " ".join(f"{x / w:.6f} {y / h:.6f}" for x, y in corners)

    def _polygon_to_obb(polygon, w: int, h: int) -> Optional[str]:
        """4点ポリゴンをそのまま OBB として使う（点数が違う場合は外接矩形で代用）"""
        pts = []
        for pt in polygon.get("points", "").split(";"):
            pt = pt.strip()
            if "," in pt:
                x, y = pt.split(",")
                pts.append((float(x), float(y)))
        if len(pts) == 4:
            return " ".join(f"{x / w:.6f} {y / h:.6f}" for x, y in pts)
        if len(pts) >= 3:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            return " ".join(f"{x / w:.6f} {y / h:.6f}"
                            for x, y in [(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
        return None

    def _polygon_to_segment(polygon, w: int, h: int) -> str:
        pts = []
        for pt in polygon.get("points", "").split(";"):
            pt = pt.strip()
            if "," in pt:
                x, y = pt.split(",")
                pts.append(f"{float(x)/w:.6f} {float(y)/h:.6f}")
        return " ".join(pts)

    # ── 画像分類 (classify) ────────────────────────────────────────────────
    # YOLO の分類はラベル txt ではなく「クラス名ディレクトリ」でデータを表現する:
    #   <dataset>/train/<class>/xxx.jpg
    # 元データは CVAT の tag（画像単位のラベル）。
    if task_type == "classify":
        cls_samples: list[dict] = []
        for img_elem in root.findall("image"):
            img_name = img_elem.get("name", "")
            tags = [t.get("label", "") for t in img_elem.findall("tag")]
            tags = [t for t in tags if t in label2id]
            if not tags:
                continue
            # 分類は1画像1クラス。複数付いている場合は最初のものを採用する
            cls_samples.append({"name": img_name, "cls": tags[0]})

        if not cls_samples:
            st.error(
                "選択したラベルの tag（画像単位のラベル）が見つかりません。"
                "画像分類では CVAT で矩形ではなく「タグ」を付ける必要があります。"
            )
            return None

        random.shuffle(cls_samples)
        split = max(1, int(len(cls_samples) * (1 - val_ratio)))
        cls_splits = {"train": cls_samples[:split], "val": cls_samples[split:]}

        used_classes: list[str] = []
        for sp, sp_samples in cls_splits.items():
            for s in sp_samples:
                img_src = _find_image(s["name"])
                if img_src is None:
                    continue
                dst_dir = out_dir / sp / s["cls"]
                dst_dir.mkdir(parents=True, exist_ok=True)
                img_dst = dst_dir / img_src.name
                if not img_dst.exists():
                    try:
                        img_dst.symlink_to(img_src.resolve())
                    except Exception:
                        import shutil
                        shutil.copy2(img_src, img_dst)
                if s["cls"] not in used_classes:
                    used_classes.append(s["cls"])

        # 学習時は data=<このディレクトリ> を渡す。data.yaml は
        # 「このUIがデータセットとして認識するため」のメタ情報として置く。
        cfg = {
            "path": str(out_dir.resolve()),
            "train": "train",
            "val": "val",
            "task": "classify",
            "nc": len(used_classes),
            "names": sorted(used_classes),
        }
        with open(out_dir / "data.yaml", "w") as f:
            yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)
        record_dataset_provenance(
            out_dir, source="cvat", task_type=task_type,
            labels=sorted(used_classes), cvat_tasks=cvat_tasks,
            extra={"val_ratio": val_ratio, "xml_path": str(xml_path)},
        )
        return out_dir

    samples: list[dict] = []
    for img_elem in root.findall("image"):
        img_name = img_elem.get("name", "")
        w = int(img_elem.get("width", 1))
        h = int(img_elem.get("height", 1))
        lines: list[str] = []

        if task_type == "detect":
            for box in img_elem.findall("box"):
                lbl = box.get("label", "")
                if lbl not in label2id:
                    continue
                lines.append(f"{label2id[lbl]} {_box_to_detect(box, w, h)}")

        elif task_type == "segment":
            for polygon in img_elem.findall("polygon"):
                lbl = polygon.get("label", "")
                if lbl not in label2id:
                    continue
                seg = _polygon_to_segment(polygon, w, h)
                if seg:
                    lines.append(f"{label2id[lbl]} {seg}")
            for box in img_elem.findall("box"):
                lbl = box.get("label", "")
                if lbl not in label2id:
                    continue
                xtl, ytl = float(box.get("xtl", 0)), float(box.get("ytl", 0))
                xbr, ybr = float(box.get("xbr", 0)), float(box.get("ybr", 0))
                pts = " ".join([
                    f"{xtl/w:.6f} {ytl/h:.6f}",
                    f"{xbr/w:.6f} {ytl/h:.6f}",
                    f"{xbr/w:.6f} {ybr/h:.6f}",
                    f"{xtl/w:.6f} {ybr/h:.6f}",
                ])
                lines.append(f"{label2id[lbl]} {pts}")

        elif task_type == "obb":
            # 回転BBOX。CVAT の回転付き box と 4点ポリゴンの両方から作れる
            for box in img_elem.findall("box"):
                lbl = box.get("label", "")
                if lbl not in label2id:
                    continue
                lines.append(f"{label2id[lbl]} {_box_to_obb(box, w, h)}")
            for polygon in img_elem.findall("polygon"):
                lbl = polygon.get("label", "")
                if lbl not in label2id:
                    continue
                obb = _polygon_to_obb(polygon, w, h)
                if obb:
                    lines.append(f"{label2id[lbl]} {obb}")

        elif task_type == "pose":
            for box in img_elem.findall("box"):
                lbl = box.get("label", "")
                if lbl not in label2id:
                    continue
                lines.append(f"{label2id[lbl]} {_box_to_detect(box, w, h)}")

        if lines:
            samples.append({"name": img_name, "lines": lines})

    if not samples:
        st.error("選択したラベルにマッチするアノテーションがありません")
        return None

    random.shuffle(samples)
    split = max(1, int(len(samples) * (1 - val_ratio)))
    splits = {"train": samples[:split], "val": samples[split:]}

    for sp in ("train", "val"):
        (out_dir / "images" / sp).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / sp).mkdir(parents=True, exist_ok=True)

    for sp, sp_samples in splits.items():
        for s in sp_samples:
            img_src = _find_image(s["name"])
            if img_src is None:
                continue
            stem = Path(s["name"]).stem
            img_dst = out_dir / "images" / sp / img_src.name
            lbl_dst = out_dir / "labels" / sp / f"{stem}.txt"
            if not img_dst.exists():
                try:
                    img_dst.symlink_to(img_src.resolve())
                except Exception:
                    import shutil
                    shutil.copy2(img_src, img_dst)
            with open(lbl_dst, "w") as f:
                f.write("\n".join(s["lines"]))

    cfg = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "task": task_type,      # detect / segment / pose / obb
        "nc": len(selected_labels),
        "names": selected_labels,
    }
    with open(out_dir / "data.yaml", "w") as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)

    record_dataset_provenance(
        out_dir, source="cvat", task_type=task_type,
        labels=selected_labels, cvat_tasks=cvat_tasks,
        extra={"val_ratio": val_ratio, "xml_path": str(xml_path)},
    )
    return out_dir


def check_dataset_quality(dataset_dir: Path, tiny_area: float = 0.0005) -> dict:
    """YOLO 形式データセットの整合性を検査する。

    Returns: {
      "classes": [...], "splits": {split: {...}}, "class_counts": {name: n},
      "issues": [{"severity","kind","path","detail"}], "n_issues": int,
    }
    """
    res: dict = {
        "dataset": dataset_dir.name,
        "classes": [],
        "splits": {},
        "class_counts": {},
        "issues": [],          # 詳細（種別ごとに ISSUE_CAP 件まで）
        "issue_counts": {},    # 種別ごとの総数
        "n_issues": 0,
        "error": None,
    }

    # 同じ種別の指摘が数千件出ると読めなくなるため、詳細は種別ごとに打ち切る
    ISSUE_CAP = 20
    _kind_counts: dict[str, int] = {}
    _kind_sev: dict[str, str] = {}

    def _issue(sev: str, kind: str, path: str, detail: str) -> None:
        n = _kind_counts.get(kind, 0) + 1
        _kind_counts[kind] = n
        _kind_sev.setdefault(kind, sev)
        if sev == "error":
            _kind_sev[kind] = "error"
        if n <= ISSUE_CAP:
            res["issues"].append({"severity": sev, "kind": kind, "path": path, "detail": detail})

    # data.yaml からクラス名を読む（無くても検査は続行する）
    yaml_path = next(iter(sorted(dataset_dir.rglob("data.yaml"))), None)
    if yaml_path:
        try:
            import yaml as _yml
            names = (_yml.safe_load(yaml_path.read_text()) or {}).get("names")
            if isinstance(names, dict):
                res["classes"] = [names[k] for k in sorted(names)]
            elif isinstance(names, list):
                res["classes"] = list(names)
        except Exception as e:
            _issue("warn", "data.yaml", str(yaml_path), f"読み込めません: {e}")
    else:
        _issue("warn", "data.yaml", str(dataset_dir), "data.yaml が見つかりません")

    n_classes = len(res["classes"])
    _ds_task = dataset_task_type(str(yaml_path)) if yaml_path else "detect"
    counts: dict[str, int] = {}

    # ── 画像分類: images/labels ではなく <split>/<クラス名>/ 構造 ──────────
    if yaml_path and dataset_task_type(str(yaml_path)) == "classify":
        counts_cls: dict[str, int] = {}
        for sp in ("train", "val", "test"):
            sp_dir = dataset_dir / sp
            if not sp_dir.exists():
                continue
            n_img, n_cls = 0, 0
            for cdir in sorted(p for p in sp_dir.iterdir() if p.is_dir()):
                imgs = [p for p in cdir.iterdir()
                        if p.is_file() and p.suffix.lower() in IMG_EXTS]
                if not imgs:
                    _issue("warn", "空のクラス", f"{sp}/{cdir.name}",
                           "画像が1枚もありません")
                n_cls += 1
                n_img += len(imgs)
                counts_cls[cdir.name] = counts_cls.get(cdir.name, 0) + len(imgs)
            res["splits"][sp] = {"images": n_img, "labels": n_cls, "missing_label": 0,
                                 "orphan_label": 0, "empty_label": 0, "boxes": n_img}
        res["class_counts"] = counts_cls
        if "train" in res["splits"] and "val" not in res["splits"]:
            _issue("warn", "スプリット", res["dataset"], "val がありません（評価ができません）")
        if len(counts_cls) >= 2:
            mx, mn = max(counts_cls.values()), min(counts_cls.values())
            if mn > 0 and mx / mn >= 20:
                _issue("warn", "クラス分布の偏り", res["dataset"],
                       f"最多 {mx} 枚 / 最少 {mn} 枚 — 少数クラスの精度が出ない可能性があります")
        res["issue_counts"] = {k: {"count": v, "severity": _kind_sev.get(k, "warn")}
                               for k, v in _kind_counts.items()}
        res["n_issues"] = sum(_kind_counts.values())
        res["n_errors"] = sum(v for k, v in _kind_counts.items()
                              if _kind_sev.get(k) == "error")
        return res

    img_root = dataset_dir / "images"
    lbl_root = dataset_dir / "labels"
    if not img_root.exists():
        res["error"] = (
            "images/ ディレクトリがありません。YOLO 形式ではありません"
            "（CVAT の raw エクスポートのままの可能性があります。"
            "Step2: データ取込 の「データセット生成」で YOLO 形式に変換してください）"
        )
        return res

    splits = sorted([d.name for d in img_root.iterdir() if d.is_dir()])
    for sp in splits:
        img_dir, lbl_dir = img_root / sp, lbl_root / sp
        images = sorted(p for p in img_dir.iterdir()
                        if p.is_file() and p.suffix.lower() in IMG_EXTS)

        # labels/<split>/ 自体が無い場合は画像1枚ずつ警告しても意味がないので集約する
        if not lbl_dir.exists():
            _issue("error", "labelsディレクトリ無し", f"labels/{sp}",
                   f"images/{sp}/ に {len(images)} 枚ありますが labels/{sp}/ がありません"
                   "（アノテーション未取込のため、このままでは学習できません）")
            res["splits"][sp] = {"images": len(images), "labels": 0, "missing_label": len(images),
                                 "orphan_label": 0, "empty_label": 0, "boxes": 0}
            continue

        labels = sorted(p for p in lbl_dir.glob("*.txt"))

        img_stems = {p.stem for p in images}
        lbl_stems = {p.stem for p in labels}

        stat = {
            "images": len(images), "labels": len(labels),
            "missing_label": 0, "orphan_label": 0, "empty_label": 0, "boxes": 0,
        }

        # 画像はあるがラベルが無い = 未アノテーション（背景画像として意図的な場合もある）
        for stem in sorted(img_stems - lbl_stems):
            stat["missing_label"] += 1
            _issue("warn", "ラベル無し画像", f"{sp}/{stem}",
                   "対応する .txt がありません（未アノテーション、または背景画像）")

        # ラベルはあるが画像が無い = 学習に使われない迷子ファイル
        for stem in sorted(lbl_stems - img_stems):
            stat["orphan_label"] += 1
            _issue("error", "画像無しラベル", f"{sp}/{stem}.txt",
                   "対応する画像がありません（このラベルは学習に使われません）")

        for lp in labels:
            try:
                lines = [l.strip() for l in lp.read_text().splitlines() if l.strip()]
            except Exception as e:
                _issue("error", "読み込み失敗", f"{sp}/{lp.name}", str(e))
                continue

            if not lines:
                stat["empty_label"] += 1
                continue

            for ln_no, line in enumerate(lines, 1):
                parts = line.split()
                if len(parts) < 5:
                    _issue("error", "行フォーマット", f"{sp}/{lp.name}:{ln_no}",
                           f"フィールド数が不足しています ({len(parts)})")
                    continue
                try:
                    cls_id = int(float(parts[0]))
                    coords = [float(v) for v in parts[1:]]
                except ValueError:
                    _issue("error", "数値変換", f"{sp}/{lp.name}:{ln_no}",
                           "数値として解釈できない値があります")
                    continue

                stat["boxes"] += 1
                cls_name = (res["classes"][cls_id]
                            if 0 <= cls_id < n_classes else f"id={cls_id}")
                counts[cls_name] = counts.get(cls_name, 0) + 1

                if n_classes and not (0 <= cls_id < n_classes):
                    _issue("error", "クラスID範囲外", f"{sp}/{lp.name}:{ln_no}",
                           f"クラスID {cls_id} は data.yaml の {n_classes} クラスの範囲外です")

                if any(c < -1e-6 or c > 1 + 1e-6 for c in coords):
                    if _ds_task == "obb":
                        # 回転BBOX は角が画像外にはみ出しうるので異常ではない
                        _issue("warn", "座標範囲外(OBB)", f"{sp}/{lp.name}:{ln_no}",
                               "回転により画像外にはみ出しています（OBB では許容されます）")
                    else:
                        _issue("error", "座標範囲外", f"{sp}/{lp.name}:{ln_no}",
                               "正規化座標が 0〜1 の範囲を超えています")

                # detect 形式 (cx cy w h) のみ面積を検査する
                if len(coords) == 4:
                    bw, bh = coords[2], coords[3]
                    if bw <= 0 or bh <= 0:
                        _issue("error", "サイズ不正", f"{sp}/{lp.name}:{ln_no}",
                               f"幅または高さが 0 以下です (w={bw:.4f}, h={bh:.4f})")
                    elif bw * bh < tiny_area:
                        _issue("warn", "極小ボックス", f"{sp}/{lp.name}:{ln_no}",
                               f"面積比 {bw * bh:.5f} — ノイズの可能性があります")

        res["splits"][sp] = stat

    res["class_counts"] = counts

    # クラス分布の偏り（最多と最少で 20 倍以上開いていたら警告）
    if len(counts) >= 2:
        mx, mn = max(counts.values()), min(counts.values())
        if mn > 0 and mx / mn >= 20:
            _issue("warn", "クラス分布の偏り", res["dataset"],
                   f"最多 {mx} 件 / 最少 {mn} 件 — 少数クラスの精度が出ない可能性があります")

    # train/val のどちらかが欠けている
    if "train" in res["splits"] and "val" not in res["splits"]:
        _issue("warn", "スプリット", res["dataset"], "val がありません（評価ができません）")

    res["issue_counts"] = {
        k: {"count": v, "severity": _kind_sev.get(k, "warn")} for k, v in _kind_counts.items()
    }
    res["n_issues"] = sum(_kind_counts.values())
    res["n_errors"] = sum(v for k, v in _kind_counts.items() if _kind_sev.get(k) == "error")
    return res


def dataset_split_counts(dataset_dir: Path) -> dict:
    """現在の train/val の枚数を返す（再分割前の確認用）"""
    from .provenance import count_dataset_items
    return count_dataset_items(Path(dataset_dir))


def resplit_dataset(
    dataset_dir: Path,
    val_ratio: float = 0.2,
    seed: int = 0,
) -> dict:
    """既存データセットの train / val を混ぜ直して分割し直す。

    生成時に決めた比率のままでは「val が偏っていて評価が信用できない」ときに
    手が出せないため。画像とラベルを対で動かす。
    classify はクラスごとに比率を保って分割する（層化）。

    ※ ファイルを移動するだけなので、何度でもやり直せる。
    """
    import random
    import shutil

    res = {"ok": False, "task": "", "moved": 0, "before": {}, "after": {},
           "error": None}
    ds = Path(dataset_dir)
    res["before"] = dataset_split_counts(ds)

    if not (0.01 <= val_ratio <= 0.9):
        res["error"] = "val の割合は 0.01〜0.9 の範囲で指定してください"
        return res

    rng = random.Random(seed)
    yaml_path = ds / "data.yaml"
    task = dataset_task_type(str(yaml_path)) if yaml_path.exists() else "detect"
    res["task"] = task

    def _move(src: Path, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.resolve() == dst.resolve():
            return
        # シンボリックリンクを壊さないよう、リンク自体を張り直す
        if src.is_symlink():
            target = os.readlink(src)
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            os.symlink(target, dst)
            src.unlink()
        else:
            shutil.move(str(src), str(dst))

    try:
        if task == "classify":
            # クラスごとに集めて、クラス内で分割する（少数クラスが片側に寄らないように）
            per_class: dict[str, list[Path]] = {}
            for sp in ("train", "val"):
                sp_dir = ds / sp
                if not sp_dir.exists():
                    continue
                for cdir in sp_dir.iterdir():
                    if not cdir.is_dir():
                        continue
                    per_class.setdefault(cdir.name, []).extend(
                        p for p in cdir.iterdir()
                        if p.is_file() and p.suffix.lower() in IMG_EXTS
                    )
            if not per_class:
                res["error"] = "画像が見つかりません"
                return res

            for cname, files in per_class.items():
                rng.shuffle(files)
                n_val = max(1, int(len(files) * val_ratio)) if len(files) > 1 else 0
                for i, f in enumerate(files):
                    sp = "val" if i < n_val else "train"
                    dst = ds / sp / cname / f.name
                    if f.parent != dst.parent:
                        _move(f, dst)
                        res["moved"] += 1
        else:
            img_root, lbl_root = ds / "images", ds / "labels"
            if not img_root.exists():
                res["error"] = "images/ がありません（YOLO 形式ではありません）"
                return res

            samples: list[tuple[Path, Optional[Path]]] = []
            for sp in ("train", "val"):
                sp_dir = img_root / sp
                if not sp_dir.exists():
                    continue
                for img in sorted(sp_dir.iterdir()):
                    if not img.is_file() or img.suffix.lower() not in IMG_EXTS:
                        continue
                    lbl = lbl_root / sp / f"{img.stem}.txt"
                    samples.append((img, lbl if lbl.exists() else None))

            if not samples:
                res["error"] = "画像が見つかりません"
                return res

            rng.shuffle(samples)
            n_val = max(1, int(len(samples) * val_ratio)) if len(samples) > 1 else 0
            for i, (img, lbl) in enumerate(samples):
                sp = "val" if i < n_val else "train"
                img_dst = img_root / sp / img.name
                if img.parent != img_dst.parent:
                    _move(img, img_dst)
                    res["moved"] += 1
                if lbl is not None:
                    lbl_dst = lbl_root / sp / lbl.name
                    if lbl.parent != lbl_dst.parent:
                        _move(lbl, lbl_dst)

        res["after"] = dataset_split_counts(ds)
        res["ok"] = True
    except Exception as e:
        res["error"] = f"{type(e).__name__}: {e}"
    return res


def dataset_class_names(dataset_dir: Path) -> list[str]:
    """data.yaml のクラス名を順序どおりに返す（インデックス = クラスID）"""
    yaml_path = Path(dataset_dir) / "data.yaml"
    if not yaml_path.exists():
        return []
    try:
        import yaml as _y
        names = (_y.safe_load(yaml_path.read_text()) or {}).get("names")
        if isinstance(names, dict):
            return [names[k] for k in sorted(names)]
        if isinstance(names, list):
            return list(names)
    except Exception:
        pass
    return []


def remap_dataset_classes(
    dataset_dir: Path,
    mapping: dict[str, Optional[str]],
    backup: bool = True,
) -> dict:
    """クラス名のリネーム・統合・削除を行う。

    mapping: {現在のクラス名: 新しいクラス名}。値が None または空文字なら
             そのクラスのアノテーションを削除する。
             複数の旧名に同じ新名を割り当てると統合になる。

    ラベル txt のクラスID を振り直し、data.yaml の names を書き換える。
    書き換える .txt は `.txt.bak` にバックアップしてから上書きする。
    """
    import shutil

    res = {"ok": False, "task": "", "old_classes": [], "new_classes": [],
           "files_changed": 0, "lines_removed": 0, "dirs_merged": 0, "error": None}

    ds = Path(dataset_dir)
    yaml_path = ds / "data.yaml"
    if not yaml_path.exists():
        res["error"] = "data.yaml がありません"
        return res

    old_classes = dataset_class_names(ds)
    res["old_classes"] = old_classes
    task = dataset_task_type(str(yaml_path))
    res["task"] = task

    # 新しいクラス一覧（元の並び順を保ちつつ、統合先をまとめる）
    new_classes: list[str] = []
    for c in old_classes:
        new = mapping.get(c, c)
        if new and new not in new_classes:
            new_classes.append(new)
    if not new_classes:
        res["error"] = "すべてのクラスが削除対象です。1つ以上残してください。"
        return res
    res["new_classes"] = new_classes

    # 旧クラスID → 新クラスID（None は削除）
    id_map: dict[int, Optional[int]] = {}
    for i, c in enumerate(old_classes):
        new = mapping.get(c, c)
        id_map[i] = new_classes.index(new) if new else None

    try:
        if task == "classify":
            # ディレクトリ名の変更・統合
            for sp in ("train", "val", "test"):
                sp_dir = ds / sp
                if not sp_dir.exists():
                    continue
                for cdir in sorted(p for p in sp_dir.iterdir() if p.is_dir()):
                    new = mapping.get(cdir.name, cdir.name)
                    if not new:
                        shutil.rmtree(cdir)
                        res["dirs_merged"] += 1
                        continue
                    if new == cdir.name:
                        continue
                    dst = sp_dir / new
                    if dst.exists():
                        for f in cdir.iterdir():       # 統合: 中身を移してから削除
                            target = dst / f.name
                            if target.exists():
                                target = dst / f"{f.stem}_{cdir.name}{f.suffix}"
                            shutil.move(str(f), str(target))
                        cdir.rmdir()
                    else:
                        cdir.rename(dst)
                    res["dirs_merged"] += 1
        else:
            lbl_root = ds / "labels"
            if lbl_root.exists():
                for lbl_dir in sorted(p for p in lbl_root.iterdir() if p.is_dir()):
                    for lp in sorted(lbl_dir.glob("*.txt")):
                        try:
                            lines = lp.read_text().splitlines()
                        except Exception:
                            continue
                        kept, removed, changed = [], 0, False
                        for line in lines:
                            s = line.strip()
                            if not s:
                                continue
                            parts = s.split()
                            try:
                                cid = int(float(parts[0]))
                            except (ValueError, IndexError):
                                continue
                            new_id = id_map.get(cid, cid)
                            if new_id is None:
                                removed += 1
                                changed = True
                                continue
                            if new_id != cid:
                                changed = True
                            kept.append(" ".join([str(new_id)] + parts[1:]))
                        if changed:
                            if backup:
                                lp.with_suffix(".txt.bak").write_text(
                                    "\n".join(lines) + "\n")
                            lp.write_text(("\n".join(kept) + "\n") if kept else "")
                            res["files_changed"] += 1
                            res["lines_removed"] += removed

        # data.yaml を更新
        import yaml as _y
        cfg = _y.safe_load(yaml_path.read_text()) or {}
        cfg["names"] = new_classes
        cfg["nc"] = len(new_classes)
        yaml_path.write_text(_y.dump(cfg, allow_unicode=True,
                                     default_flow_style=False))
        res["ok"] = True
    except Exception as e:
        res["error"] = f"{type(e).__name__}: {e}"
    return res


def dataset_size_bytes(dataset_dir: Path, labels_only: bool = False) -> int:
    """データセットの概算サイズ（ZIP 生成前の警告用）"""
    total = 0
    for p in dataset_dir.rglob("*"):
        if not p.is_file() or p.name.endswith(".bak"):
            continue
        if labels_only and p.suffix.lower() in IMG_EXTS:
            continue
        total += p.stat().st_size
    return total


def build_dataset_zip(dataset_dir: Path, out_path: Path,
                      labels_only: bool = False) -> tuple[bool, str, int]:
    """データセットを ZIP に固めてディスクへ書き出す。

    画像を含めると数 GB になりうるためメモリ上には載せず、
    一時ファイル経由で st.download_button に渡す。
    Returns: (成功, メッセージ, ファイル数)
    """
    try:
        n = 0
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for p in sorted(dataset_dir.rglob("*")):
                if not p.is_file() or p.name.endswith(".bak"):
                    continue
                if labels_only and p.suffix.lower() in IMG_EXTS:
                    continue
                zf.write(p, arcname=str(p.relative_to(dataset_dir)))
                n += 1
        return True, str(out_path), n
    except Exception as e:
        return False, f"{type(e).__name__}: {e}", 0


def fix_dataset_labels(
    dataset_dir: Path,
    drop_invalid_size: bool = True,
    drop_out_of_range: bool = True,
    drop_tiny: bool = False,
    delete_orphan_labels: bool = False,
    tiny_area: float = 0.0005,
) -> dict:
    """品質チェックで見つかった壊れたラベルを修正する。

    書き換える .txt は同じディレクトリに `<name>.txt.bak` としてバックアップしてから
    上書きする（元に戻せるようにするため）。
    """
    res = {"files_changed": 0, "lines_removed": 0, "files_emptied": 0,
           "orphans_deleted": 0, "backup_suffix": ".bak", "details": [], "error": None}

    img_root, lbl_root = dataset_dir / "images", dataset_dir / "labels"
    if not lbl_root.exists():
        res["error"] = "labels/ ディレクトリがありません"
        return res

    for lbl_dir in sorted(p for p in lbl_root.iterdir() if p.is_dir()):
        sp = lbl_dir.name
        img_dir = img_root / sp
        img_stems = ({p.stem for p in img_dir.iterdir()
                      if p.is_file() and p.suffix.lower() in IMG_EXTS}
                     if img_dir.exists() else set())

        for lp in sorted(lbl_dir.glob("*.txt")):
            # 画像が存在しないラベルの削除
            if delete_orphan_labels and img_stems and lp.stem not in img_stems:
                lp.rename(lp.with_suffix(".txt.bak"))
                res["orphans_deleted"] += 1
                res["details"].append(f"{sp}/{lp.name}: 画像が無いため退避")
                continue

            try:
                lines = lp.read_text().splitlines()
            except Exception:
                continue

            kept, removed = [], 0
            for line in lines:
                s = line.strip()
                if not s:
                    continue
                parts = s.split()
                if len(parts) < 5:
                    removed += 1
                    continue
                try:
                    coords = [float(v) for v in parts[1:]]
                except ValueError:
                    removed += 1
                    continue

                if drop_out_of_range and any(c < -1e-6 or c > 1 + 1e-6 for c in coords):
                    removed += 1
                    continue
                if len(coords) == 4:
                    bw, bh = coords[2], coords[3]
                    if drop_invalid_size and (bw <= 0 or bh <= 0):
                        removed += 1
                        continue
                    if drop_tiny and (bw * bh) < tiny_area:
                        removed += 1
                        continue
                kept.append(s)

            if removed:
                lp.with_suffix(".txt.bak").write_text("\n".join(lines) + "\n")
                lp.write_text(("\n".join(kept) + "\n") if kept else "")
                res["files_changed"] += 1
                res["lines_removed"] += removed
                if not kept:
                    res["files_emptied"] += 1
                res["details"].append(f"{sp}/{lp.name}: {removed} 行を除去"
                                      + ("（空になりました）" if not kept else ""))

    return res


# ---------------------------------------------------------------------------
# GT vs 予測の差分分析
#
#   学習済みモデルの予測を正解ラベル(GT)と突き合わせ、画像ごとに
#   FN(取りこぼし) / FP(余計な検出) を数える。
#   精度の高いモデルが FN を出す画像は「モデルが悪い」だけでなく
#   「GT のアノテーションが漏れている」ことも多く、ラベルの抜けを見つける手段になる。
# ---------------------------------------------------------------------------
def _yolo_txt_to_xyxy(txt_path: Path, w: int, h: int,
                      names: list[str]) -> list[dict]:
    """YOLO ラベル txt を絶対座標の xyxy に変換する。

    detect 形式 (cls cx cy bw bh) と segment 形式 (cls x1 y1 x2 y2 ... 正規化ポリゴン)
    の両方を受け付ける。segment の場合はポリゴンの外接矩形を bbox として扱い、
    輪郭は mask_xy として保持する。
    """
    out = []
    if not txt_path.exists():
        return out
    try:
        for line in txt_path.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                cid = int(float(parts[0]))
                vals = [float(v) for v in parts[1:]]
            except ValueError:
                continue

            label = names[cid] if 0 <= cid < len(names) else f"id={cid}"

            if len(vals) == 4:
                cx, cy, bw, bh = vals
                x1, y1 = (cx - bw / 2) * w, (cy - bh / 2) * h
                x2, y2 = (cx + bw / 2) * w, (cy + bh / 2) * h
                out.append({
                    "label": label,
                    "bbox_xyxy": [x1, y1, x2, y2],
                    # FiftyOne 形式(左上+wh の正規化)
                    "bbox_xywhn": [cx - bw / 2, cy - bh / 2, bw, bh],
                })
            elif len(vals) >= 6 and len(vals) % 2 == 0:
                xs = [vals[i] * w for i in range(0, len(vals), 2)]
                ys = [vals[i] * h for i in range(1, len(vals), 2)]
                x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                out.append({
                    "label": label,
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "bbox_xywhn": [x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h],
                    "mask_xy":  [[x, y] for x, y in zip(xs, ys)],
                    "mask_xyn": [[vals[i], vals[i + 1]] for i in range(0, len(vals), 2)],
                })
    except Exception:
        pass
    return out


def dataset_task_type(data_yaml: str) -> str:
    """data.yaml からタスク種別を返す（未記載なら detect）。

    画像分類のデータセットはクラス名ディレクトリ構造で、`model.train()` には
    data.yaml ではなくディレクトリを渡す必要があるため、ここで区別する。
    """
    try:
        import yaml as _y
        cfg = _y.safe_load(Path(data_yaml).read_text()) or {}
        return str(cfg.get("task") or "detect")
    except Exception:
        return "detect"


def resolve_train_data_arg(data_yaml: str) -> str:
    """`model.train(data=...)` / `model.val(data=...)` に渡すべき値を返す。

    classify はデータセットのルートディレクトリ、それ以外は data.yaml のパス。
    """
    if dataset_task_type(data_yaml) == "classify":
        return str(Path(data_yaml).parent)
    return data_yaml
