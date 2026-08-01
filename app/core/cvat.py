# =============================================================================
# CVAT との入出力（取得・エクスポート・書き戻し）
# =============================================================================
from __future__ import annotations

import json
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import streamlit as st

from .config import (CVAT_HOST, CVAT_PASS, CVAT_USER, CVAT_WEB)



# ===========================================================================
# ヘルパー関数群
# ===========================================================================

# ---------------------------------------------------------------------------
# CVAT API クライアント
# ---------------------------------------------------------------------------
def get_cvat_client():
    """cvat-sdk の CvatClient を返す（接続失敗時は None）"""
    try:
        from cvat_sdk import make_client
        client = make_client(
            host=CVAT_HOST,
            credentials=(CVAT_USER, CVAT_PASS),
        )
        return client
    except Exception as e:
        st.error(f"CVAT接続エラー: {e}")
        return None


def fetch_cvat_tasks() -> list[dict]:
    """CVATのタスク一覧を取得する"""
    try:
        client = get_cvat_client()
        if not client:
            return []
        tasks = client.tasks.list()
        result = []
        for t in tasks:
            assignee_name = ""
            if hasattr(t, "assignee") and t.assignee:
                assignee_name = getattr(t.assignee, "username", "") or getattr(t.assignee, "email", "")
            result.append({
                "id": t.id,
                "name": t.name,
                "size": t.size,
                "status": t.status,
                "assignee": assignee_name,
            })
        return result
    except Exception as e:
        st.error(f"CVATタスク取得エラー: {e}")
        return []


def fetch_cvat_jobs() -> list[dict]:
    """CVAT のジョブ一覧を取得する。

    タスクの `status` は粗い（completed か否か）ため、実際の進捗はジョブ単位で見る。
    ジョブは stage(annotation→validation→acceptance) と state(new/in progress/completed)
    を持ち、担当者も「タスクの担当者」ではなくジョブ単位で割り当てられる。
    """
    try:
        client = get_cvat_client()
        if not client:
            return []

        # task_id → タスク名 の対応（ジョブ側はタスク名を持たない）
        task_names: dict[int, str] = {}
        try:
            for t in client.tasks.list():
                task_names[t.id] = t.name
        except Exception:
            pass

        rows = []
        for j in client.jobs.list():
            assignee = ""
            _asg = getattr(j, "assignee", None)
            if _asg:
                assignee = getattr(_asg, "username", "") or getattr(_asg, "email", "")

            start = getattr(j, "start_frame", 0) or 0
            stop  = getattr(j, "stop_frame", 0) or 0
            rows.append({
                "job_id":    j.id,
                "task_id":   getattr(j, "task_id", None),
                "task_name": task_names.get(getattr(j, "task_id", None), ""),
                "state":     str(getattr(j, "state", "") or ""),
                "stage":     str(getattr(j, "stage", "") or ""),
                "type":      str(getattr(j, "type", "") or ""),
                "assignee":  assignee,
                "frames":    max(stop - start + 1, 0),
                "updated":   getattr(j, "updated_date", None),
            })
        return rows
    except Exception as e:
        st.error(f"CVATジョブ取得エラー: {e}")
        return []


def fetch_cvat_task_labels(task_ids: list[int]) -> dict[str, list[str]]:
    """複数タスクIDからラベル名リストを返す {タスク名(ID:xx): [label, ...]}"""
    try:
        client = get_cvat_client()
        if not client:
            return {}
        result = {}
        for tid in task_ids:
            try:
                task = client.tasks.retrieve(tid)
                labels = task.get_labels()
                result[f"{task.name}  (ID: {tid})"] = [lb.name for lb in labels]
            except Exception as e:
                st.warning(f"タスクID {tid} のラベル取得失敗: {e}")
        return result
    except Exception as e:
        st.error(f"ラベル取得エラー: {e}")
        return {}


def export_cvat_task_raw(task_id: int, out_dir: Path) -> Optional[Path]:
    """指定タスクを「CVAT for images 1.1」(XML形式) でエクスポートし、
    out_dir/raw/ に解凍したパスを返す。
    CVAT v2.64.0 の非同期エクスポート API (POST→ポーリング→ダウンロード) を使用。
    """
    import requests as _requests

    raw_dir  = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / "cvat_export.zip"
    session  = _requests.Session()

    try:
        login = session.post(
            f"{CVAT_HOST}/api/auth/login",
            json={"username": CVAT_USER, "password": CVAT_PASS},
        )
        login.raise_for_status()
        token = login.json().get("key")
        session.headers.update({"Authorization": f"Token {token}"})

        export = session.post(
            f"{CVAT_HOST}/api/tasks/{task_id}/dataset/export",
            params={
                "save_images": "True",
                "format": "CVAT for images 1.1",
            },
        )
        export.raise_for_status()
        rq_id = export.json().get("rq_id")
        if not rq_id:
            st.error("エクスポートジョブID が取得できませんでした")
            return None

        result_url = None
        for _ in range(180):
            status_resp = session.get(f"{CVAT_HOST}/api/requests/{rq_id}")
            status_resp.raise_for_status()
            data = status_resp.json()
            status = data.get("status")
            if status == "finished":
                result_url = data.get("result_url")
                break
            elif status == "failed":
                st.error(f"エクスポートに失敗しました: {data}")
                return None
            time.sleep(1)
        else:
            st.error("エクスポートがタイムアウトしました（180秒）")
            return None

        dl = session.get(result_url, stream=True)
        dl.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in dl.iter_content(chunk_size=8192):
                f.write(chunk)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(raw_dir)
        zip_path.unlink()
        return raw_dir

    except Exception as e:
        st.error(f"CVATからのエクスポート中にエラーが発生しました: {e}")
        return None


def parse_cvat_xml(raw_dir: Path) -> Optional[dict]:
    """CVAT for images 1.1 のXMLを解析してメタ情報を返す。

    Returns:
        {
          "xml_path": str,
          "labels": [str, ...],           # タスク定義のラベル一覧
          "annotation_types": [str, ...], # 実際に使われている種別 (box/polygon/points)
          "image_count": int,
          "annotated_count": int,         # 1件以上アノテーション付きの画像数
        }
    """
    import xml.etree.ElementTree as ET

    xml_candidates = list(raw_dir.glob("**/*.xml"))
    if not xml_candidates:
        st.error("XMLファイルが見つかりません")
        return None

    xml_path = xml_candidates[0]
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # タスク定義のラベル一覧
        labels: list[str] = []
        for lbl in root.findall(".//meta/task/labels/label"):
            name = lbl.find("name")
            if name is not None and name.text:
                labels.append(name.text.strip())

        # 画像・アノテーション統計
        annotation_types: set[str] = set()
        image_count = 0
        annotated_count = 0
        for img in root.findall("image"):
            image_count += 1
            has_annot = False
            for child in img:
                # tag は画像単位のラベル（画像分類のアノテーション）
                if child.tag in ("box", "polygon", "polyline", "points", "ellipse", "tag"):
                    annotation_types.add(child.tag)
                    has_annot = True
            if has_annot:
                annotated_count += 1

        return {
            "xml_path": str(xml_path),
            "labels": labels,
            "annotation_types": sorted(annotation_types),
            "image_count": image_count,
            "annotated_count": annotated_count,
        }
    except Exception as e:
        st.error(f"XML解析エラー: {e}")
        return None


# ---------------------------------------------------------------------------
# CVAT への書き戻し (推論結果を新規タスクとして投入)
#
#   ZIP のダウンロード → 手動アップロードを不要にする。
#   予測ボックスが事前アノテーションとして入った状態でタスクが作られるので、
#   作業者は「ゼロから引く」のではなく「直す」だけで済む。
# ---------------------------------------------------------------------------
def _collect_prediction_items(json_paths: list[Path]) -> tuple[list[dict], list[str]]:
    """予測 JSON 群から (画像情報リスト, 出現ラベル一覧) を作る。
    元画像が見つからないもの・読めないものは除外する。
    """
    import cv2

    items: list[dict] = []
    labels: list[str] = []
    for jf in json_paths:
        try:
            pred = json.loads(Path(jf).read_text())
        except Exception:
            continue

        img_path = Path(pred.get("image_path", ""))
        if not img_path.exists():
            continue

        # 推論時に記録した寸法があれば画像を読み直さない（大量件数で効く）
        size = pred.get("image_size")
        if size and len(size) == 2:
            w, h = int(size[0]), int(size[1])
        else:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]

        boxes = pred.get("boxes", []) or []
        for b in boxes:
            lb = b.get("label", "")
            if lb and lb not in labels:
                labels.append(lb)

        items.append({"path": img_path, "width": w, "height": h, "boxes": boxes})
    return items, labels


def build_cvat_xml(items: list[dict], labels: list[str], task_name: str = "") -> str:
    """画像情報から CVAT for images 1.1 形式の annotations.xml を組み立てる"""
    import xml.etree.ElementTree as ET

    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"
    meta = ET.SubElement(root, "meta")
    task_el = ET.SubElement(meta, "task")
    if task_name:
        ET.SubElement(task_el, "name").text = task_name
    labels_el = ET.SubElement(task_el, "labels")
    for lb in labels:
        lb_el = ET.SubElement(labels_el, "label")
        ET.SubElement(lb_el, "name").text = lb
        ET.SubElement(lb_el, "attributes")

    for idx, it in enumerate(items):
        img_el = ET.SubElement(
            root, "image",
            id=str(idx), name=it["path"].name,
            width=str(it["width"]), height=str(it["height"]),
        )
        for b in it["boxes"]:
            conf = b.get("confidence")

            # マスクがあるものは polygon として書き出す（CVAT 側もポリゴンで開く）
            mask = b.get("mask_xy")
            if mask and len(mask) >= 3:
                pts = ";".join(f"{float(x):.2f},{float(y):.2f}" for x, y in mask)
                shape_el = ET.SubElement(
                    img_el, "polygon",
                    label=b.get("label", ""), points=pts, occluded="0",
                )
                if conf is not None:
                    ET.SubElement(shape_el, "attribute",
                                  name="confidence").text = f"{float(conf):.4f}"
                continue

            xyxy = b.get("bbox_xyxy")
            if not xyxy or len(xyxy) != 4:
                continue
            x1, y1, x2, y2 = [float(v) for v in xyxy]
            box_el = ET.SubElement(
                img_el, "box",
                label=b.get("label", ""),
                xtl=f"{x1:.2f}", ytl=f"{y1:.2f}",
                xbr=f"{x2:.2f}", ybr=f"{y2:.2f}",
                occluded="0",
            )
            if conf is not None:
                attr = ET.SubElement(box_el, "attribute", name="confidence")
                attr.text = f"{float(conf):.4f}"

    ET.indent(root)
    return ('<?xml version="1.0" encoding="utf-8"?>\n'
            + ET.tostring(root, encoding="unicode", xml_declaration=False))


def push_items_to_cvat(
    items: list[dict],
    labels: list[str],
    task_name: str,
    with_annotations: bool = True,
) -> dict:
    """画像情報リストから CVAT の新規タスクを作成する（送信処理の本体）。

    items: [{"path": Path, "width": int, "height": int, "boxes": [...]}]
    """
    import shutil as _sh
    import tempfile

    out = {"ok": False, "task_id": None, "url": "", "n_images": 0,
           "labels": [], "error": None}

    if not items:
        out["error"] = "送信できる画像がありません（元画像が見つからない可能性があります）"
        return out
    if not labels:
        out["error"] = ("ラベルが1つも決まりません。"
                        "検出ゼロの画像だけを送る場合は、タスクに付けるラベル名を指定してください。")
        return out

    client = get_cvat_client()
    if not client:
        out["error"] = "CVAT に接続できません"
        return out

    tmp_dir = Path(tempfile.mkdtemp(prefix="cvat_push_"))
    try:
        from cvat_sdk.api_client import models

        # 画像を一時ディレクトリへ集約（同名衝突は連番で回避）
        resources, used = [], set()
        for it in items:
            fname = it["path"].name
            if fname in used:
                fname = f"{it['path'].stem}_{len(used)}{it['path'].suffix}"
            used.add(fname)
            dst = tmp_dir / fname
            _sh.copy2(it["path"], dst)
            resources.append(dst)
            it["path"] = dst          # XML の name と実ファイル名を一致させる

        # マスク付きの結果を送る場合は polygon も引けるようにラベル種別を any にする
        _has_mask = any(b.get("mask_xy") for it in items for b in it.get("boxes", []))
        _label_type = "any" if _has_mask else "rectangle"
        spec = models.TaskWriteRequest(
            name=task_name,
            labels=[models.PatchedLabelRequest(name=lb, type=_label_type) for lb in labels],
        )

        ann_path = ""
        if with_annotations:
            ann_path = str(tmp_dir / "annotations.xml")
            Path(ann_path).write_text(build_cvat_xml(items, labels, task_name))

        task = client.tasks.create_from_data(
            spec=spec,
            resources=resources,
            annotation_path=ann_path,
            annotation_format="CVAT 1.1",
        )

        out.update({
            "ok": True,
            "task_id": task.id,
            "url": f"{CVAT_WEB}/tasks/{task.id}",
            "n_images": len(resources),
            "labels": labels,
        })
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    finally:
        _sh.rmtree(tmp_dir, ignore_errors=True)
    return out


def create_cvat_task_from_images(
    task_name: str,
    image_paths: list[Path],
    labels: list[str],
    label_type: str = "rectangle",
) -> dict:
    """画像だけを渡して CVAT の新規タスクを作る（アノテーションの入口）。

    これまで最初のタスク作成は CVAT を直接操作する必要があったため、
    UI 側から始められるようにする。
    label_type: rectangle / polygon / points / tag / any
    """
    import shutil as _sh
    import tempfile

    out = {"ok": False, "task_id": None, "url": "", "n_images": 0,
           "labels": [], "error": None}

    paths = [Path(p) for p in image_paths if Path(p).exists()]
    if not paths:
        out["error"] = "アップロードする画像がありません"
        return out
    if not labels:
        out["error"] = "ラベルを1つ以上指定してください"
        return out

    client = get_cvat_client()
    if not client:
        out["error"] = "CVAT に接続できません"
        return out

    tmp_dir = Path(tempfile.mkdtemp(prefix="cvat_new_"))
    try:
        from cvat_sdk.api_client import models

        # 同名ファイルがあると CVAT 側で扱いづらいので連番を振って退避する
        resources, used = [], set()
        for p in paths:
            fname = p.name
            if fname in used:
                fname = f"{p.stem}_{len(used)}{p.suffix}"
            used.add(fname)
            dst = tmp_dir / fname
            _sh.copy2(p, dst)
            resources.append(dst)

        spec = models.TaskWriteRequest(
            name=task_name,
            labels=[models.PatchedLabelRequest(name=lb, type=label_type)
                    for lb in labels],
        )
        task = client.tasks.create_from_data(spec=spec, resources=resources)

        out.update({
            "ok": True,
            "task_id": task.id,
            "url": f"{CVAT_WEB}/tasks/{task.id}",
            "n_images": len(resources),
            "labels": list(labels),
        })
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    finally:
        _sh.rmtree(tmp_dir, ignore_errors=True)
    return out


def push_predictions_to_cvat(
    json_paths: list[Path],
    task_name: str,
    extra_labels: Optional[list[str]] = None,
    with_annotations: bool = True,
) -> dict:
    """予測結果 JSON 群を CVAT の新規タスクとして作成する"""
    items, labels = _collect_prediction_items(json_paths)
    for lb in (extra_labels or []):
        if lb and lb not in labels:
            labels.append(lb)
    return push_items_to_cvat(items, labels, task_name, with_annotations)
