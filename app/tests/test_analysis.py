"""推論結果の分析・CVAT XML・来歴・エラー解釈"""
from __future__ import annotations

import json
from pathlib import Path

from conftest import box

from core import (analyze_predictions, build_cvat_xml, count_dataset_items,
                  explain_error, prediction_display_name, prediction_json_path,
                  read_provenance, record_dataset_provenance,
                  record_model_provenance, slugify_function_name,
                  _box_iou, _collect_prediction_items, _iou)


# ── IoU ─────────────────────────────────────────────────────────────────
def test_iou_identical_boxes():
    assert _iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0


def test_iou_no_overlap():
    assert _iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_iou_half_overlap():
    assert abs(_iou([0, 0, 10, 10], [5, 0, 15, 10]) - 1 / 3) < 1e-6


def test_box_iou_matches_iou():
    a, b = [0, 0, 10, 10], [5, 5, 15, 15]
    assert abs(_box_iou(a, b) - _iou(a, b)) < 1e-9


# ── 要確認画像の抽出 ────────────────────────────────────────────────────
def test_flags_zero_detection(prediction_json):
    rows = analyze_predictions([prediction_json("zero", [])])
    assert rows[0]["reasons"] == ["検出ゼロ"]


def test_flags_low_confidence(prediction_json):
    p = prediction_json("low", [box("a", 0.31, [10, 10, 100, 100])])
    rows = analyze_predictions([p], conf_low=0.5)
    assert rows[0]["reasons"][0].startswith("低信頼度")


def test_flags_class_conflict(prediction_json):
    """ほぼ同じ位置に別クラスが重なる = モデルが迷っている"""
    p = prediction_json("conf", [
        box("a", 0.9, [10, 10, 100, 100]),
        box("b", 0.85, [12, 12, 102, 102]),
    ])
    assert "クラス競合" in analyze_predictions([p])[0]["reasons"]


def test_no_conflict_for_same_class(prediction_json):
    p = prediction_json("same", [
        box("a", 0.9, [10, 10, 100, 100]),
        box("a", 0.85, [12, 12, 102, 102]),
    ])
    assert "クラス競合" not in analyze_predictions([p])[0]["reasons"]


def test_flags_tiny_box(prediction_json):
    p = prediction_json("tiny", [
        box("a", 0.95, [10, 10, 14, 14], xywhn=[0.01, 0.01, 0.002, 0.0004])])
    assert "極小ボックス" in analyze_predictions([p], tiny_area=0.001)[0]["reasons"]


def test_clean_prediction_is_not_flagged(prediction_json):
    p = prediction_json("ok", [
        box("a", 0.93, [10, 10, 200, 200], xywhn=[0.15, 0.15, 0.2, 0.2])])
    row = analyze_predictions([p], conf_low=0.5)[0]
    assert row["reasons"] == [] and row["flagged"] is False


def test_display_name_uses_original_image(prediction_json):
    p = prediction_json("x", [], image_name="orig.png")
    assert analyze_predictions([p])[0]["display_name"] == "x_orig.png"


# ── 推論結果のファイル名 ────────────────────────────────────────────────
def test_same_name_different_directory_does_not_collide(tmp_path: Path):
    a = prediction_json_path(tmp_path, str(tmp_path / "dsA" / "1000color.png"))
    b = prediction_json_path(tmp_path, str(tmp_path / "dsB" / "1000color.png"))
    assert a != b
    assert a.name.startswith("1000color__") and b.name.startswith("1000color__")


def test_same_image_maps_to_same_name(tmp_path: Path):
    img = str(tmp_path / "ds" / "a.png")
    assert prediction_json_path(tmp_path, img) == prediction_json_path(tmp_path, img)


def test_display_name_falls_back_to_filename(tmp_path: Path):
    p = tmp_path / "broken.json"
    p.write_text("{ not json")
    assert prediction_display_name(p) == "broken.json"


# ── CVAT XML の組み立て ─────────────────────────────────────────────────
def _items(width=100, height=80, boxes=None):
    return [{"path": Path("a.png"), "width": width, "height": height,
             "boxes": boxes or []}]


def test_xml_writes_box_for_detection():
    import xml.etree.ElementTree as ET
    xml = build_cvat_xml(_items(boxes=[box("a", 0.9, [1, 2, 3, 4])]), ["a"], "t")
    root = ET.fromstring(xml)
    assert len(root.findall(".//box")) == 1
    assert len(root.findall(".//polygon")) == 0


def test_xml_writes_polygon_when_mask_present():
    """セグメンテーション結果はポリゴンとして戻す"""
    import xml.etree.ElementTree as ET
    b = box("a", 0.9, [1, 2, 3, 4], mask=[[1, 2], [3, 2], [3, 4], [1, 4]])
    root = ET.fromstring(build_cvat_xml(_items(boxes=[b]), ["a"], "t"))
    polys = root.findall(".//polygon")
    assert len(polys) == 1 and len(root.findall(".//box")) == 0
    assert len(polys[0].get("points").split(";")) == 4


def test_xml_includes_confidence_attribute():
    import xml.etree.ElementTree as ET
    root = ET.fromstring(
        build_cvat_xml(_items(boxes=[box("a", 0.8765, [1, 2, 3, 4])]), ["a"], "t"))
    assert root.find(".//box/attribute").text == "0.8765"


def test_xml_declares_labels():
    import xml.etree.ElementTree as ET
    root = ET.fromstring(build_cvat_xml(_items(), ["a", "b"], "t"))
    assert [e.text for e in root.findall(".//labels/label/name")] == ["a", "b"]


def test_collect_skips_missing_images(tmp_path: Path):
    p = tmp_path / "m.json"
    p.write_text(json.dumps({"image_path": "/nonexistent/x.jpg", "boxes": []}))
    items, labels = _collect_prediction_items([p])
    assert items == [] and labels == []


def test_collect_uses_recorded_size_without_reading_image(tmp_path: Path):
    """image_size があれば画像を読み直さない（大量件数で効く）"""
    img = tmp_path / "a.png"
    img.write_bytes(b"not a real image")     # 読めないファイル
    p = tmp_path / "a.json"
    p.write_text(json.dumps({"image_path": str(img), "image_size": [640, 480],
                             "boxes": []}))
    items, _ = _collect_prediction_items([p])
    assert len(items) == 1
    assert (items[0]["width"], items[0]["height"]) == (640, 480)


# ── 来歴 ────────────────────────────────────────────────────────────────
def test_dataset_provenance_roundtrip(detect_dataset: Path):
    record_dataset_provenance(detect_dataset, source="cvat", task_type="detect",
                              labels=["red"], cvat_tasks=[{"id": 11, "name": "t"}])
    prov = read_provenance(detect_dataset)
    assert prov["source"] == "cvat"
    assert prov["cvat_tasks"][0]["id"] == 11
    assert prov["counts"] == {"train": 6, "val": 2}


def test_model_provenance_copies_dataset_history(detect_dataset: Path, tmp_path: Path):
    """データセットが消えても学習時点の情報が残るようコピーして持つ"""
    record_dataset_provenance(detect_dataset, source="cvat",
                              cvat_tasks=[{"id": 7, "name": "src"}])
    run = tmp_path / "run"
    run.mkdir()
    record_model_provenance(run, str(detect_dataset / "data.yaml"),
                            "yolo11s.pt", {"epochs": 10})
    prov = read_provenance(run)
    assert prov["dataset"]["counts_at_train"] == {"train": 6, "val": 2}
    assert prov["dataset"]["classes"] == ["red", "green", "blue"]
    assert prov["dataset"]["provenance"]["cvat_tasks"][0]["id"] == 7


def test_counts_detect_dataset(detect_dataset: Path):
    assert count_dataset_items(detect_dataset) == {"train": 6, "val": 2}


def test_counts_classify_dataset(classify_dataset: Path):
    assert count_dataset_items(classify_dataset) == {"train": 11, "val": 3}


def test_read_provenance_missing(tmp_path: Path):
    assert read_provenance(tmp_path) is None


# ── エラーの解釈 ────────────────────────────────────────────────────────
def test_explains_gpu_out_of_memory():
    r = explain_error("torch.cuda.OutOfMemoryError: CUDA out of memory.")
    assert r and "batch" in r["hint"]


def test_explains_resume_finished():
    r = explain_error("last.pt training to 30 epochs is finished, nothing to resume.")
    assert r and "再開できません" in r["title"]


def test_explains_model_version_mismatch():
    r = explain_error("AttributeError: Can't get attribute 'C2PSA' on module "
                      "ultralytics.nn.modules")
    assert r and "読み込めません" in r["title"]


def test_explains_connection_error():
    r = explain_error("ConnectionError: Max retries exceeded with url")
    assert r and "接続" in r["title"]


def test_unknown_error_returns_none():
    """当てはまらないときに誤った対処を出さない"""
    assert explain_error("ValueError: なにか未知のエラー") is None
    assert explain_error("") is None


# ── スラッグ化 ──────────────────────────────────────────────────────────
def test_slugify_replaces_symbols():
    assert slugify_function_name("yolo11s_ep100_2120") == "yolo11s-ep100-2120"
    assert slugify_function_name("My Model v2!!") == "my-model-v2"


def test_slugify_falls_back_when_empty():
    assert slugify_function_name("日本語") == "model"
