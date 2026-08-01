"""データセットの検査・修正・分割・クラス編集"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

from core import (check_dataset_quality, dataset_class_names, dataset_task_type,
                  fix_dataset_labels, remap_dataset_classes, resolve_train_data_arg,
                  resplit_dataset, _yolo_txt_to_xyxy)


# ── タスク種別の判定 ────────────────────────────────────────────────────
def test_task_type_detect(detect_dataset: Path):
    assert dataset_task_type(str(detect_dataset / "data.yaml")) == "detect"


def test_task_type_classify(classify_dataset: Path):
    assert dataset_task_type(str(classify_dataset / "data.yaml")) == "classify"


def test_task_type_defaults_to_detect(tmp_path: Path):
    """task 未記載の古い data.yaml は detect 扱いにする"""
    y = tmp_path / "data.yaml"
    y.write_text(yaml.dump({"names": ["a"], "nc": 1}))
    assert dataset_task_type(str(y)) == "detect"


def test_classify_passes_directory_not_yaml(classify_dataset: Path):
    """classify だけは train() にディレクトリを渡す必要がある"""
    y = str(classify_dataset / "data.yaml")
    assert resolve_train_data_arg(y) == str(classify_dataset)


def test_detect_passes_yaml(detect_dataset: Path):
    y = str(detect_dataset / "data.yaml")
    assert resolve_train_data_arg(y) == y


# ── 品質チェック ────────────────────────────────────────────────────────
def test_quality_detects_broken_labels(broken_dataset: Path):
    r = check_dataset_quality(broken_dataset)
    kinds = {k: v["count"] for k, v in r["issue_counts"].items()}
    assert kinds.get("サイズ不正") == 1
    assert kinds.get("座標範囲外") == 1
    assert kinds.get("極小ボックス") == 1
    assert kinds.get("ラベル無し画像") == 1      # img3
    assert kinds.get("画像無しラベル") == 1      # orphan
    assert r["n_errors"] > 0


def test_quality_counts_are_capped(tmp_path: Path):
    """同じ指摘が大量に出ても詳細は打ち切り、総数は保つ"""
    import cv2
    import numpy as np
    ds = tmp_path / "many"
    (ds / "labels" / "train").mkdir(parents=True)
    for i in range(50):
        p = ds / "images" / "train" / f"{i}.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p), np.zeros((8, 8, 3), np.uint8))
        (ds / "labels" / "train" / f"{i}.txt").write_text("0 0.5 0.5 0.0 0.1\n")
    (ds / "data.yaml").write_text(yaml.dump({"names": ["x"], "nc": 1}))

    r = check_dataset_quality(ds)
    assert r["issue_counts"]["サイズ不正"]["count"] == 50
    assert len([i for i in r["issues"] if i["kind"] == "サイズ不正"]) == 20


def test_quality_reports_missing_labels_dir_once(tmp_path: Path):
    """labels/ が丸ごと無いときは画像1枚ずつではなく1件にまとめる"""
    import cv2
    import numpy as np
    ds = tmp_path / "nolabels"
    for i in range(30):
        p = ds / "images" / "train" / f"{i}.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p), np.zeros((8, 8, 3), np.uint8))
    (ds / "data.yaml").write_text(yaml.dump({"names": ["x"], "nc": 1}))

    r = check_dataset_quality(ds)
    assert r["issue_counts"]["labelsディレクトリ無し"]["count"] == 1
    assert "ラベル無し画像" not in r["issue_counts"]


def test_quality_rejects_non_yolo_layout(tmp_path: Path):
    ds = tmp_path / "raw"
    ds.mkdir()
    (ds / "annotations.xml").write_text("<annotations/>")
    r = check_dataset_quality(ds)
    assert r["error"] and "images/" in r["error"]


def test_quality_handles_classify(classify_dataset: Path):
    r = check_dataset_quality(classify_dataset)
    assert r["splits"]["train"]["images"] == 11
    assert r["class_counts"] == {"a": 10, "b": 4}


def test_obb_out_of_range_is_warning_not_error(tmp_path: Path):
    """回転BBOX は角が画像外に出るのが正常なのでエラーにしない"""
    import cv2
    import numpy as np
    ds = tmp_path / "obb"
    p = ds / "images" / "train" / "a.png"
    p.parent.mkdir(parents=True)
    cv2.imwrite(str(p), np.zeros((8, 8, 3), np.uint8))
    (ds / "labels" / "train").mkdir(parents=True)
    (ds / "labels" / "train" / "a.txt").write_text(
        "0 -0.01 0.1 0.9 0.1 0.9 0.9 0.1 0.9\n")
    (ds / "data.yaml").write_text(yaml.dump({"names": ["x"], "nc": 1, "task": "obb"}))

    r = check_dataset_quality(ds)
    assert "座標範囲外(OBB)" in r["issue_counts"]
    assert r["issue_counts"]["座標範囲外(OBB)"]["severity"] == "warn"


# ── ラベルの自動修正 ────────────────────────────────────────────────────
def test_fix_removes_broken_lines_and_backs_up(broken_dataset: Path):
    r = fix_dataset_labels(broken_dataset, drop_invalid_size=True,
                           drop_out_of_range=True, drop_tiny=True,
                           delete_orphan_labels=True)
    assert r["error"] is None
    assert r["lines_removed"] == 3
    assert r["orphans_deleted"] == 1

    kept = (broken_dataset / "labels" / "train" / "img0.txt").read_text().strip()
    assert kept == "0 0.5 0.5 0.2 0.2"                       # 正常な行だけ残る
    assert (broken_dataset / "labels" / "train" / "img0.txt.bak").exists()
    assert len((broken_dataset / "labels" / "train" / "img0.txt.bak")
               .read_text().strip().splitlines()) == 4       # 元の4行が残っている

    after = check_dataset_quality(broken_dataset)
    for kind in ("サイズ不正", "座標範囲外", "極小ボックス", "画像無しラベル"):
        assert kind not in after["issue_counts"]


def test_fix_keeps_tiny_boxes_when_not_requested(broken_dataset: Path):
    """小さい物体を意図的に付けている場合を壊さない"""
    r = fix_dataset_labels(broken_dataset, drop_tiny=False)
    assert r["lines_removed"] == 2        # 幅0 と 範囲外 のみ
    body = (broken_dataset / "labels" / "train" / "img0.txt").read_text()
    assert "0.005" in body


# ── train/val の再分割 ──────────────────────────────────────────────────
def test_resplit_keeps_total_and_pairs(detect_dataset: Path):
    r = resplit_dataset(detect_dataset, val_ratio=0.25, seed=42)
    assert r["ok"], r["error"]
    assert sum(r["before"].values()) == sum(r["after"].values()) == 8

    q = check_dataset_quality(detect_dataset)
    assert "ラベル無し画像" not in q["issue_counts"]   # 画像とラベルが対で動く
    assert "画像無しラベル" not in q["issue_counts"]


def test_resplit_is_deterministic(detect_dataset: Path):
    """同じシードなら同じ分け方になる"""
    resplit_dataset(detect_dataset, val_ratio=0.25, seed=7)
    first = sorted(p.name for p in (detect_dataset / "images" / "val").iterdir())
    resplit_dataset(detect_dataset, val_ratio=0.25, seed=7)
    assert sorted(p.name for p in (detect_dataset / "images" / "val").iterdir()) == first


def test_resplit_classify_is_stratified(classify_dataset: Path):
    """少数クラスが片側に寄らないこと"""
    r = resplit_dataset(classify_dataset, val_ratio=0.25, seed=1)
    assert r["ok"] and r["task"] == "classify"
    for split in ("train", "val"):
        for cname in ("a", "b"):
            assert (classify_dataset / split / cname).exists()
            assert len(list((classify_dataset / split / cname).iterdir())) > 0


def test_resplit_preserves_symlinks(tmp_path: Path):
    """データセット生成は symlink を張るので、移動で壊してはいけない"""
    import cv2
    import numpy as np
    src = tmp_path / "src"
    src.mkdir()
    ds = tmp_path / "ds"
    for split in ("train", "val"):
        (ds / "images" / split).mkdir(parents=True)
        (ds / "labels" / split).mkdir(parents=True)
    for i in range(6):
        real = src / f"{i}.png"
        cv2.imwrite(str(real), np.zeros((8, 8, 3), np.uint8))
        split = "train" if i < 5 else "val"
        os.symlink(real, ds / "images" / split / f"{i}.png")
        (ds / "labels" / split / f"{i}.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    (ds / "data.yaml").write_text(yaml.dump({"names": ["x"], "nc": 1, "task": "detect"}))

    r = resplit_dataset(ds, val_ratio=0.34, seed=3)
    assert r["ok"]
    links = list((ds / "images").rglob("*.png"))
    assert links and all(p.is_symlink() for p in links)
    assert all(p.resolve().exists() for p in links)


def test_resplit_rejects_bad_ratio(detect_dataset: Path):
    assert resplit_dataset(detect_dataset, val_ratio=0.99)["error"]


# ── クラス名の編集 ──────────────────────────────────────────────────────
def test_remap_renames_and_merges(detect_dataset: Path):
    r = remap_dataset_classes(detect_dataset,
                              {"red": "pepper", "green": "other", "blue": "other"})
    assert r["ok"], r["error"]
    assert r["new_classes"] == ["pepper", "other"]
    assert dataset_class_names(detect_dataset) == ["pepper", "other"]

    ids = {int(line.split()[0])
           for t in (detect_dataset / "labels").rglob("*.txt")
           for line in t.read_text().splitlines() if line.strip()}
    assert ids <= {0, 1}
    assert list((detect_dataset / "labels").rglob("*.txt.bak"))   # バックアップがある


def test_remap_deletes_class(detect_dataset: Path):
    r = remap_dataset_classes(detect_dataset,
                              {"red": "red", "green": None, "blue": None})
    assert r["ok"]
    assert r["new_classes"] == ["red"]
    assert r["lines_removed"] > 0
    ids = {int(line.split()[0])
           for t in (detect_dataset / "labels").rglob("*.txt")
           for line in t.read_text().splitlines() if line.strip()}
    assert ids == {0}


def test_remap_refuses_to_delete_everything(detect_dataset: Path):
    r = remap_dataset_classes(detect_dataset,
                              {"red": None, "green": None, "blue": None})
    assert not r["ok"] and r["error"]


def test_remap_merges_classify_directories(classify_dataset: Path):
    before = sum(1 for _ in (classify_dataset).rglob("*.png"))
    r = remap_dataset_classes(classify_dataset, {"a": "merged", "b": "merged"})
    assert r["ok"] and r["new_classes"] == ["merged"]
    assert sum(1 for _ in (classify_dataset).rglob("*.png")) == before  # 枚数は保つ
    assert sorted(p.name for p in (classify_dataset / "train").iterdir()) == ["merged"]


# ── YOLO ラベルの読み取り ───────────────────────────────────────────────
def test_yolo_txt_detect_format(tmp_path: Path):
    p = tmp_path / "a.txt"
    p.write_text("0 0.5 0.5 0.4 0.2\n")
    boxes = _yolo_txt_to_xyxy(p, 100, 100, ["x"])
    assert len(boxes) == 1
    assert boxes[0]["label"] == "x"
    assert boxes[0]["bbox_xyxy"] == [30.0, 40.0, 70.0, 60.0]


def test_yolo_txt_segment_format(tmp_path: Path):
    """segment のポリゴンは外接矩形に変換しつつ輪郭も保持する"""
    p = tmp_path / "a.txt"
    p.write_text("0 0.1 0.1 0.5 0.1 0.5 0.9 0.1 0.9\n")
    boxes = _yolo_txt_to_xyxy(p, 100, 100, ["x"])
    assert boxes[0]["bbox_xyxy"] == [10.0, 10.0, 50.0, 90.0]
    assert len(boxes[0]["mask_xy"]) == 4


def test_yolo_txt_missing_file(tmp_path: Path):
    assert _yolo_txt_to_xyxy(tmp_path / "none.txt", 10, 10, ["x"]) == []
