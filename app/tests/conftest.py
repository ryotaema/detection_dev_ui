"""テスト共通の準備。

core/ のロジックだけを対象にする。Streamlit の画面や GPU、
外部サービス（CVAT / MLflow）に依存するものはここでは扱わない。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

# app/ を import パスに入れる（`from core import ...` を使えるようにする）
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _write_image(path: Path, value: int = 0, size: int = 32) -> None:
    import cv2
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((size, size, 3), value, np.uint8))


@pytest.fixture
def detect_dataset(tmp_path: Path) -> Path:
    """detect 形式の小さなデータセット（train 6 / val 2、3クラス）"""
    ds = tmp_path / "detect_ds"
    for split, n in (("train", 6), ("val", 2)):
        for i in range(n):
            _write_image(ds / "images" / split / f"{split}{i}.png", value=i * 10)
            (ds / "labels" / split).mkdir(parents=True, exist_ok=True)
            (ds / "labels" / split / f"{split}{i}.txt").write_text(
                f"{i % 3} 0.5 0.5 0.2 0.2\n")
    (ds / "data.yaml").write_text(yaml.dump({
        "path": str(ds), "train": "images/train", "val": "images/val",
        "task": "detect", "nc": 3, "names": ["red", "green", "blue"],
    }))
    return ds


@pytest.fixture
def classify_dataset(tmp_path: Path) -> Path:
    """classify 形式のデータセット（クラスごとに枚数が偏っている）"""
    ds = tmp_path / "classify_ds"
    for split, counts in (("train", {"a": 8, "b": 3}), ("val", {"a": 2, "b": 1})):
        for cname, n in counts.items():
            for i in range(n):
                _write_image(ds / split / cname / f"{i}.png")
    (ds / "data.yaml").write_text(yaml.dump({
        "path": str(ds), "train": "train", "val": "val",
        "task": "classify", "nc": 2, "names": ["a", "b"],
    }))
    return ds


@pytest.fixture
def broken_dataset(tmp_path: Path) -> Path:
    """壊れたラベルを含むデータセット（品質チェック・自動修正の対象）"""
    ds = tmp_path / "broken_ds"
    for i in range(4):
        _write_image(ds / "images" / "train" / f"img{i}.png")
    lbl = ds / "labels" / "train"
    lbl.mkdir(parents=True, exist_ok=True)
    (lbl / "img0.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n"      # 正常
        "0 0.5 0.5 0.0 0.1\n"      # 幅0
        "0 1.5 0.5 0.2 0.2\n"      # 範囲外
        "0 0.5 0.5 0.005 0.005\n"  # 極小 (面積 0.000025)
    )
    (lbl / "img1.txt").write_text("0 0.4 0.4 0.3 0.3\n")
    (lbl / "img2.txt").write_text("")               # 空ラベル
    # img3 はラベルなし / orphan は画像なし
    (lbl / "orphan.txt").write_text("0 0.4 0.4 0.3 0.3\n")
    (ds / "data.yaml").write_text(yaml.dump({
        "path": str(ds), "train": "images/train", "val": "images/val",
        "task": "detect", "nc": 1, "names": ["x"],
    }))
    return ds


@pytest.fixture
def prediction_json(tmp_path: Path):
    """予測 JSON を作るヘルパーを返す"""
    import json

    img_dir = tmp_path / "src_images"

    def _make(name: str, boxes: list[dict], image_name: str = "img.png") -> Path:
        img = img_dir / f"{name}_{image_name}"
        _write_image(img)
        p = tmp_path / f"{name}.json"
        p.write_text(json.dumps({"image_path": str(img), "boxes": boxes}))
        return p

    return _make


def box(label: str, conf: float, xyxy: list[float],
        xywhn: list[float] | None = None, mask: list | None = None) -> dict:
    """テスト用の検出ボックスを作る"""
    d = {"label": label, "confidence": conf, "bbox_xyxy": xyxy,
         "bbox_xywhn": xywhn or [0.1, 0.1, 0.1, 0.1]}
    if mask:
        d["mask_xy"] = mask
    return d
