"""データ拡張のプレビュー生成"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core import build_augment_preview, describe_augment, list_sample_images

ALL_OFF = {"hsv_h": 0, "hsv_s": 0, "hsv_v": 0, "degrees": 0, "translate": 0,
           "scale": 0, "shear": 0, "fliplr": 0, "flipud": 0, "mosaic": 0,
           "erasing": 0}


def test_finds_sample_images(detect_dataset: Path):
    assert len(list_sample_images(detect_dataset)) > 0


def test_sample_images_respects_limit(detect_dataset: Path):
    assert len(list_sample_images(detect_dataset, limit=3)) == 3


def test_no_images_returns_empty(tmp_path: Path):
    assert list_sample_images(tmp_path) == []


def test_describe_lists_only_enabled():
    assert describe_augment(ALL_OFF) == []
    active = describe_augment({**ALL_OFF, "fliplr": 0.5, "mosaic": 1.0})
    assert len(active) == 2


def test_preview_changes_image(detect_dataset: Path):
    imgs = list_sample_images(detect_dataset)
    orig, variants = build_augment_preview(
        imgs, {**ALL_OFF, "degrees": 30, "hsv_v": 0.5}, seed=0, n_variants=2)
    assert orig is not None and len(variants) == 2
    assert all(not np.array_equal(orig, v) for _, v in variants)


def test_variants_differ_from_each_other(detect_dataset: Path):
    """シードをずらしているので、パターンごとに違うかかり方になる"""
    imgs = list_sample_images(detect_dataset)
    _, variants = build_augment_preview(
        imgs, {**ALL_OFF, "degrees": 30}, seed=0, n_variants=3)
    assert not np.array_equal(variants[0][1], variants[1][1])


def test_no_augmentation_returns_original(detect_dataset: Path):
    imgs = list_sample_images(detect_dataset)
    orig, variants = build_augment_preview(imgs, ALL_OFF, seed=0, n_variants=1)
    assert np.array_equal(orig, variants[0][1])


def test_same_seed_is_reproducible(detect_dataset: Path):
    imgs = list_sample_images(detect_dataset)
    p = {**ALL_OFF, "degrees": 20, "translate": 0.2}
    _, a = build_augment_preview(imgs, p, seed=5, n_variants=1)
    _, b = build_augment_preview(imgs, p, seed=5, n_variants=1)
    assert np.array_equal(a[0][1], b[0][1])


def test_missing_images_fail_safely():
    assert build_augment_preview([], {**ALL_OFF, "degrees": 10}) == (None, [])
    assert build_augment_preview([Path("/nonexistent/x.png")], ALL_OFF) == (None, [])


@pytest.mark.parametrize("key,value", [
    ("hsv_h", 0.9), ("hsv_s", 0.9), ("hsv_v", 0.9), ("degrees", 45),
    ("translate", 0.5), ("scale", 0.9), ("shear", 10),
    ("fliplr", 1.0), ("flipud", 1.0), ("mosaic", 1.0), ("erasing", 1.0),
])
def test_each_augmentation_runs(detect_dataset: Path, key: str, value: float):
    """どのパラメータ単体でも例外なく生成でき、サイズが変わらないこと"""
    imgs = list_sample_images(detect_dataset)
    orig, variants = build_augment_preview(
        imgs, {**ALL_OFF, key: value}, seed=1, n_variants=1)
    assert orig is not None and len(variants) == 1
    assert variants[0][1].shape == orig.shape
