# =============================================================================
# 重複画像の検出のテスト
#
#   同じ画像が train と val の両方に入ると、評価が実力より高く出る。
#   「val で 0.95 なのに実機で使えない」の典型的な原因なので、
#   またぎを確実に見つけられることを重点的に確かめる。
# =============================================================================
from __future__ import annotations

from pathlib import Path

import pytest

from core.dataset import (
    _image_hash, _split_of, duplicate_warning, find_duplicate_images,
    hash_dataset_images,
)


def _ds(root, name, layout):
    """layout = {"train": [b"a", b"b"], "val": [b"a"]} のように中身で指定する"""
    d = root / name
    for sp, blobs in layout.items():
        (d / "images" / sp).mkdir(parents=True, exist_ok=True)
        for i, blob in enumerate(blobs):
            (d / "images" / sp / f"{sp}{i}.jpg").write_bytes(blob)
    return d


# ---------------------------------------------------------------------------
# 下ごしらえ
# ---------------------------------------------------------------------------
def test_中身が同じならハッシュも同じ(tmp_path):
    a, b = tmp_path / "a.jpg", tmp_path / "b.jpg"
    a.write_bytes(b"same")
    b.write_bytes(b"same")
    assert _image_hash(a) == _image_hash(b), "名前が違っても中身が同じなら重複"


def test_中身が違えばハッシュも違う(tmp_path):
    a, b = tmp_path / "a.jpg", tmp_path / "b.jpg"
    a.write_bytes(b"one")
    b.write_bytes(b"two")
    assert _image_hash(a) != _image_hash(b)


def test_読めなければNone(tmp_path):
    assert _image_hash(tmp_path / "ない.jpg") is None


def test_スプリットを判別する(tmp_path):
    d = _ds(tmp_path, "ds", {"train": [b"a"], "val": [b"b"]})
    assert _split_of(d / "images" / "train" / "train0.jpg", d) == "train"
    assert _split_of(d / "images" / "val" / "val0.jpg", d) == "val"
    assert _split_of(tmp_path / "よそ.jpg", d) == ""


def test_validもvalとして扱う(tmp_path):
    d = tmp_path / "ds"
    (d / "images" / "valid").mkdir(parents=True)
    (d / "images" / "valid" / "a.jpg").write_bytes(b"x")
    assert _split_of(d / "images" / "valid" / "a.jpg", d) == "val"


def test_退避ディレクトリは数えない(tmp_path):
    """モザイクの _backup_original を重複として拾わないこと"""
    d = _ds(tmp_path, "ds", {"train": [b"a"]})
    bak = d / "_backup_original" / "images" / "train"
    bak.mkdir(parents=True)
    (bak / "train0.jpg").write_bytes(b"a")
    idx = hash_dataset_images(d)
    assert sum(len(v) for v in idx.values()) == 1


# ---------------------------------------------------------------------------
# 重複の検出
# ---------------------------------------------------------------------------
def test_重複が無ければ空(tmp_path):
    d = _ds(tmp_path, "ds", {"train": [b"a", b"b"], "val": [b"c"]})
    r = find_duplicate_images([d])
    assert r["groups"] == [] and r["n_duplicates"] == 0
    assert r["n_images"] == 3 and r["n_unique"] == 3


def test_trainとvalにまたがる重複を見つける(tmp_path):
    d = _ds(tmp_path, "ds", {"train": [b"a", b"b"], "val": [b"a"]})
    r = find_duplicate_images([d])
    assert len(r["groups"]) == 1
    assert len(r["cross_split"]) == 1, "またぎを見逃している"
    assert r["cross_split"][0]["splits"] == ["train", "val"]
    assert r["n_duplicates"] == 1


def test_同じスプリット内の重複はまたぎにしない(tmp_path):
    d = _ds(tmp_path, "ds", {"train": [b"a", b"a"], "val": [b"c"]})
    r = find_duplicate_images([d])
    assert len(r["groups"]) == 1
    assert r["cross_split"] == [], "同じ split 内なので評価は汚れない"


def test_データセット間の重複を見つける(tmp_path):
    a = _ds(tmp_path, "ds_a", {"train": [b"x"]})
    b = _ds(tmp_path, "ds_b", {"train": [b"x"]})
    r = find_duplicate_images([a, b])
    assert len(r["cross_dataset"]) == 1
    assert r["cross_dataset"][0]["datasets"] == ["ds_a", "ds_b"]


def test_統合したときのまたぎを予見できる(tmp_path):
    """A の train と B の val に同じ画像 → 混ぜると評価が汚れる"""
    a = _ds(tmp_path, "ds_a", {"train": [b"x"]})
    b = _ds(tmp_path, "ds_b", {"val": [b"x"]})
    r = find_duplicate_images([a, b])
    assert len(r["cross_split"]) == 1
    assert len(r["cross_dataset"]) == 1


def test_3枚以上の重複もまとめる(tmp_path):
    d = _ds(tmp_path, "ds", {"train": [b"a", b"a", b"a"]})
    r = find_duplicate_images([d])
    assert r["groups"][0]["count"] == 3
    assert r["n_duplicates"] == 2


def test_件数の多い順に並ぶ(tmp_path):
    d = _ds(tmp_path, "ds", {"train": [b"a", b"a", b"a", b"b", b"b"]})
    r = find_duplicate_images([d])
    assert [g["count"] for g in r["groups"]] == [3, 2]


def test_画像の無いデータセットでも落ちない(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    r = find_duplicate_images([d])
    assert r["n_images"] == 0 and r["groups"] == []


def test_進捗が通知される(tmp_path):
    d = _ds(tmp_path, "ds", {"train": [b"a", b"b", b"c"]})
    seen = []
    find_duplicate_images([d], on_progress=lambda x, y: seen.append((x, y)))
    assert seen == [(1, 3), (2, 3), (3, 3)]


# ---------------------------------------------------------------------------
# 警告文
# ---------------------------------------------------------------------------
def test_重複が無ければ警告しない(tmp_path):
    d = _ds(tmp_path, "ds", {"train": [b"a"], "val": [b"b"]})
    assert duplicate_warning(find_duplicate_images([d])) == ""


def test_またぎがあれば強く伝える(tmp_path):
    d = _ds(tmp_path, "ds", {"train": [b"a"], "val": [b"a"]})
    msg = duplicate_warning(find_duplicate_images([d]))
    assert "train と val" in msg
    assert "実力より高く" in msg, "何が問題なのかを書くこと"


def test_データセット間の重複も伝える(tmp_path):
    a = _ds(tmp_path, "ds_a", {"train": [b"x"]})
    b = _ds(tmp_path, "ds_b", {"train": [b"x"]})
    msg = duplicate_warning(find_duplicate_images([a, b]))
    assert "別のデータセット" in msg
