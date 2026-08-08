# =============================================================================
# データセット統合のテスト
#
#   いちばん怖いのはクラス ID の取り違え。
#   names の並びが違うデータセットを混ぜたとき、
#   ラベルが別のクラスに化けないことを重点的に確かめる。
# =============================================================================
from __future__ import annotations

import yaml

import pytest

from core.dataset import merge_datasets
from core.provenance import read_provenance, read_status, read_tags


def _make_dataset(root, name, labels, per_split, task="detect"):
    """detect 形式のデータセットを作る。

    per_split: {"train": [[(cls, "0.5 0.5 0.2 0.2"), ...], ...]}
                画像 1 枚ぶんのラベル行の並びを画像ごとに与える
    """
    d = root / name
    for sp, images in per_split.items():
        (d / "images" / sp).mkdir(parents=True, exist_ok=True)
        (d / "labels" / sp).mkdir(parents=True, exist_ok=True)
        for i, lines in enumerate(images):
            (d / "images" / sp / f"img{i}.jpg").write_bytes(b"x")
            (d / "labels" / sp / f"img{i}.txt").write_text(
                "\n".join(f"{c} {box}" for c, box in lines))
    (d / "data.yaml").write_text(yaml.dump(
        {"path": str(d), "train": "images/train", "val": "images/val",
         "task": task, "nc": len(labels), "names": labels},
        allow_unicode=True))
    return d


def _labels_of(ds, split):
    """{画像stem: [(クラスID, 残り), ...]} を読み出す"""
    out = {}
    for f in sorted((ds / "labels" / split).iterdir()):
        rows = []
        for line in f.read_text().splitlines():
            if line.strip():
                parts = line.split()
                rows.append((int(parts[0]), " ".join(parts[1:])))
        out[f.stem] = rows
    return out


@pytest.fixture
def two_datasets(tmp_path):
    """names の並びが違う 2 つのデータセット"""
    a = _make_dataset(tmp_path, "ds_a", ["cat", "dog"], {
        "train": [[(0, "0.1 0.1 0.2 0.2")], [(1, "0.3 0.3 0.2 0.2")]],
        "val":   [[(0, "0.5 0.5 0.2 0.2")]],
    })
    # B は dog しか持たず、その ID は 0
    b = _make_dataset(tmp_path, "ds_b", ["dog"], {
        "train": [[(0, "0.7 0.7 0.2 0.2")]],
        "val":   [[(0, "0.9 0.9 0.1 0.1")]],
    })
    return a, b


# ---------------------------------------------------------------------------
# 中核: クラス ID の振り直し
# ---------------------------------------------------------------------------
def test_クラスIDが正しく振り直される(tmp_path, two_datasets):
    a, b = two_datasets
    out = tmp_path / "merged"
    res = merge_datasets([a, b], out)
    assert res["ok"], res["error"]

    # 統合後のクラス順は「先に出てきた順」
    assert res["labels"] == ["cat", "dog"]

    rows = _labels_of(out, "train")
    # A 由来はそのまま（cat=0, dog=1）
    assert rows["ds_a__img0"] == [(0, "0.1 0.1 0.2 0.2")]
    assert rows["ds_a__img1"] == [(1, "0.3 0.3 0.2 0.2")]
    # B 由来の「0 = dog」は統合後の dog=1 に振り直される
    assert rows["ds_b__img0"] == [(1, "0.7 0.7 0.2 0.2")], \
        "B のクラスIDが振り直されていない（cat に化けている）"


def test_座標は変えない(tmp_path, two_datasets):
    a, b = two_datasets
    out = tmp_path / "merged"
    merge_datasets([a, b], out)
    for rows in _labels_of(out, "train").values():
        for _, box in rows:
            assert box in ("0.1 0.1 0.2 0.2", "0.3 0.3 0.2 0.2", "0.7 0.7 0.2 0.2")


def test_同じクラス構成なら順序が保たれる(tmp_path):
    a = _make_dataset(tmp_path, "a", ["x", "y"],
                      {"train": [[(0, "0.1 0.1 0.1 0.1")]]})
    b = _make_dataset(tmp_path, "b", ["x", "y"],
                      {"train": [[(1, "0.2 0.2 0.1 0.1")]]})
    out = tmp_path / "m"
    res = merge_datasets([a, b], out)
    assert res["labels"] == ["x", "y"]
    rows = _labels_of(out, "train")
    assert rows["a__img0"] == [(0, "0.1 0.1 0.1 0.1")]
    assert rows["b__img0"] == [(1, "0.2 0.2 0.1 0.1")]


# ---------------------------------------------------------------------------
# 構造
# ---------------------------------------------------------------------------
def test_画像とラベルが対応して置かれる(tmp_path, two_datasets):
    a, b = two_datasets
    out = tmp_path / "merged"
    merge_datasets([a, b], out)

    for sp in ("train", "val"):
        imgs = {p.stem for p in (out / "images" / sp).iterdir()}
        lbls = {p.stem for p in (out / "labels" / sp).iterdir()}
        assert imgs == lbls, f"{sp}: 画像とラベルが対応していない"
    assert (out / "images" / "train").exists()
    assert not (out / "train" / "images").exists(), "旧来の誤った構造になっている"


def test_data_yamlが正しい(tmp_path, two_datasets):
    a, b = two_datasets
    out = tmp_path / "merged"
    merge_datasets([a, b], out)
    cfg = yaml.safe_load((out / "data.yaml").read_text())
    assert cfg["train"] == "images/train"
    assert cfg["val"] == "images/val"
    assert cfg["names"] == ["cat", "dog"]
    assert cfg["nc"] == 2
    assert cfg["task"] == "detect"


def test_ファイル名の衝突を避ける(tmp_path):
    """どちらも img0.jpg を持っていても両方残ること"""
    a = _make_dataset(tmp_path, "a", ["x"], {"train": [[(0, "0.1 0.1 0.1 0.1")]]})
    b = _make_dataset(tmp_path, "b", ["x"], {"train": [[(0, "0.2 0.2 0.1 0.1")]]})
    out = tmp_path / "m"
    merge_datasets([a, b], out)
    assert len(list((out / "images" / "train").iterdir())) == 2


def test_元のデータセットには触らない(tmp_path, two_datasets):
    a, b = two_datasets
    before_a = _labels_of(a, "train")
    before_b = _labels_of(b, "train")
    merge_datasets([a, b], tmp_path / "m")
    assert _labels_of(a, "train") == before_a
    assert _labels_of(b, "train") == before_b


# ---------------------------------------------------------------------------
# 来歴
# ---------------------------------------------------------------------------
def test_統合元が来歴に残る(tmp_path, two_datasets):
    a, b = two_datasets
    out = tmp_path / "merged"
    merge_datasets([a, b], out)

    prov = read_provenance(out)
    assert prov["source"] == "merge"
    names = [s["name"] for s in prov["sources"]]
    assert names == ["ds_a", "ds_b"]
    # その時点の枚数が焼き込まれている
    assert prov["sources"][0]["counts"] == {"train": 2, "val": 1}
    assert prov["sources"][1]["counts"] == {"train": 1, "val": 1}


def test_統合時に状態とタグを付けられる(tmp_path, two_datasets):
    a, b = two_datasets
    out = tmp_path / "merged"
    merge_datasets([a, b], out, status="reviewed", tags="統合済み, 実験用")
    assert read_status(out) == "reviewed"
    assert read_tags(out) == ["統合済み", "実験用"]


# ---------------------------------------------------------------------------
# 拒否する条件
# ---------------------------------------------------------------------------
def test_1つでは統合できない(tmp_path, two_datasets):
    a, _ = two_datasets
    res = merge_datasets([a], tmp_path / "m")
    assert not res["ok"] and "2 つ以上" in res["error"]


def test_存在しないデータセットを拒否する(tmp_path, two_datasets):
    a, _ = two_datasets
    res = merge_datasets([a, tmp_path / "ない"], tmp_path / "m")
    assert not res["ok"] and "ありません" in res["error"]


def test_出力先が既にあれば中断する(tmp_path, two_datasets):
    a, b = two_datasets
    out = tmp_path / "m"
    out.mkdir()
    (out / "既存.txt").write_text("消されたら困る")
    res = merge_datasets([a, b], out)
    assert not res["ok"] and "既にあります" in res["error"]
    assert (out / "既存.txt").exists(), "既存の中身が消えている"


def test_タスク種別が違えば拒否する(tmp_path):
    a = _make_dataset(tmp_path, "a", ["x"],
                      {"train": [[(0, "0.1 0.1 0.1 0.1")]]}, task="detect")
    b = _make_dataset(tmp_path, "b", ["x"],
                      {"train": [[(0, "0.1 0.1 0.1 0.1")]]}, task="segment")
    res = merge_datasets([a, b], tmp_path / "m")
    assert not res["ok"] and "タスク種別" in res["error"]


def test_壊れたラベル行は除外して警告する(tmp_path):
    a = _make_dataset(tmp_path, "a", ["x"], {"train": [[(0, "0.1 0.1 0.1 0.1")]]})
    b = _make_dataset(tmp_path, "b", ["x"], {"train": [[(0, "0.2 0.2 0.1 0.1")]]})
    (b / "labels" / "train" / "img0.txt").write_text(
        "0 0.2 0.2 0.1 0.1\nこれは壊れた行\n9 0.3 0.3 0.1 0.1")
    res = merge_datasets([a, b], tmp_path / "m")
    assert res["ok"]
    assert res["n_skipped"] == 2      # 読めない行と、存在しないクラスID
    assert any("除外" in w for w in res["warnings"])


def test_ラベルが無い画像は背景画像として空ラベルになる(tmp_path):
    a = _make_dataset(tmp_path, "a", ["x"], {"train": [[(0, "0.1 0.1 0.1 0.1")]]})
    b = _make_dataset(tmp_path, "b", ["x"], {"train": [[(0, "0.2 0.2 0.1 0.1")]]})
    (b / "labels" / "train" / "img0.txt").unlink()
    out = tmp_path / "m"
    res = merge_datasets([a, b], out)
    assert res["ok"]
    assert (out / "labels" / "train" / "b__img0.txt").read_text() == ""
    assert (out / "images" / "train" / "b__img0.jpg").exists()
