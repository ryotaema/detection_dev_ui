# =============================================================================
# 片付けのテスト
#
#   まとめて削除する操作なので、範囲の取り違えが起きないことを重点的に見る。
#   「使えるデータセットを未完成と判定して消す」のが最悪の事故。
# =============================================================================
from __future__ import annotations

from pathlib import Path

import pytest

from core.cleanup import (
    TEMP_DIRS, cleanup_summary, delete_paths, find_incomplete_datasets,
    find_runs_without_weights, find_temp_files,
)


def _ds(root, name, *, yaml=False, images=0, raw=False):
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    if yaml:
        (d / "data.yaml").write_text("names: [a]\n")
    if images:
        (d / "images" / "train").mkdir(parents=True, exist_ok=True)
        for i in range(images):
            (d / "images" / "train" / f"{i}.jpg").write_bytes(b"x" * 10)
    if raw:
        (d / "raw").mkdir(exist_ok=True)
        (d / "raw" / "a.xml").write_text("<x/>")
    return d


def _run(root, name, *, weights=False, files=1):
    d = root / name
    (d / "weights").mkdir(parents=True, exist_ok=True)
    if weights:
        (d / "weights" / "best.pt").write_bytes(b"x" * 100)
    for i in range(files):
        (d / f"plot{i}.png").write_bytes(b"x" * 50)
    return d


# ---------------------------------------------------------------------------
# 未完成のデータセット
# ---------------------------------------------------------------------------
def test_datayamlがあれば未完成にしない(tmp_path):
    """使えるデータセットを消させないことが最重要"""
    _ds(tmp_path, "ok", yaml=True, images=3)
    assert find_incomplete_datasets(tmp_path) == []


def test_画像があってもdatayamlが無ければ挙げる(tmp_path):
    """CVAT からエクスポートしただけの状態は学習に使えない"""
    _ds(tmp_path, "raw_only", images=5, raw=True)
    got = find_incomplete_datasets(tmp_path)
    assert len(got) == 1
    assert got[0]["images"] == 5
    assert got[0]["has_raw"] is True
    assert "変換" in got[0]["reason"]


def test_空のデータセットは削除してよいと言う(tmp_path):
    _ds(tmp_path, "empty")
    got = find_incomplete_datasets(tmp_path)
    assert got[0]["images"] == 0
    assert "削除して構いません" in got[0]["hint"]


def test_画像はあるがrawもyamlも無い場合(tmp_path):
    _ds(tmp_path, "odd", images=2)
    got = find_incomplete_datasets(tmp_path)
    assert "data.yaml がありません" in got[0]["reason"]


def test_使用実績も一緒に返す(tmp_path):
    """消す前に「学習に使われていないか」を確かめられること"""
    _ds(tmp_path, "empty")
    got = find_incomplete_datasets(tmp_path)
    assert "usage" in got[0] and "safe_to_delete" in got[0]["usage"]


def test_退避ディレクトリの画像は数えない(tmp_path):
    d = _ds(tmp_path, "x")
    bak = d / "_backup_original" / "images"
    bak.mkdir(parents=True)
    (bak / "a.jpg").write_bytes(b"x")
    assert find_incomplete_datasets(tmp_path)[0]["images"] == 0


def test_データがなければ空(tmp_path):
    assert find_incomplete_datasets(tmp_path / "ない") == []


# ---------------------------------------------------------------------------
# 重みの無い run
# ---------------------------------------------------------------------------
def test_重みがあればあげない(tmp_path):
    _run(tmp_path, "ok", weights=True)
    assert find_runs_without_weights(tmp_path) == []


def test_重みが無いrunをあげる(tmp_path):
    _run(tmp_path, "failed", weights=False, files=3)
    got = find_runs_without_weights(tmp_path)
    assert len(got) == 1 and got[0]["name"] == "failed"
    assert got[0]["files"] >= 3


def test_何エポックまで進んだか分かる(tmp_path):
    d = _run(tmp_path, "partial")
    (d / "results.csv").write_text("epoch,loss\n1,0.5\n2,0.4\n3,0.3\n")
    assert find_runs_without_weights(tmp_path)[0]["epochs"] == 3


def test_resultsが無くても落ちない(tmp_path):
    _run(tmp_path, "bare")
    assert find_runs_without_weights(tmp_path)[0]["epochs"] is None


# ---------------------------------------------------------------------------
# 一時ファイル
# ---------------------------------------------------------------------------
def test_一時ディレクトリを見つける(tmp_path):
    for name in ("_tmp_uploads", "exports"):
        d = tmp_path / name
        d.mkdir()
        (d / "a.png").write_bytes(b"x" * 1000)
    got = find_temp_files(tmp_path)
    assert {g["name"] for g in got} == {"_tmp_uploads", "exports"}
    assert all(g["desc"] for g in got), "何のためのものか説明を付けること"


def test_大きい順に並ぶ(tmp_path):
    (tmp_path / "exports").mkdir()
    (tmp_path / "exports" / "a").write_bytes(b"x" * 5000)
    (tmp_path / "_tmp_uploads").mkdir()
    (tmp_path / "_tmp_uploads" / "a").write_bytes(b"x" * 100)
    assert [g["name"] for g in find_temp_files(tmp_path)] == \
        ["exports", "_tmp_uploads"]


def test_空の一時ディレクトリは挙げない(tmp_path):
    (tmp_path / "exports").mkdir()
    assert find_temp_files(tmp_path) == []


def test_知らないディレクトリは触らない(tmp_path):
    """predictions 直下の推論結果そのものを一時ファイル扱いしないこと"""
    d = tmp_path / "大事なもの"
    d.mkdir()
    (d / "a.json").write_text("{}")
    assert find_temp_files(tmp_path) == []


# ---------------------------------------------------------------------------
# 削除（範囲の取り違えを防ぐ）
# ---------------------------------------------------------------------------
def test_まとめて削除できる(tmp_path):
    a = _ds(tmp_path, "a")
    b = _ds(tmp_path, "b")
    res = delete_paths([a, b], guard_root=tmp_path)
    assert res["ok"] and len(res["deleted"]) == 2
    assert not a.exists() and not b.exists()


def test_範囲の外は消さない(tmp_path):
    """guard_root の外にあるものを消さないこと"""
    inside = _ds(tmp_path / "data", "a")
    outside = tmp_path / "よそ"
    outside.mkdir()
    (outside / "大事.txt").write_text("消されたら困る")

    res = delete_paths([inside, outside], guard_root=tmp_path / "data")
    assert not inside.exists()
    assert outside.exists(), "範囲外を消してしまった"
    assert any("外にあります" in why for _, why in res["skipped"])


def test_既に無いものは飛ばす(tmp_path):
    res = delete_paths([tmp_path / "ない"], guard_root=tmp_path)
    assert res["ok"] and res["deleted"] == []
    assert any("既にありません" in why for _, why in res["skipped"])


def test_ファイルも消せる(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("x")
    assert delete_paths([f], guard_root=tmp_path)["ok"]
    assert not f.exists()


def test_範囲指定なしでも動く(tmp_path):
    d = _ds(tmp_path, "a")
    assert delete_paths([d])["ok"] and not d.exists()


# ---------------------------------------------------------------------------
# まとめ
# ---------------------------------------------------------------------------
def test_まとめの形(monkeypatch, tmp_path):
    import core.cleanup as cu
    (tmp_path / "data").mkdir()
    (tmp_path / "models").mkdir()
    (tmp_path / "predictions").mkdir()
    monkeypatch.setattr(cu, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(cu, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(cu, "PREDICTIONS_DIR", tmp_path / "predictions")

    _ds(tmp_path / "data", "empty")
    _run(tmp_path / "models", "failed")
    d = tmp_path / "predictions" / "exports"
    d.mkdir()
    (d / "a.png").write_bytes(b"x" * 2000)

    s = cleanup_summary()
    assert len(s["incomplete_datasets"]) == 1
    assert len(s["runs_without_weights"]) == 1
    assert len(s["temp_files"]) == 1
    assert s["n_items"] == 3 and s["total_size"] > 0
