# =============================================================================
# 精査の記録のテスト
#
#   これまでフラグは session_state にしか無く、再読み込みで消えていた。
#   5000 枚を見る作業では致命的なので、ファイルに残ることを確かめる。
# =============================================================================
from __future__ import annotations

import json
from pathlib import Path

import pytest

from core import review as rv


@pytest.fixture
def ds(tmp_path, monkeypatch):
    monkeypatch.setattr(rv, "DATA_DIR", tmp_path)
    d = tmp_path / "ds1"
    (d / "images" / "train").mkdir(parents=True)
    for i in range(10):
        (d / "images" / "train" / f"img{i}.jpg").write_bytes(b"x")
    return d


# ---------------------------------------------------------------------------
# 記録の読み書き
# ---------------------------------------------------------------------------
def test_記録が無ければ空(ds):
    st = rv.load_review(ds)
    assert st == {"reviewed": {}, "flagged": {}}


def test_壊れた記録でも落ちない(ds):
    rv.review_path(ds).write_text("{ 壊れている", encoding="utf-8")
    assert rv.load_review(ds) == {"reviewed": {}, "flagged": {}}


def test_想定外の形でも落ちない(ds):
    rv.review_path(ds).write_text("[1,2]", encoding="utf-8")
    assert rv.load_review(ds) == {"reviewed": {}, "flagged": {}}


def test_データセットの中に残る(ds):
    """再読み込みしても消えないこと（session_state ではなくファイル）"""
    rv.mark(ds, "img0.jpg", flagged=True)
    assert rv.review_path(ds).exists()
    assert rv.review_path(ds).parent == ds

    # 別のプロセスから読み直しても残っている
    data = json.loads(rv.review_path(ds).read_text(encoding="utf-8"))
    assert "img0.jpg" in data["flagged"]


# ---------------------------------------------------------------------------
# 印を付ける
# ---------------------------------------------------------------------------
def test_要修正として印を付ける(ds):
    rv.mark(ds, "img0.jpg", flagged=True)
    assert rv.is_flagged(ds, "img0.jpg") is True
    assert rv.is_reviewed(ds, "img0.jpg") is True


def test_これでよいとして印を付ける(ds):
    """フラグは立てないが、確認したことは記録する"""
    rv.mark(ds, "img0.jpg", flagged=False)
    assert rv.is_flagged(ds, "img0.jpg") is False
    assert rv.is_reviewed(ds, "img0.jpg") is True


def test_フラグを外しても確認済みは残る(ds):
    rv.mark(ds, "img0.jpg", flagged=True)
    rv.mark(ds, "img0.jpg", flagged=False)
    assert rv.is_flagged(ds, "img0.jpg") is False
    assert rv.is_reviewed(ds, "img0.jpg") is True, "見たことは変わらない"


def test_確認そのものを取り消せる(ds):
    rv.mark(ds, "img0.jpg", flagged=True)
    rv.unmark(ds, "img0.jpg")
    assert rv.is_reviewed(ds, "img0.jpg") is False
    assert rv.is_flagged(ds, "img0.jpg") is False


def test_フラグ済みの一覧(ds):
    rv.mark(ds, "img1.jpg", flagged=True)
    rv.mark(ds, "img0.jpg", flagged=True)
    rv.mark(ds, "img2.jpg", flagged=False)
    assert rv.flagged_images(ds) == ["img0.jpg", "img1.jpg"]


# ---------------------------------------------------------------------------
# 進捗
# ---------------------------------------------------------------------------
def test_何も見ていなければ0(ds):
    p = rv.review_progress(ds)
    assert p["total"] == 10 and p["reviewed"] == 0
    assert p["ratio"] == 0.0 and p["done"] is False
    assert rv.progress_label(ds) == ""


def test_進捗を数える(ds):
    for i in range(3):
        rv.mark(ds, f"img{i}.jpg", flagged=(i == 0))
    p = rv.review_progress(ds)
    assert p["reviewed"] == 3 and p["flagged"] == 1
    assert p["ok"] == 2 and p["remaining"] == 7
    assert p["ratio"] == pytest.approx(0.3)


def test_全部見たら完了(ds):
    for i in range(10):
        rv.mark(ds, f"img{i}.jpg", flagged=False)
    p = rv.review_progress(ds)
    assert p["done"] is True and p["remaining"] == 0
    assert "ひととおり確認済み" in rv.progress_label(ds)


def test_表示用の1行(ds):
    rv.mark(ds, "img0.jpg", flagged=True)
    s = rv.progress_label(ds)
    assert "1 / 10" in s and "10%" in s and "要修正 1" in s


def test_退避ディレクトリの画像は数えない(ds):
    bak = ds / "_backup_original" / "images"
    bak.mkdir(parents=True)
    (bak / "a.jpg").write_bytes(b"x")
    assert rv.review_progress(ds)["total"] == 10


# ---------------------------------------------------------------------------
# 画像からデータセットを引く
# ---------------------------------------------------------------------------
def test_画像からデータセットを引く(ds, tmp_path):
    img = ds / "images" / "train" / "img0.jpg"
    assert rv.dataset_of_image(img) == ds


def test_データの外はNone(ds, tmp_path, monkeypatch):
    """アップロードした一時ファイルなどは記録の置き場が決まらない"""
    outside = tmp_path.parent / "よそ.jpg"
    outside.write_bytes(b"x")
    assert rv.dataset_of_image(outside) is None


# ---------------------------------------------------------------------------
# 画面側の集合との同期
# ---------------------------------------------------------------------------
def test_画面の集合を記録に反映する(ds):
    rv.sync_from_names(ds, ["img0.jpg", "img1.jpg"])
    assert rv.flagged_images(ds) == ["img0.jpg", "img1.jpg"]
    assert rv.review_progress(ds)["reviewed"] == 2


def test_同期は既存を消さない(ds):
    rv.mark(ds, "img9.jpg", flagged=False)
    rv.sync_from_names(ds, ["img0.jpg"])
    assert rv.is_reviewed(ds, "img9.jpg") is True
    assert rv.review_progress(ds)["reviewed"] == 2
