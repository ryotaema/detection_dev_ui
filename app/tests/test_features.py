# =============================================================================
# オプション機能の表示切り替えのテスト
#
#   既定はすべてオフ。clone した直後は素の状態から始まること。
#   壊れた設定ファイルで画面が落ちないこと。
# =============================================================================
from __future__ import annotations

import json

import pytest

from ui import features as F


@pytest.fixture
def store(tmp_path, monkeypatch):
    """設定ファイルを差し替えて隔離する"""
    path = tmp_path / ".user_features.json"
    monkeypatch.setattr(F, "FEATURES_PATH", path)
    return path


def test_既定は何も有効でない(store):
    """clone した直後は素の状態から始まる"""
    assert F.load_enabled() == set()


def test_保存して読み直せる(store):
    assert F.save_enabled(["mosaic"])
    assert F.load_enabled() == {"mosaic"}
    assert json.loads(store.read_text(encoding="utf-8")) == ["mosaic"]


def test_知らない名前は捨てる(store):
    """設定ファイルを手で書き換えられても、知らない機能を有効にしない"""
    F.save_enabled(["mosaic", "でたらめ"])
    assert F.load_enabled() == {"mosaic"}


def test_壊れた設定でも落ちない(store):
    store.write_text("{ これは JSON ではない", encoding="utf-8")
    assert F.load_enabled() == set()


def test_想定外の形でも落ちない(store):
    store.write_text('{"mosaic": true}', encoding="utf-8")   # リストではない
    assert F.load_enabled() == set()


def test_全部外せる(store):
    F.save_enabled(["mosaic"])
    F.save_enabled([])
    assert F.load_enabled() == set()


def test_登録内容の形が揃っている():
    for name, info in F.OPTIONAL_FEATURES.items():
        assert info["kind"] in ("inline", "tab"), name
        for key in ("label", "desc", "where"):
            assert info.get(key), f"{name} に {key} が無い"


def test_タブ型だけが順序に入る():
    for name in F.TAB_FEATURE_ORDER:
        assert F.OPTIONAL_FEATURES[name]["kind"] == "tab"
