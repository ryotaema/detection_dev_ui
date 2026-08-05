# =============================================================================
# モデルのお気に入り・使用回数・新着のテスト
#
#   ねらいは「せっかく学習したのに使い忘れる」を防ぐこと。
#   新着判定（作ったのに一度も使っていない）が要。
# =============================================================================
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from core import model_prefs as mp


@pytest.fixture
def models(tmp_path, monkeypatch):
    """models/ を差し替えて隔離する"""
    root = tmp_path / "models"
    root.mkdir()
    monkeypatch.setattr(mp, "MODELS_DIR", root)
    monkeypatch.setattr(mp, "PREFS_PATH", root / ".model_prefs.json")
    return root


def _model(root, name, *, age_days=0.0):
    p = root / name / "weights" / "best.pt"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x" * 1000)
    if age_days:
        t = time.time() - age_days * 86400
        os.utime(p, (t, t))
    return p


# ---------------------------------------------------------------------------
# 記録の読み書き
# ---------------------------------------------------------------------------
def test_記録が無ければ空(models):
    assert mp.load_prefs() == {}
    assert mp.is_favorite(_model(models, "a")) is False
    assert mp.use_count(_model(models, "a")) == 0


def test_壊れた記録でも落ちない(models):
    (models / ".model_prefs.json").write_text("{ 壊れている", encoding="utf-8")
    assert mp.load_prefs() == {}


def test_リストが入っていても落ちない(models):
    (models / ".model_prefs.json").write_text("[1,2,3]", encoding="utf-8")
    assert mp.load_prefs() == {}


def test_モデルの位置で識別する(models):
    a = _model(models, "run_a")
    assert mp._key(a) == "run_a/weights/best.pt"


# ---------------------------------------------------------------------------
# お気に入り
# ---------------------------------------------------------------------------
def test_お気に入りを切り替えられる(models):
    a = _model(models, "a")
    assert mp.toggle_favorite(a) is True
    assert mp.is_favorite(a) is True
    assert mp.toggle_favorite(a) is False
    assert mp.is_favorite(a) is False


def test_お気に入りは他のモデルに影響しない(models):
    a, b = _model(models, "a"), _model(models, "b")
    mp.toggle_favorite(a)
    assert mp.is_favorite(a) and not mp.is_favorite(b)


# ---------------------------------------------------------------------------
# 使用回数
# ---------------------------------------------------------------------------
def test_使うたびに数える(models):
    a = _model(models, "a")
    mp.record_use(a)
    mp.record_use(a)
    assert mp.use_count(a) == 2
    assert mp.last_used(a), "最後に使った日時が入ること"


def test_用途ごとの内訳も残す(models):
    a = _model(models, "a")
    mp.record_use(a, "infer")
    mp.record_use(a, "infer")
    mp.record_use(a, "deploy")
    e = mp.load_prefs()[mp._key(a)]
    assert e["actions"] == {"infer": 2, "deploy": 1}
    assert e["uses"] == 3


def test_使っていなければ未使用(models):
    a = _model(models, "a")
    assert mp.is_unused(a) is True
    mp.record_use(a)
    assert mp.is_unused(a) is False


# ---------------------------------------------------------------------------
# 新着（使い忘れ防止の要）
# ---------------------------------------------------------------------------
def test_作ったばかりで未使用なら新着(models):
    a = _model(models, "a", age_days=1)
    assert mp.is_new(a) is True


def test_一度でも使えば新着から外れる(models):
    """「使い忘れ」を拾うのが目的なので、使ったら知らせない"""
    a = _model(models, "a", age_days=1)
    mp.record_use(a)
    assert mp.is_new(a) is False


def test_古すぎるものは新着にしない(models):
    a = _model(models, "a", age_days=90)
    assert mp.is_new(a) is False


def test_新着の期間を変えられる(models):
    a = _model(models, "a", age_days=45)
    assert mp.is_new(a, days=30) is False
    assert mp.is_new(a, days=60) is True


def test_未使用の新着だけを新しい順に返す(models):
    old = _model(models, "old", age_days=90)      # 古い
    used = _model(models, "used", age_days=2)     # 使用済み
    n1 = _model(models, "n1", age_days=5)
    n2 = _model(models, "n2", age_days=1)
    mp.record_use(used)

    got = mp.unused_new_models([old, used, n1, n2])
    assert [Path(x["path"]).parent.parent.name for x in got] == ["n2", "n1"]


# ---------------------------------------------------------------------------
# 並べ替え
# ---------------------------------------------------------------------------
def test_並べ方の選択肢(models):
    assert set(mp.SORT_OPTIONS) == {
        "recommended", "recent", "used", "last_used", "map", "name"}


def test_新しい順(models):
    a = _model(models, "a", age_days=10)
    b = _model(models, "b", age_days=1)
    got = mp.sort_models([a, b], "recent")
    assert [x["key"].split("/")[0] for x in got] == ["b", "a"]


def test_よく使う順(models):
    a, b = _model(models, "a"), _model(models, "b")
    mp.record_use(b)
    mp.record_use(b)
    mp.record_use(a)
    got = mp.sort_models([a, b], "used")
    assert [x["key"].split("/")[0] for x in got] == ["b", "a"]


def test_名前順(models):
    b, a = _model(models, "b"), _model(models, "a")
    got = mp.sort_models([b, a], "name")
    assert [x["key"].split("/")[0] for x in got] == ["a", "b"]


def test_おすすめ順はお気に入りを先に出す(models):
    a = _model(models, "a", age_days=10)
    b = _model(models, "b", age_days=1)
    mp.toggle_favorite(a)
    got = mp.sort_models([a, b], "recommended")
    assert got[0]["key"].startswith("a"), "お気に入りが先頭に来ること"


def test_評価の無いものは精度順で後ろへ(models):
    a, b = _model(models, "a"), _model(models, "b")
    got = mp.sort_models([a, b], "map")
    assert all(x["map"] is None for x in got)   # どちらも未評価でも落ちない


def test_表示用の情報がそろう(models):
    a = _model(models, "a", age_days=3)
    mp.toggle_favorite(a)
    mp.record_use(a)
    d = mp.describe(a)
    for k in ("path", "key", "favorite", "uses", "last_used",
              "age_days", "is_new", "map", "status", "size_mb"):
        assert k in d, k
    assert d["favorite"] is True and d["uses"] == 1
    assert d["is_new"] is False


# ---------------------------------------------------------------------------
# 後片付け
# ---------------------------------------------------------------------------
def test_消えたモデルの記録を落とす(models):
    a, b = _model(models, "a"), _model(models, "b")
    mp.toggle_favorite(a)
    mp.toggle_favorite(b)
    assert len(mp.load_prefs()) == 2

    assert mp.prune_prefs([a]) == 1
    assert list(mp.load_prefs()) == [mp._key(a)]


def test_残すものが同じなら何もしない(models):
    a = _model(models, "a")
    mp.toggle_favorite(a)
    assert mp.prune_prefs([a]) == 0
