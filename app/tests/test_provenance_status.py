# =============================================================================
# 来歴の status / tags のテスト
#
#   既存の来歴を壊さないことがいちばん大事。
#   update_provenance() は「差し替えたい項目だけ」を触ること。
# =============================================================================
from __future__ import annotations

import json

import pytest

from core.config import PROVENANCE_FILE
from core.provenance import (
    DATASET_STATUSES, DEFAULT_DATASET_STATUS, DEFAULT_MODEL_STATUS,
    MAX_TAG_LEN, MAX_TAGS, MODEL_STATUSES, collect_tags, default_status,
    normalize_tags, read_note, read_provenance, read_status, read_tags,
    record_dataset_provenance, snapshot_dataset_source, status_label,
    status_table, update_provenance, write_provenance,
)


@pytest.fixture
def ds(tmp_path):
    """来歴つきのデータセットを模したディレクトリ"""
    d = tmp_path / "dataset_a"
    (d / "images" / "train").mkdir(parents=True)
    (d / "images" / "val").mkdir(parents=True)
    for i in range(3):
        (d / "images" / "train" / f"{i}.jpg").write_bytes(b"x")
    (d / "images" / "val" / "0.jpg").write_bytes(b"x")
    write_provenance(d, {
        "created_at": "2026-01-01 00:00:00",
        "dataset": "dataset_a",
        "source": "cvat",
        "task_type": "detect",
        "labels": ["bell_pepper"],
        "cvat_tasks": [{"id": 7, "name": "task7"}],
        "counts": {"train": 3, "val": 1},
    })
    return d


# ---------------------------------------------------------------------------
# 語彙
# ---------------------------------------------------------------------------
def test_状態の語彙は設計どおり():
    assert list(DATASET_STATUSES) == [
        "draft", "auto_annotated", "reviewed", "test_only", "archived"]
    assert list(MODEL_STATUSES) == [
        "experimental", "candidate", "production", "deprecated"]
    assert DEFAULT_DATASET_STATUS == "draft"
    assert DEFAULT_MODEL_STATUS == "experimental"


def test_種別で参照する表が切り替わる():
    assert status_table("dataset") is DATASET_STATUSES
    assert status_table("model") is MODEL_STATUSES
    assert default_status("model") == "experimental"
    assert default_status("dataset") == "draft"


def test_表示用ラベル():
    assert status_label("reviewed") == "🟢 精査済み"
    assert status_label("production", "model") == "🚀 実用"
    assert "❔" in status_label("知らない値")
    assert "❔" in status_label("")


# ---------------------------------------------------------------------------
# タグの正規化
# ---------------------------------------------------------------------------
def test_カンマ区切りの文字列を分解する():
    assert normalize_tags("屋内, ペッパー ,自動アノテ由来") == \
        ["屋内", "ペッパー", "自動アノテ由来"]


def test_空とNoneを落とす():
    assert normalize_tags(None) == []
    assert normalize_tags("") == []
    assert normalize_tags(" , ,  ") == []
    assert normalize_tags(["a", "", "  ", "b"]) == ["a", "b"]


def test_重複を除き入力順を保つ():
    assert normalize_tags(["b", "a", "b", "a"]) == ["b", "a"]


def test_長すぎるタグと多すぎるタグを切る():
    assert normalize_tags(["あ" * 100])[0] == "あ" * MAX_TAG_LEN
    assert len(normalize_tags([f"t{i}" for i in range(MAX_TAGS + 20)])) == MAX_TAGS


# ---------------------------------------------------------------------------
# 読み取り
# ---------------------------------------------------------------------------
def test_記録が無ければ既定の状態を返す(tmp_path):
    """来歴を入れる前に作られたものを勝手に精査済みにしない"""
    d = tmp_path / "empty"
    d.mkdir()
    assert read_status(d) == "draft"
    assert read_status(d, "model") == "experimental"
    assert read_tags(d) == []
    assert read_note(d) == ""


def test_status未設定の既存来歴も既定になる(ds):
    assert read_status(ds) == "draft"
    assert read_tags(ds) == []


def test_知らない状態が入っていても既定に落とす(ds):
    prov = read_provenance(ds)
    prov["status"] = "でたらめ"
    write_provenance(ds, prov)
    assert read_status(ds) == "draft"


# ---------------------------------------------------------------------------
# 書き込み（既存を壊さないこと）
# ---------------------------------------------------------------------------
def test_状態を付けても既存の来歴が残る(ds):
    before = read_provenance(ds)
    res = update_provenance(ds, status="reviewed", tags="屋内,ペッパー", note="メモ")
    assert res["ok"], res["error"]

    after = read_provenance(ds)
    for key in ("created_at", "dataset", "source", "task_type",
                "labels", "cvat_tasks", "counts"):
        assert after[key] == before[key], f"{key} が変わってしまった"
    assert after["status"] == "reviewed"
    assert after["tags"] == ["屋内", "ペッパー"]
    assert after["note"] == "メモ"
    assert "status_updated_at" in after


def test_Noneを渡した項目は触らない(ds):
    update_provenance(ds, status="reviewed", tags=["a"], note="最初")

    update_provenance(ds, status="archived")          # tags と note は触らない
    assert read_tags(ds) == ["a"]
    assert read_note(ds) == "最初"
    assert read_status(ds) == "archived"

    update_provenance(ds, tags=["b"])                 # status と note は触らない
    assert read_status(ds) == "archived"
    assert read_note(ds) == "最初"
    assert read_tags(ds) == ["b"]


def test_来歴が無いディレクトリにも付けられる(tmp_path):
    d = tmp_path / "no_prov"
    d.mkdir()
    res = update_provenance(d, status="test_only", tags="評価用")
    assert res["ok"]
    assert read_status(d) == "test_only"
    assert read_tags(d) == ["評価用"]
    # 後から足した記録だと分かるようにしておく
    assert read_provenance(d)["provenance_added_later"] is True
    assert read_provenance(d)["source"] == "unknown"


def test_知らない状態は拒否して書き込まない(ds):
    before = read_provenance(ds)
    res = update_provenance(ds, status="でたらめ")
    assert not res["ok"]
    assert "知らない状態" in res["error"]
    assert read_provenance(ds) == before, "拒否したのに書き換わっている"


def test_存在しないディレクトリは拒否する(tmp_path):
    res = update_provenance(tmp_path / "ない", status="reviewed")
    assert not res["ok"]
    assert "ディレクトリがありません" in res["error"]


def test_モデルの状態も扱える(tmp_path):
    run = tmp_path / "run1"
    run.mkdir()
    assert update_provenance(run, kind="model", status="production")["ok"]
    assert read_status(run, "model") == "production"
    # データセットの語彙は使えない
    assert not update_provenance(run, kind="model", status="reviewed")["ok"]


def test_書き込んだJSONが読める形になっている(ds):
    update_provenance(ds, status="reviewed", tags=["屋内"])
    raw = (ds / PROVENANCE_FILE).read_text(encoding="utf-8")
    data = json.loads(raw)
    assert data["status"] == "reviewed"
    assert "屋内" in raw, "日本語がエスケープされている"


# ---------------------------------------------------------------------------
# 生成時に状態を付ける / 統合元を残す
# ---------------------------------------------------------------------------
def test_生成時に状態とタグを付けられる(tmp_path):
    d = tmp_path / "new_ds"
    (d / "images" / "train").mkdir(parents=True)
    prov = record_dataset_provenance(
        d, source="cvat", task_type="detect",
        status="auto_annotated", tags="自動アノテ由来, 屋外")
    assert prov["status"] == "auto_annotated"
    assert prov["tags"] == ["自動アノテ由来", "屋外"]
    assert read_status(d) == "auto_annotated"


def test_生成時に状態を指定しなければ既定になる(tmp_path):
    d = tmp_path / "new_ds2"
    (d / "images" / "train").mkdir(parents=True)
    assert record_dataset_provenance(d, source="upload_zip")["status"] == "draft"


def test_統合元のスナップショットを取れる(ds):
    update_provenance(ds, status="reviewed", tags=["屋内"])
    snap = snapshot_dataset_source(ds)
    assert snap["name"] == "dataset_a"
    assert snap["counts"] == {"train": 3, "val": 1}
    assert snap["status"] == "reviewed"
    assert snap["tags"] == ["屋内"]
    assert snap["task_type"] == "detect"
    assert snap["cvat_tasks"] == [{"id": 7, "name": "task7"}]


def test_統合の来歴に親が残る(tmp_path, ds):
    merged = tmp_path / "merged"
    (merged / "images" / "train").mkdir(parents=True)
    prov = record_dataset_provenance(
        merged, source="merge", sources=[snapshot_dataset_source(ds)])
    assert len(prov["sources"]) == 1
    assert prov["sources"][0]["name"] == "dataset_a"
    # 親が消えても記録は残る
    import shutil
    shutil.rmtree(ds)
    assert read_provenance(merged)["sources"][0]["counts"] == {"train": 3, "val": 1}


# ---------------------------------------------------------------------------
# 絞り込み用のタグ収集
# ---------------------------------------------------------------------------
def test_使われているタグを集める(tmp_path):
    dirs = []
    for i, tags in enumerate([["屋内", "ペッパー"], ["屋外", "ペッパー"], []]):
        d = tmp_path / f"d{i}"
        d.mkdir()
        update_provenance(d, tags=tags)
        dirs.append(d)
    assert collect_tags(dirs) == sorted(["屋内", "屋外", "ペッパー"])


# ---------------------------------------------------------------------------
# 使用実績の逆引き（このデータセットは消してよいか）
# ---------------------------------------------------------------------------
from core.provenance import (  # noqa: E402
    dataset_usage_summary, models_using_dataset,
)


def _make_run(models_root, name, dataset_name, sources=None, status=None):
    run = models_root / name
    run.mkdir(parents=True, exist_ok=True)
    prov = {
        "trained_at": "2026-08-04 10:00:00",
        "run": name,
        "base_model": "/x/yolo11s.pt",
        "dataset": {
            "name": dataset_name,
            "counts_at_train": {"train": 100, "val": 20},
            "provenance": {"sources": sources} if sources else {},
        },
    }
    if status:
        prov["status"] = status
    write_provenance(run, prov)
    return run


def test_そのデータで学習したモデルを見つける(tmp_path):
    models = tmp_path / "models"
    _make_run(models, "run_a", "ds1")
    _make_run(models, "run_b", "ds2")

    got = models_using_dataset(tmp_path / "data" / "ds1", models_root=models)
    assert [m["run"] for m in got] == ["run_a"]
    assert got[0]["counts_at_train"] == {"train": 100, "val": 20}
    assert got[0]["via"] == ""


def test_統合の親としても拾う(tmp_path):
    """統合データセットで学習した場合、親も「使われている」"""
    models = tmp_path / "models"
    _make_run(models, "run_m", "merged", sources=[{"name": "ds1"}, {"name": "ds2"}])

    got = models_using_dataset(tmp_path / "data" / "ds1", models_root=models)
    assert len(got) == 1
    assert got[0]["via"] == "merged", "統合経由だと分かること"


def test_使われていなければ空(tmp_path):
    models = tmp_path / "models"
    _make_run(models, "run_a", "ds1")
    assert models_using_dataset(tmp_path / "data" / "ない", models_root=models) == []


def test_来歴の無いrunは無視する(tmp_path):
    models = tmp_path / "models"
    (models / "no_prov").mkdir(parents=True)
    assert models_using_dataset(tmp_path / "data" / "ds1", models_root=models) == []


def test_モデルの状態も一緒に返す(tmp_path):
    models = tmp_path / "models"
    run = _make_run(models, "run_a", "ds1")
    update_provenance(run, kind="model", status="production")
    got = models_using_dataset(tmp_path / "data" / "ds1", models_root=models)
    assert got[0]["status"] == "production"


def test_削除してよいかの要約(tmp_path, monkeypatch):
    """使われていなければ消してよい、実用モデルが使っていれば消せない"""
    import core.config as cfg
    import core.provenance as pv

    models = tmp_path / "models"
    models.mkdir()
    monkeypatch.setattr(cfg, "MODELS_DIR", models)
    monkeypatch.setattr(pv, "MODELS_DIR", models, raising=False)

    ds = tmp_path / "data" / "ds1"
    ds.mkdir(parents=True)

    s = dataset_usage_summary(ds)
    assert s["n_models"] == 0 and s["safe_to_delete"] is True

    run = _make_run(models, "run_a", "ds1")
    update_provenance(run, kind="model", status="production")

    s2 = dataset_usage_summary(ds)
    assert s2["n_models"] == 1, "使われているのに 0 件と出ている"
    assert s2["safe_to_delete"] is False
    assert [m["run"] for m in s2["in_production"]] == ["run_a"]
