# =============================================================================
# ハイパーパラメータ探索のテスト
#
#   実際の探索は学習を何度も回すのでテストしない。
#   ここで見るのは「始める前に所要時間を出せるか」「結果を読めるか」。
#   押してから8時間かかると気づくのが最悪なので、見積もりは重要。
# =============================================================================
from __future__ import annotations

from pathlib import Path

import pytest

from core import tuning as tn


@pytest.fixture
def models(tmp_path, monkeypatch):
    monkeypatch.setattr(tn, "MODELS_DIR", tmp_path)
    return tmp_path


def _run_with_csv(root, name, epochs=10, total_time=600.0, dataset=None):
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    lines = ["epoch,time,metrics/mAP50-95(B)"]
    for i in range(1, epochs + 1):
        lines.append(f"{i},{total_time * i / epochs},{0.5 + i * 0.01}")
    (d / "results.csv").write_text("\n".join(lines))
    if dataset:
        import json
        (d / ".provenance.json").write_text(
            json.dumps({"dataset": {"name": dataset}}))
    return d


# ---------------------------------------------------------------------------
# 探索空間のプリセット
# ---------------------------------------------------------------------------
def test_プリセットの形が揃っている():
    assert set(tn.SEARCH_PRESETS) == {"lr", "loss", "aug", "all"}
    for k, v in tn.SEARCH_PRESETS.items():
        assert v["label"] and v["desc"], k
        assert "space" in v, k


def test_学習率のプリセットは項目が少ない():
    """少ない回数でも効くよう絞ってある"""
    assert len(tn.SEARCH_PRESETS["lr"]["space"]) <= 5


def test_すべてはNoneで本家に任せる():
    """自前で26項目を書き写すと本家とずれるので None を渡す"""
    assert tn.SEARCH_PRESETS["all"]["space"] is None


def test_範囲は下限より上限が大きい():
    for k, v in tn.SEARCH_PRESETS.items():
        for name, rng in (v["space"] or {}).items():
            assert rng[0] < rng[1], f"{k}.{name}"


# ---------------------------------------------------------------------------
# 所要時間の見積もり（押す前に出すのが要点）
# ---------------------------------------------------------------------------
def test_記録が無ければ分からないと言う(models):
    """推測で数字を出すより「分からない」と言うほうがよい"""
    est = tn.estimate_tuning(10, 50)
    assert est["known"] is False
    assert "見積もれません" in est["text"]


def test_過去の学習から1エポックを推定する(models):
    _run_with_csv(models, "run_a", epochs=10, total_time=600.0)
    assert tn.estimate_epoch_seconds() == pytest.approx(60.0)


def test_見積もりは回数とエポックに比例する(models):
    _run_with_csv(models, "run_a", epochs=10, total_time=600.0)
    e1 = tn.estimate_tuning(10, 50)
    e2 = tn.estimate_tuning(20, 50)
    assert e1["known"] and e2["known"]
    assert e2["total"] == pytest.approx(e1["total"] * 2)
    assert e1["per_run"] == pytest.approx(60.0 * 50)


def test_同じデータセットの実測を優先する(models):
    """速いデータで測った値を、遅いデータの見積もりに使わない"""
    _run_with_csv(models, "fast", epochs=10, total_time=100.0, dataset="small")
    _run_with_csv(models, "slow", epochs=10, total_time=1000.0, dataset="big")
    assert tn.estimate_epoch_seconds(Path("/x/big")) == pytest.approx(100.0)
    assert tn.estimate_epoch_seconds(Path("/x/small")) == pytest.approx(10.0)


def test_時間の表し方():
    assert "秒" in tn._fmt_dur(30)
    assert "分" in tn._fmt_dur(600)
    assert "時間" in tn._fmt_dur(7200)


def test_壊れたCSVは飛ばす(models):
    (models / "bad").mkdir()
    (models / "bad" / "results.csv").write_text("これは,CSVでは\nない")
    assert tn.estimate_epoch_seconds() is None


def test_timeの無いCSVは使わない(models):
    d = models / "notime"
    d.mkdir()
    (d / "results.csv").write_text("epoch,loss\n1,0.5\n2,0.4\n")
    assert tn.estimate_epoch_seconds() is None


# ---------------------------------------------------------------------------
# 結果の読み取り
# ---------------------------------------------------------------------------
def _tune_csv(d, rows):
    d.mkdir(parents=True, exist_ok=True)
    lines = ["fitness,lr0,momentum"]
    for f, lr, mo in rows:
        lines.append(f"{f},{lr},{mo}")
    (d / "tune_results.csv").write_text("\n".join(lines))
    return d


def test_探索結果を読む(tmp_path):
    d = _tune_csv(tmp_path / "t", [(0.70, 0.01, 0.9), (0.75, 0.005, 0.85)])
    rows = tn.read_tune_results(d)
    assert len(rows) == 2
    assert rows[0]["iteration"] == 1 and rows[1]["fitness"] == 0.75


def test_結果が無ければ空(tmp_path):
    assert tn.read_tune_results(tmp_path / "ない") == []


def test_最良を選ぶ(tmp_path):
    d = _tune_csv(tmp_path / "t", [(0.70, 0.01, 0.9), (0.82, 0.005, 0.85),
                                   (0.61, 0.02, 0.7)])
    best = tn.best_of(tn.read_tune_results(d))
    assert best["fitness"] == 0.82 and best["iteration"] == 2


def test_結果が無ければ最良もNone():
    assert tn.best_of([]) is None


def test_最良のパラメータを読む(tmp_path):
    d = tmp_path / "t"
    d.mkdir()
    (d / "best_hyperparameters.yaml").write_text("lr0: 0.00832\nmomentum: 0.891\n")
    assert tn.read_best_params(d) == {"lr0": 0.00832, "momentum": 0.891}


def test_壊れたyamlでも落ちない(tmp_path):
    d = tmp_path / "t"
    d.mkdir()
    (d / "best_hyperparameters.yaml").write_text("{ 壊れている")
    assert tn.read_best_params(d) == {}


def test_過去の探索を新しい順に返す(models):
    import os, time
    for i, name in enumerate(["old", "new"]):
        d = _tune_csv(models / name, [(0.5, 0.01, 0.9)])
        t = time.time() - (100 if name == "old" else 0)
        os.utime(d, (t, t))
    assert [p.name for p in tn.find_tune_dirs(models)] == ["new", "old"]


# ---------------------------------------------------------------------------
# 結果を学習プリセットに変換
# ---------------------------------------------------------------------------
def test_探索結果をプリセットにできる():
    """探索して終わりでは意味がないので、そのまま学習に使える形にする"""
    got = tn.params_to_preset({"lr0": 0.00832, "momentum": 0.891},
                              base={"model": "yolo11s", "epochs": 100})
    assert got["model"] == "yolo11s" and got["epochs"] == 100
    assert got["lr0"] == 0.00832


def test_数値でない値は取り込まない():
    got = tn.params_to_preset({"lr0": 0.01, "note": "メモ"})
    assert "note" not in got and got["lr0"] == 0.01
