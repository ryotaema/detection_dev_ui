"""MLflow 実験比較のうち、外部サービスに依存しない部分"""
from __future__ import annotations

from pathlib import Path

import core.experiments as ex
from core import local_results_curves, preferred_metric_index


# ── 曲線の初期選択 ──────────────────────────────────────────────────────
def test_prefers_map50_over_map50_95():
    """並び順では mAP50-95 が先に来るが、最初に見たいのは mAP50"""
    metrics = ["metrics/mAP50-95B", "metrics/mAP50B", "val/box_loss"]
    assert metrics[preferred_metric_index(metrics)] == "metrics/mAP50B"


def test_prefers_top1_for_classification():
    metrics = ["metrics/accuracy_top1", "metrics/accuracy_top5"]
    assert metrics[preferred_metric_index(metrics)] == "metrics/accuracy_top1"


def test_falls_back_to_first_metric():
    assert preferred_metric_index(["val/box_loss", "train/cls_loss"]) == 0


def test_empty_metrics_is_safe():
    assert preferred_metric_index([]) == 0


# ── results.csv からのフォールバック ────────────────────────────────────
def test_reads_results_csv(tmp_path: Path):
    run = tmp_path / "run1"
    run.mkdir()
    (run / "results.csv").write_text(
        "epoch, metrics/mAP50(B)\n1, 0.5\n2, 0.7\n")
    curves = local_results_curves(["run1"], tmp_path)
    assert list(curves) == ["run1"]
    assert len(curves["run1"]) == 2
    # 列名の前後空白は落として扱う
    assert "metrics/mAP50(B)" in curves["run1"].columns


def test_skips_runs_without_results(tmp_path: Path):
    (tmp_path / "empty").mkdir()
    assert local_results_curves(["empty", "missing"], tmp_path) == {}


def test_broken_csv_does_not_raise(tmp_path: Path):
    """壊れたファイルでも例外を投げないこと（読めたかどうかは問わない）"""
    run = tmp_path / "bad"
    run.mkdir()
    (run / "results.csv").write_bytes(b"\x00\x01\x02")
    assert isinstance(local_results_curves(["bad"], tmp_path), dict)


# ── MLflow が落ちていても壊れないこと ───────────────────────────────────
class _Boom:
    def __getattr__(self, name):
        raise RuntimeError("mlflow down")


def test_functions_degrade_gracefully(monkeypatch):
    """接続できないときに例外を投げず、空を返して UI を守る"""
    monkeypatch.setattr(ex, "_client", lambda: _Boom())

    ok, err = ex.mlflow_available()
    assert ok is False and "mlflow down" in err
    assert ex.list_experiments() == []
    assert ex.list_runs(["1"]) == []
    assert ex.available_metrics(["a"]) == []
    assert ex.metric_history(["a"], "m") == {}
    assert ex.run_detail("a") is None


def test_list_runs_without_experiments_returns_empty():
    assert ex.list_runs([]) == []
