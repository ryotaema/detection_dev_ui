# =============================================================================
# MLflow の実験比較
#
#   学習ごとの記録は MLflow に入っているが、これまで UI からは外部リンクを
#   開くしかなかった。「どの設定でどれだけ精度が出たか」を並べて見られるようにする。
#
#   MLflow が落ちていても UI が壊れないよう、失敗時は空を返して理由を添える。
# =============================================================================
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .config import MLFLOW_URI

# 学習曲線として並べる価値のあるメトリクス（前方一致で拾う）
CURVE_METRIC_HINTS = ("metrics/mAP50", "metrics/mAP50-95", "metrics/precision",
                      "metrics/recall", "metrics/accuracy_top1",
                      "val/box_loss", "val/cls_loss", "train/box_loss")

# 曲線の初期選択。mAP50 を最初に見たいので mAP50-95 より優先する
PREFERRED_CURVE_METRICS = ("metrics/mAP50B", "metrics/mAP50(B)",
                           "metrics/accuracy_top1", "metrics/mAP50-95B")


def preferred_metric_index(metrics: list[str]) -> int:
    """曲線表示で最初に出すメトリクスの位置を返す"""
    for want in PREFERRED_CURVE_METRICS:
        if want in metrics:
            return metrics.index(want)
    return 0

# 一覧表に出すパラメータ（多すぎると読めないので絞る）
SUMMARY_PARAMS = ("model", "epochs", "batch", "imgsz", "optimizer", "lr0", "data")


def _client():
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    return mlflow.tracking.MlflowClient()


def mlflow_available() -> tuple[bool, str]:
    """MLflow に接続できるか。UI で理由を出せるようメッセージも返す。"""
    try:
        _client().search_experiments(max_results=1)
        return True, ""
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def list_experiments() -> list[dict]:
    """実験の一覧（run 数つき）"""
    try:
        c = _client()
        out = []
        for e in c.search_experiments():
            try:
                n = len(c.search_runs([e.experiment_id], max_results=1000))
            except Exception:
                n = 0
            out.append({"id": e.experiment_id, "name": e.name, "n_runs": n})
        return sorted(out, key=lambda d: -d["n_runs"])
    except Exception:
        return []


def list_runs(experiment_ids: list[str], max_results: int = 200) -> list[dict]:
    """run の一覧。主要なメトリクスとパラメータを平坦化して返す。"""
    if not experiment_ids:
        return []
    try:
        c = _client()
        runs = c.search_runs(experiment_ids, max_results=max_results,
                             order_by=["attributes.start_time DESC"])
    except Exception:
        return []

    out = []
    for r in runs:
        info, data = r.info, r.data
        row = {
            "run_id": info.run_id,
            "run_name": info.run_name or info.run_id[:8],
            "status": info.status,
            "start_time": info.start_time,
            "duration_min": (
                round((info.end_time - info.start_time) / 60000, 1)
                if info.end_time and info.start_time else None
            ),
        }
        # 代表的な最終メトリクス。
        # MLflow はメトリクス名から括弧を落とすため mAP50(B) → mAP50B になる。
        # 記録側の違いに備えて候補をいくつか見る。
        for keys, label in (
            (("metrics/mAP50B", "metrics/mAP50(B)"), "mAP50"),
            (("metrics/mAP50-95B", "metrics/mAP50-95(B)"), "mAP50-95"),
            (("metrics/precisionB", "metrics/precision(B)"), "precision"),
            (("metrics/recallB", "metrics/recall(B)"), "recall"),
            (("metrics/accuracy_top1",), "top1"),
        ):
            for key in keys:
                if key in data.metrics:
                    row[label] = round(float(data.metrics[key]), 4)
                    break
        for p in SUMMARY_PARAMS:
            if p in data.params:
                row[p] = data.params[p]
        out.append(row)
    return out


def available_metrics(run_ids: list[str]) -> list[str]:
    """選んだ run が持つメトリクスのうち、曲線として見る価値があるものを返す"""
    try:
        c = _client()
    except Exception:
        return []
    keys: set[str] = set()
    for rid in run_ids:
        try:
            keys.update(c.get_run(rid).data.metrics.keys())
        except Exception:
            continue
    picked = [k for k in sorted(keys)
              if any(k.startswith(h) for h in CURVE_METRIC_HINTS)]
    return picked or sorted(keys)


def metric_history(run_ids: list[str], metric_key: str) -> dict:
    """run ごとの学習曲線を {run名: {step: value}} で返す"""
    try:
        c = _client()
    except Exception:
        return {}
    series: dict[str, dict[int, float]] = {}
    for rid in run_ids:
        try:
            run = c.get_run(rid)
            name = run.info.run_name or rid[:8]
            points = c.get_metric_history(rid, metric_key)
        except Exception:
            continue
        if points:
            series[name] = {p.step: p.value for p in points}
    return series


def run_detail(run_id: str) -> Optional[dict]:
    """1 run の全パラメータ・メトリクス"""
    try:
        r = _client().get_run(run_id)
    except Exception:
        return None
    return {
        "run_id": run_id,
        "run_name": r.info.run_name or run_id[:8],
        "status": r.info.status,
        "params": dict(r.data.params),
        "metrics": {k: float(v) for k, v in r.data.metrics.items()},
        "url": f"{MLFLOW_URI}/#/experiments/{r.info.experiment_id}/runs/{run_id}",
    }


def local_results_curves(run_names: list[str], models_dir: Path) -> dict:
    """MLflow を使わず `models/<run>/results.csv` から学習曲線を作る。

    MLflow が落ちていても学習の経過は見られるようにするためのフォールバック。
    """
    out: dict[str, "object"] = {}
    for name in run_names:
        csv = Path(models_dir) / name / "results.csv"
        if not csv.exists():
            continue
        try:
            import pandas as pd
            df = pd.read_csv(csv)
            df.columns = [c.strip() for c in df.columns]
            out[name] = df
        except Exception:
            continue
    return out
