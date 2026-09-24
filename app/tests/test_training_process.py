# =============================================================================
# 学習を別プロセスで回す仕組みのテスト
#
#   - 子プロセスの出力（ログ行・イベント行）を共有状態へ正しく流すか
#   - 2 つのタブから同時に押しても学習が 2 本走らないか
#   - 学習と探索が同じ GPU を奪い合わないか
#   - 実際に子プロセスで学習が完走し、停止・失敗も画面に届くか（ultralytics があれば）
# =============================================================================
from __future__ import annotations

import threading
import time

import pytest

from core import training as tr
from core.state import _get_train_shared, _get_tune_shared


def _state():
    return {"log": [], "progress": 0, "running": False, "error": None,
            "model_path": None, "metrics_history": [], "stop_requested": False}


def test_イベント行は状態に_ログ行はログに流す():
    st, lk = _state(), threading.Lock()
    tr._apply_output_line(b'@@TRAIN_EVENT@@ {"type": "progress", "value": 40}\n', st, lk)
    tr._apply_output_line(b'@@TRAIN_EVENT@@ {"type": "metrics", "row": {"epoch": 1}}\n', st, lk)
    tr._apply_output_line(b'@@TRAIN_EVENT@@ {"type": "model_path", "value": "/m/best.pt"}\n', st, lk)
    tr._apply_output_line("学習中\n".encode(), st, lk)
    assert st["progress"] == 40
    assert st["metrics_history"] == [{"epoch": 1}]
    assert st["model_path"] == "/m/best.pt"
    assert st["log"] == ["学習中"]


def test_進捗バーの上書きは最後の姿だけ残す():
    st, lk = _state(), threading.Lock()
    tr._apply_output_line(b"  1/10 10%\r  5/10 50%\x1b[K\r 10/10 100%\x1b[K\n", st, lk)
    assert st["log"] == [" 10/10 100%"]


def test_ログは上限を超えたら古いものから捨てる(monkeypatch):
    monkeypatch.setattr(tr, "MAX_LOG_LINES", 5)
    st, lk = _state(), threading.Lock()
    for i in range(12):
        tr._apply_output_line(f"line{i}\n".encode(), st, lk)
    assert st["log"] == [f"line{i}" for i in range(7, 12)]


@pytest.fixture
def clean_shared():
    """共有状態はモジュール変数なので、テストの前後で元に戻す"""
    train, tl = _get_train_shared()
    tune, ul = _get_tune_shared()
    saved = (dict(train), dict(tune))
    yield train, tune
    with tl:
        train.clear()
        train.update(saved[0])
    with ul:
        tune.clear()
        tune.update(saved[1])


def test_学習中なら2本目は始めない(clean_shared, monkeypatch):
    train, _ = clean_shared
    started = []
    monkeypatch.setattr(tr, "_train_worker", lambda *a: started.append(a))
    train["running"] = False

    ok1, _ = tr.start_training("d.yaml", "m.pt", 1, 1, "p", "r", {})
    ok2, why = tr.start_training("d.yaml", "m.pt", 1, 1, "p", "r", {})
    time.sleep(0.1)
    assert ok1 and not ok2
    assert "学習" in why
    assert len(started) == 1


def test_探索中は学習を始めない_学習中は探索を始めない(clean_shared, monkeypatch):
    train, tune = clean_shared
    monkeypatch.setattr(tr, "_train_worker", lambda *a: None)
    from core import tuning

    train["running"], tune["running"] = False, True
    ok, why = tr.start_training("d.yaml", "m.pt", 1, 1, "p", "r", {})
    assert not ok and "探索" in why

    train["running"], tune["running"] = True, False
    assert tuning.start_tuning("d.yaml", "m.pt", 1, 1, None, "t") is False
    assert tune["running"] is False


# ---------------------------------------------------------------------------
# 実際に子プロセスで学習する（CPU・極小）
# ---------------------------------------------------------------------------
def _wait(train, timeout=300):
    t0 = time.time()
    while train["running"] and time.time() - t0 < timeout:
        time.sleep(0.5)
    assert not train["running"], "学習が終わらない"


@pytest.fixture
def cpu_env(monkeypatch, tmp_path):
    pytest.importorskip("ultralytics")
    monkeypatch.setenv("MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file:{tmp_path / 'mlruns'}")
    return tmp_path


def test_子プロセスで学習が完走する(clean_shared, cpu_env, detect_dataset):
    train, _ = clean_shared
    train["running"] = False
    ok, _ = tr.start_training(
        str(detect_dataset / "data.yaml"), "yolo11n.yaml", 1, 2, "test", "run_ok",
        {"imgsz": 32, "device": "cpu", "workers": 0, "plots": False})
    assert ok
    _wait(train)
    assert train["error"] is None, train["log"][-20:]
    assert train["progress"] == 100
    assert train["model_path"].endswith("run_ok/weights/best.pt")
    assert train["metrics_history"] and train["metrics_history"][0]["epoch"] == 1


def test_停止するとエポック末で止まる(clean_shared, cpu_env, detect_dataset):
    train, lock = _get_train_shared()
    train["running"] = False
    ok, _ = tr.start_training(
        str(detect_dataset / "data.yaml"), "yolo11n.yaml", 50, 2, "test", "run_stop",
        {"imgsz": 32, "device": "cpu", "workers": 0, "plots": False})
    assert ok
    with lock:
        train["stop_requested"] = True
    _wait(train)
    assert train["error"] is None, train["log"][-20:]
    assert len(train["metrics_history"]) < 50
    assert any("中断" in line for line in train["log"])
    # 止めたエポックの重みが残っている（= 続きから再開できる）
    last = cpu_env / "models" / "run_stop" / "weights" / "last.pt"
    assert last.exists()

    # 実際に続きから再開できる（再開もすぐ止める）
    stopped_at = len(train["metrics_history"])
    ok, _ = tr.start_training(str(detect_dataset / "data.yaml"), str(last), 0, 0,
                              "test", "run_stop", {"resume": True})
    assert ok
    with lock:
        train["stop_requested"] = True
    _wait(train)
    assert train["error"] is None, train["log"][-20:]
    assert train["metrics_history"][0]["epoch"] == stopped_at + 1


def test_失敗は画面に届く(clean_shared, cpu_env, tmp_path):
    train, _ = clean_shared
    train["running"] = False
    ok, _ = tr.start_training(
        str(tmp_path / "missing.yaml"), "yolo11n.yaml", 1, 2, "test", "run_ng",
        {"imgsz": 32, "device": "cpu", "workers": 0})
    assert ok
    _wait(train)
    assert train["error"]
