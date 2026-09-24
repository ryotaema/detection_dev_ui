# =============================================================================
# 学習まわりのテスト（GPU・Ultralytics を使わない範囲）
# =============================================================================
from __future__ import annotations

import sys
import types

from core.training import default_train_device


def _fake_torch(monkeypatch, available: bool, count: int):
    cuda = types.SimpleNamespace(is_available=lambda: available,
                                 device_count=lambda: count)
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=cuda))


def test_GPUがあれば0番を使う(monkeypatch):
    _fake_torch(monkeypatch, True, 1)
    assert default_train_device() == 0


def test_GPUが無ければcpuにする(monkeypatch):
    """device=0 決め打ちだと CPU 構成で学習が必ず失敗していた"""
    _fake_torch(monkeypatch, False, 0)
    assert default_train_device() == "cpu"


def test_torchが無くてもcpuにする(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)   # import すると ImportError
    assert default_train_device() == "cpu"
