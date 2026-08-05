# =============================================================================
# コンテナ内 ↔ ホストのパス変換のテスト
#
#   決め打ちにすると別の場所に clone した人の環境で外れるので、
#   マウント表から求める。その読み替えが正しいことを確かめる。
# =============================================================================
from __future__ import annotations

import time
from pathlib import Path

import pytest

from core import hostpath as hp


@pytest.fixture
def mounts(monkeypatch):
    """マウント表を差し替える（docker に触らない）"""
    table = [
        ("/workspace/predictions", "/home/u/proj/predictions"),
        ("/workspace/extensions", "/home/u/proj/extensions"),
        ("/workspace/models", "/home/u/proj/models"),
        ("/workspace/data", "/home/u/proj/data"),
        ("/app", "/home/u/proj/app"),
    ]
    table.sort(key=lambda x: -len(x[0]))
    monkeypatch.setattr(hp, "_MOUNT_CACHE", table)
    return table


# ---------------------------------------------------------------------------
# 読み替え
# ---------------------------------------------------------------------------
def test_マウント先そのものを読み替える(mounts):
    assert hp.to_host_path("/workspace/data") == "/home/u/proj/data"


def test_配下のパスを読み替える(mounts):
    assert hp.to_host_path("/workspace/data/ds1") == "/home/u/proj/data/ds1"
    assert hp.to_host_path("/workspace/models/run/weights/best.pt") == \
        "/home/u/proj/models/run/weights/best.pt"


def test_長いマウントを優先する(monkeypatch):
    """/workspace より /workspace/data を先に当てること"""
    monkeypatch.setattr(hp, "_MOUNT_CACHE", sorted(
        [("/workspace", "/host/ws"), ("/workspace/data", "/other/place")],
        key=lambda x: -len(x[0])))
    assert hp.to_host_path("/workspace/data/x") == "/other/place/x"


def test_似た名前を取り違えない(mounts):
    """/workspace/dataset は /workspace/data の配下ではない"""
    assert hp.to_host_path("/workspace/dataset") is None


def test_対応が無ければNone(mounts):
    assert hp.to_host_path("/tmp/よそ") is None


def test_マウントが取れなければNone(monkeypatch):
    monkeypatch.setattr(hp, "_MOUNT_CACHE", [])
    assert hp.to_host_path("/workspace/data") is None
    assert hp.host_path_available() is False


def test_コマンド文字列(mounts):
    assert hp.open_command("/workspace/data") == 'xdg-open "/home/u/proj/data"'


def test_対応が無ければコマンドも空(mounts):
    assert hp.open_command("/tmp/よそ") == ""


# ---------------------------------------------------------------------------
# ウォッチャーへの依頼
# ---------------------------------------------------------------------------
@pytest.fixture
def reqdir(tmp_path, monkeypatch):
    d = tmp_path / ".open_requests"
    monkeypatch.setattr(hp, "REQUEST_DIR", d)
    monkeypatch.setattr(hp, "HEARTBEAT", d / ".watcher_alive")
    return d


def test_心拍が無ければ動いていない(reqdir):
    assert hp.watcher_running() is False


def test_心拍が新しければ動いている(reqdir):
    reqdir.mkdir(parents=True)
    (reqdir / ".watcher_alive").touch()
    assert hp.watcher_running() is True


def test_心拍が古ければ動いていない(reqdir):
    import os
    reqdir.mkdir(parents=True)
    hb = reqdir / ".watcher_alive"
    hb.touch()
    old = time.time() - hp.HEARTBEAT_TIMEOUT - 10
    os.utime(hb, (old, old))
    assert hp.watcher_running() is False


def test_動いていなければ依頼を置かない(mounts, reqdir):
    """溜め込んでも意味がなく、あとから一斉に開いても困る"""
    res = hp.request_open("/workspace/data")
    assert not res["ok"]
    assert "ウォッチャー" in res["error"]
    assert res["host_path"] == "/home/u/proj/data", "パスは返すこと"
    assert not list(reqdir.glob("open_*.txt")) if reqdir.exists() else True


def test_動いていれば依頼を置く(mounts, reqdir):
    reqdir.mkdir(parents=True)
    (reqdir / ".watcher_alive").touch()
    res = hp.request_open("/workspace/data/ds1")
    assert res["ok"], res["error"]

    files = list(reqdir.glob("open_*.txt"))
    assert len(files) == 1
    assert files[0].read_text().strip() == "/home/u/proj/data/ds1"


def test_対応の無いパスは依頼できない(mounts, reqdir):
    reqdir.mkdir(parents=True)
    (reqdir / ".watcher_alive").touch()
    res = hp.request_open("/tmp/よそ")
    assert not res["ok"] and res["host_path"] is None
