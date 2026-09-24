"""CVAT の非同期処理の待ち方（以前は 180 秒で打ち切っていた）"""
from __future__ import annotations

from core.cvat import wait_cvat_request


class _Clock:
    def __init__(self):
        self.t = 0.0
        self.sleeps = []

    def now(self):
        return self.t

    def sleep(self, s):
        self.sleeps.append(s)
        self.t += s


def _run(statuses, timeout=1800):
    it = iter(statuses)
    c = _Clock()
    last = {}

    def get():
        nonlocal last
        last = next(it, last)
        return last
    return wait_cvat_request(get, timeout, sleep=c.sleep, clock=c.now), c


def test_180秒を超える書き出しも待てる():
    # 5 分かかる書き出し（以前は 180 秒で打ち切られていた）
    statuses = [{"status": "started"}] * 80 + [{"status": "finished", "result_url": "u"}]
    res, c = _run(statuses)
    assert res["state"] == "finished"
    assert res["data"]["result_url"] == "u"
    assert c.t > 180


def test_間隔は5秒まで広げる():
    res, c = _run([{"status": "queued"}] * 20 + [{"status": "finished"}])
    assert c.sleeps[0] == 1.0
    assert max(c.sleeps) == 5.0


def test_失敗はすぐ返す():
    res, c = _run([{"status": "started"}, {"status": "failed", "message": "x"}])
    assert res["state"] == "failed" and res["data"]["message"] == "x"


def test_上限を超えたら打ち切る():
    res, c = _run([{"status": "started"}] * 10000, timeout=60)
    assert res["state"] == "timeout"
    assert 60 <= c.t < 70
