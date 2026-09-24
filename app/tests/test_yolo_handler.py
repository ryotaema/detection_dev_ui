"""自作 YOLO の Nuclio ハンドラ（serverless/_common/model_handler.py）

obb / classify のモデルをデプロイすると、`result.boxes` が無いため
結果が空になっていた。タスクごとの返り値の形を確かめる。
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from core.config import SERVERLESS_DIR

_CANDIDATES = [
    SERVERLESS_DIR / "_common" / "model_handler.py",
    Path(__file__).resolve().parent.parent.parent / "serverless" / "_common" / "model_handler.py",
]
_SRC = next((p for p in _CANDIDATES if p.exists()), _CANDIDATES[0])


@pytest.fixture(scope="module")
def mh():
    if not _SRC.exists():
        pytest.skip("serverless/_common/model_handler.py が無い")
    # 関数コンテナ用のファイルなので、ultralytics が無くても読めるように差し替える
    fake = types.ModuleType("ultralytics")
    fake.YOLO = object
    saved = sys.modules.get("ultralytics")
    sys.modules["ultralytics"] = fake
    try:
        spec = importlib.util.spec_from_file_location("_yolo_model_handler", _SRC)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)          # type: ignore[union-attr]
    finally:
        if saved is None:
            sys.modules.pop("ultralytics", None)
        else:
            sys.modules["ultralytics"] = saved
    return mod


class _T:
    """tensor の代わり（tolist / item だけ使う）"""
    def __init__(self, v):
        self.v = np.asarray(v, dtype=float)

    def tolist(self):
        return self.v.tolist()

    def item(self):
        return float(self.v.reshape(-1)[0])

    def __getitem__(self, i):
        return _T(self.v[i])


def _result(**kw):
    base = {"probs": None, "obb": None, "boxes": None, "masks": None}
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_detectは矩形を返す(mh):
    box = types.SimpleNamespace(cls=_T([1]), conf=_T([0.9]), xyxy=_T([[1, 2, 3, 4]]))
    out = mh.to_cvat(_result(boxes=[box]), {0: "a", 1: "b"}, 0.5)
    assert out == [{"confidence": "0.9", "label": "b", "points": [1, 2, 3, 4],
                    "type": "rectangle"}]


def test_obbは4点ポリゴンを返す(mh):
    obb = types.SimpleNamespace(
        cls=_T([0]), conf=_T([0.8]),
        xyxyxyxy=_T([[[0, 0], [10, 0], [10, 5], [0, 5]]]))
    out = mh.to_cvat(_result(obb=obb), ["car"], 0.5)
    assert out[0]["type"] == "polygon"
    assert out[0]["points"] == [0, 0, 10, 0, 10, 5, 0, 5]
    assert out[0]["label"] == "car"


def test_classifyはしきい値以上ならタグを返す(mh):
    probs = types.SimpleNamespace(top1=2, top1conf=0.7)
    out = mh.to_cvat(_result(probs=probs), ["a", "b", "c"], 0.5)
    assert out == [{"confidence": "0.7", "label": "c", "type": "tag"}]
    assert mh.to_cvat(_result(probs=probs), ["a", "b", "c"], 0.9) == []
