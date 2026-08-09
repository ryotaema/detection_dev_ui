"""SAM 3 の Nuclio ハンドラ（serverless/sam/model_handler.py）

このファイルは関数コンテナの中でしか動かないが、CVAT への返し方を間違えると
「動いているのにアノテーションが出ない」という分かりにくい壊れ方をする。
外部依存の無い部分だけをここで固定する（ultralytics は関数内 import にしてあるので、
モジュールの読み込みだけなら重みも GPU も要らない）。
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from core.config import SERVERLESS_DIR

# コンテナ内 (/workspace/serverless) とホスト (<repo>/serverless) の両方で見つける
_CANDIDATES = [
    SERVERLESS_DIR / "sam" / "model_handler.py",
    Path(__file__).resolve().parent.parent.parent / "serverless" / "sam" / "model_handler.py",
]
_SRC = next((p for p in _CANDIDATES if p.exists()), _CANDIDATES[0])


@pytest.fixture(scope="module")
def mh():
    if not _SRC.exists():
        pytest.skip("serverless/sam/model_handler.py が無い")
    spec = importlib.util.spec_from_file_location("_sam3_model_handler", _SRC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)          # type: ignore[union-attr]
    return mod


def test_マスクは_CVAT_の_RLE_形式で返る(mh):
    """CVAT の mask シェイプは [RLE..., left, top, right, bottom]。
    RLE は 0 の並びから始まる決まりなので、先頭が 1 のときは 0 を挿す。"""
    mask = np.zeros((2, 3), dtype=np.uint8)
    mask[0, 0] = 1                        # 先頭画素が 1

    rle = mh.mask_to_rle(mask)

    assert rle[-4:] == [0, 0, 2, 1]       # 外接矩形 (width-1, height-1)
    assert rle[0] == 0                    # 0 の並びの長さから始まる
    assert sum(rle[:-4]) == mask.size     # 全画素を覆う


def test_全部ゼロのマスクも壊れない(mh):
    rle = mh.mask_to_rle(np.zeros((4, 4), dtype=np.uint8))
    assert sum(rle[:-4]) == 16


def test_プロンプトはラベル名と分けて持てる(mh):
    """SAM 3 は英語の名詞句が前提。日本語のラベル名をそのまま渡すと何も出ない。"""
    pairs = mh.parse_prompt_map(
        '[{"label": "猫", "prompt": "cat"}]', ["猫", "object_a"])
    # 対応表に無いラベルはラベル名をそのままプロンプトにする
    assert pairs == [("猫", "cat"), ("object_a", "object_a")]


def test_プロンプトが壊れていてもラベル名で動く(mh):
    """env が壊れていても「何も検出しない」より「ラベル名で試す」ほうがまし。"""
    assert mh.parse_prompt_map("{壊れた", ["a"]) == [("a", "a")]
    assert mh.parse_prompt_map("", ["a"]) == [("a", "a")]


def test_画像は_BGR_で渡す(mh):
    """Ultralytics の preprocess は OpenCV 由来の BGR を前提にしている。
    RGB のまま渡すと色が入れ替わったまま推論される。"""
    from PIL import Image

    img = Image.new("RGB", (2, 2), (255, 0, 0))      # 赤
    arr = mh.to_bgr(img)
    assert arr.shape == (2, 2, 3)
    assert tuple(arr[0, 0]) == (0, 0, 255)           # BGR では赤が末尾


def test_点は_1_オブジェクトとしてまとめて渡す(mh):
    """(N, 2) を渡すと Predictor._prepare_prompts が (N, 1, 2) に直し、
    点の数だけ別オブジェクトになる。欲しいのは「N 個の点で指した 1 個」なので、
    ハンドラ側で (1, N, 2) にしてから渡すこと。"""
    text = _SRC.read_text(encoding="utf-8")
    assert 'kwargs["points"] = [pos + neg]' in text, "点を 1 オブジェクトにまとめていない"
    assert 'kwargs["labels"] = [[1] * len(pos) + [0] * len(neg)]' in text


def test_CVAT_の_obj_bbox_は点のペアで来る(mh):
    """**CVAT UI は `[[x1,y1],[x2,y2]]` で送ってくる。**
    フラットな `[x1,y1,x2,y2]` を前提にすると float() に list を渡して落ち、
    CVAT 側には 500 としてしか見えない（実際に踏んだ）。"""
    assert mh.flatten_bbox([[10, 20], [110, 220]]) == [10, 20, 110, 220]
    assert mh.flatten_bbox([10, 20, 110, 220]) == [10, 20, 110, 220]
    # 描いた向きによって左上・右下が逆に来ても外接矩形にまとめる
    assert mh.flatten_bbox([[110, 220], [10, 20]]) == [10, 20, 110, 220]


def test_obj_bbox_が無い_足りないときは使わない(mh):
    assert mh.flatten_bbox(None) is None
    assert mh.flatten_bbox([]) is None
    assert mh.flatten_bbox([[1, 2]]) is None        # 点が 1 つでは矩形にならない
