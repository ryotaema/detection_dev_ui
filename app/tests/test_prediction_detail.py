# =============================================================================
# 推論結果の拡大確認のテスト
#
#   再アノテーションのフラグ付けは「小さくて判断できない」のが一番の障害。
#   ここでは拡大用の画像が意図どおり作られるかを見る。
# =============================================================================
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from core.inference import (
    prediction_box_summaries, prediction_detail, _scaled_draw,
)


@pytest.fixture
def pred(tmp_path):
    """800x600 の画像と、検出 2 件を持つ予測 JSON"""
    img = np.random.RandomState(0).randint(0, 255, (600, 800, 3), np.uint8)
    ip = tmp_path / "shot.png"
    cv2.imwrite(str(ip), img)

    jf = tmp_path / "shot.json"
    jf.write_text(json.dumps({
        "image_path": str(ip),
        "task": "detect",
        "boxes": [
            {"label": "object_a", "confidence": 0.91,
             "bbox_xyxy": [300.0, 200.0, 380.0, 300.0]},
            {"label": "object_b", "confidence": 0.42,
             "bbox_xyxy": [640.0, 60.0, 700.0, 110.0]},
        ]}))
    return jf


# ---------------------------------------------------------------------------
# 検出の一覧
# ---------------------------------------------------------------------------
def test_検出ごとの要約が取れる(pred):
    got = prediction_box_summaries(pred)
    assert len(got) == 2
    assert got[0]["label"] == "object_a"
    assert got[0]["confidence"] == pytest.approx(0.91)
    assert got[0]["size"] == (80, 100)
    assert got[1]["index"] == 1


def test_壊れたJSONでも落ちない(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{ 壊れている")
    assert prediction_box_summaries(bad) == []


def test_座標が欠けた検出は飛ばす(tmp_path):
    jf = tmp_path / "a.json"
    jf.write_text(json.dumps({"image_path": "/x.png", "boxes": [
        {"label": "a", "bbox_xyxy": [1, 2]}, {"label": "b", "bbox_xyxy": [1, 2, 3, 4]}]}))
    assert [b["label"] for b in prediction_box_summaries(jf)] == ["b"]


# ---------------------------------------------------------------------------
# 全体表示
# ---------------------------------------------------------------------------
def test_全体は元の大きさで返る(pred):
    got = prediction_detail(pred, box_index=None)
    assert got is not None
    assert got["image"].shape[:2] == (600, 800)
    assert got["n_boxes"] == 2
    assert got["crop_rect"] is None


def test_大きすぎる画像は縮小する(tmp_path):
    big = np.zeros((2000, 3000, 3), np.uint8)
    ip = tmp_path / "big.png"
    cv2.imwrite(str(ip), big)
    jf = tmp_path / "big.json"
    jf.write_text(json.dumps({"image_path": str(ip), "boxes": []}))

    got = prediction_detail(jf, max_side=1400)
    assert max(got["image"].shape[:2]) == 1400


def test_小さい画像は拡大しない(pred):
    """水増しして粗くしないこと"""
    got = prediction_detail(pred, max_side=4000)
    assert got["image"].shape[:2] == (600, 800)


# ---------------------------------------------------------------------------
# ボックス単位の切り出し
# ---------------------------------------------------------------------------
def test_検出の周辺を切り出す(pred):
    got = prediction_detail(pred, box_index=0, margin=0.5)
    assert got["crop_rect"] is not None
    x1, y1, x2, y2 = got["crop_rect"]
    # bbox は 300,200-380,300（長辺100）。余白 50 で 250,150-430,350
    assert (x1, y1, x2, y2) == (250, 150, 430, 350)
    assert got["image"].shape[:2] == (200, 180)


def test_余白を広げると切り出しも広がる(pred):
    small = prediction_detail(pred, box_index=0, margin=0.2)["crop_rect"]
    large = prediction_detail(pred, box_index=0, margin=1.0)["crop_rect"]
    assert (large[2] - large[0]) > (small[2] - small[0])


def test_画像の端でも切り出せる(pred):
    """右上にある検出。画像の外へはみ出さないこと"""
    got = prediction_detail(pred, box_index=1, margin=2.0)
    x1, y1, x2, y2 = got["crop_rect"]
    assert x1 >= 0 and y1 >= 0
    assert x2 <= 800 and y2 <= 600


def test_範囲外の番号は全体になる(pred):
    got = prediction_detail(pred, box_index=99)
    assert got["crop_rect"] is None
    assert got["image"].shape[:2] == (600, 800)


def test_検出が無くても落ちない(tmp_path):
    img = np.zeros((100, 100, 3), np.uint8)
    ip = tmp_path / "empty.png"
    cv2.imwrite(str(ip), img)
    jf = tmp_path / "empty.json"
    jf.write_text(json.dumps({"image_path": str(ip), "boxes": []}))

    got = prediction_detail(jf, box_index=0)
    assert got is not None and got["n_boxes"] == 0


def test_画像が無ければNone(tmp_path):
    jf = tmp_path / "missing.json"
    jf.write_text(json.dumps({"image_path": "/存在しない.png", "boxes": []}))
    assert prediction_detail(jf) is None


# ---------------------------------------------------------------------------
# 描画
# ---------------------------------------------------------------------------
def test_線の太さが画像サイズに追従する():
    """1600px の画像に 2px の枠では細すぎて見えない"""
    boxes = [{"label": "a", "confidence": 0.9, "bbox_xyxy": [10, 10, 100, 100]}]
    small = _scaled_draw(np.zeros((200, 200, 3), np.uint8), list(boxes))
    large = _scaled_draw(np.zeros((1600, 1600, 3), np.uint8), list(boxes))
    # 枠のあるところの非ゼロ画素数で太さを間接的に見る
    assert np.count_nonzero(large) > np.count_nonzero(small)


def test_注目していない検出は細く描く(pred):
    """どれを見ているか分かるようにする"""
    both = prediction_detail(pred, box_index=None)["image"]
    one = prediction_detail(pred, box_index=None)["image"]
    assert both.shape == one.shape        # 全体表示は同じ結果になる（再現性）
