# =============================================================================
# クロップ生成のテスト
#
#   make_crop() は実機と共通のコア。ここがずれると学習と実機で
#   切り出し規則が食い違い、精度が壊れる。
#   特に「マスクを元画像へ戻せること」（座標変換の往復）を重点的に見る。
# =============================================================================
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from core.crop import (
    crop_to_source, dedup_detections, generate_background_tiles,
    generate_crops, make_crop, read_manifest, source_to_crop,
    target_rect_in_crop,
)


def _img(w=800, h=600):
    a = np.zeros((h, w, 3), np.uint8)
    a[:, :, 1] = 80
    # 位置が分かるよう格子を入れる
    a[::50, :] = 255
    a[:, ::50] = 255
    return a


@pytest.fixture
def src(tmp_path):
    """検出 2 件を持つ元画像ディレクトリ"""
    d = tmp_path / "raw"
    d.mkdir()
    for i in range(2):
        cv2.imwrite(str(d / f"IMG_{i}.png"), _img())
    return d


def _dets(src):
    return {
        str(src / "IMG_0.png"): [
            {"bbox_xyxy": [300, 200, 400, 300], "confidence": 0.9,
             "label": "bell_pepper", "class_id": 0},
            {"bbox_xyxy": [600, 100, 660, 160], "confidence": 0.5,
             "label": "bell_pepper", "class_id": 0},
        ],
        str(src / "IMG_1.png"): [
            {"bbox_xyxy": [50, 50, 150, 130], "confidence": 0.8,
             "label": "bell_pepper", "class_id": 0},
        ],
    }


# ---------------------------------------------------------------------------
# コア: make_crop
# ---------------------------------------------------------------------------
def test_倍率どおりの大きさで切り出す():
    img = _img()
    # bbox 100x100、倍率 2.0 → 200x200 を切り出す
    _, g = make_crop(img, [300, 200, 400, 300], scale=2.0, out_size=400)
    assert g["crop_rect_in_source"] == [250, 150, 200, 200]


def test_正方でない切り出しもできる():
    img = _img()
    _, g = make_crop(img, [300, 200, 400, 300], scale=2.0, square=False,
                     out_size=400)
    _, _, w, h = g["crop_rect_in_source"]
    assert (w, h) == (200, 200)      # bbox が正方なので同じ

    _, g2 = make_crop(img, [300, 200, 400, 250], scale=2.0, square=False,
                      out_size=400)
    _, _, w2, h2 = g2["crop_rect_in_source"]
    assert w2 == 200 and h2 == 100    # 縦横それぞれ 2 倍


def test_対角基準だと大きくなる():
    img = _img()
    _, a = make_crop(img, [300, 200, 400, 300], scale=2.0,
                     scale_basis="long_side", out_size=400)
    _, b = make_crop(img, [300, 200, 400, 300], scale=2.0,
                     scale_basis="diagonal", out_size=400)
    assert b["crop_rect_in_source"][2] > a["crop_rect_in_source"][2]


def test_中心がbboxの中心と一致する():
    img = _img()
    _, g = make_crop(img, [300, 200, 400, 300], scale=2.0, out_size=400)
    x0, y0, w, h = g["crop_rect_in_source"]
    assert (x0 + w / 2, y0 + h / 2) == (350, 250)


def test_画像の端でも欠けない():
    """左上にある対象。パディングして正方を保つ"""
    img = _img()
    crop, g = make_crop(img, [10, 10, 50, 50], scale=3.0, out_size=240)
    assert g["padding"]["left"] > 0 and g["padding"]["top"] > 0
    # 出力は正方
    assert crop.shape[0] == crop.shape[1]


def test_はみ出しが大きいと単色で埋める():
    """reflect は元領域より広くは反射できないので落ちないこと"""
    img = _img(100, 100)
    crop, g = make_crop(img, [0, 0, 10, 10], scale=20.0,
                        pad_mode="reflect", out_size=200)
    assert crop.shape[0] == crop.shape[1]
    assert g["padding"]["left"] > 0


def test_出力サイズにそろう():
    img = _img()
    crop, g = make_crop(img, [300, 200, 400, 300], scale=2.0,
                        out_size=512, max_upscale=10.0)
    assert max(crop.shape[:2]) == 512
    assert g["output_size"] == [512, 512]


def test_拡大しすぎない():
    """低解像度由来の小さな対象を 1024 へ無理に引き伸ばさない"""
    img = _img(640, 480)
    crop, g = make_crop(img, [300, 200, 340, 240], scale=2.0,
                        out_size=1024, max_upscale=1.5)
    assert g["max_upscale_applied"] is True
    # 切り出しは 80px、上限 1.5 倍なので 120px 止まり
    assert max(crop.shape[:2]) == 120
    assert g["resize_ratio"] == pytest.approx(1.5)


def test_上限内なら適用フラグは立たない():
    img = _img()
    _, g = make_crop(img, [100, 100, 500, 500], scale=2.0,
                     out_size=400, max_upscale=1.5)
    assert g["max_upscale_applied"] is False


def test_縮小のときは補間を変える():
    """大きい切り出しを縮めても落ちないこと"""
    img = _img(2000, 2000)
    crop, g = make_crop(img, [500, 500, 1500, 1500], scale=1.5, out_size=512)
    assert max(crop.shape[:2]) == 512
    assert g["resize_ratio"] < 1.0


# ---------------------------------------------------------------------------
# 座標変換（マスクを元画像へ戻せること）
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("scale,out_size,max_up", [
    (2.0, 512, 10.0), (1.5, 1024, 10.0), (3.0, 256, 10.0), (2.0, 1024, 1.5),
])
def test_往復して元の座標に戻る(scale, out_size, max_up):
    img = _img()
    _, g = make_crop(img, [300, 200, 400, 300], scale=scale,
                     out_size=out_size, max_upscale=max_up)
    for sx, sy in [(300.0, 200.0), (350.0, 250.0), (400.0, 300.0)]:
        cx, cy = source_to_crop(sx, sy, g)
        bx, by = crop_to_source(cx, cy, g)
        assert bx == pytest.approx(sx, abs=1e-6)
        assert by == pytest.approx(sy, abs=1e-6)


def test_bboxの中心はクロップの中心へ写る():
    img = _img()
    _, g = make_crop(img, [300, 200, 400, 300], scale=2.0,
                     out_size=400, max_upscale=10.0)
    cx, cy = source_to_crop(350.0, 250.0, g)
    ow, oh = g["output_size"]
    assert cx == pytest.approx(ow / 2, abs=1.0)
    assert cy == pytest.approx(oh / 2, abs=1.0)


def test_端の対象でも往復する():
    """パディングが入っても座標系が狂わないこと"""
    img = _img()
    _, g = make_crop(img, [10, 10, 50, 50], scale=3.0, out_size=240,
                     max_upscale=10.0)
    cx, cy = source_to_crop(30.0, 30.0, g)
    bx, by = crop_to_source(cx, cy, g)
    assert (bx, by) == pytest.approx((30.0, 30.0))


def test_内側target矩形は倍率どおり():
    img = _img()
    _, g = make_crop(img, [300, 200, 400, 300], scale=2.0,
                     out_size=400, max_upscale=10.0)
    # annotation 2.0 に対し target 1.5 → 出力の 3/4
    x, y, w, h = target_rect_in_crop(g, 1.5)
    assert w == pytest.approx(300, abs=1) and h == pytest.approx(300, abs=1)
    assert x == pytest.approx(50, abs=1) and y == pytest.approx(50, abs=1)


def test_target倍率が同じなら全体になる():
    img = _img()
    _, g = make_crop(img, [300, 200, 400, 300], scale=2.0,
                     out_size=400, max_upscale=10.0)
    x, y, w, h = target_rect_in_crop(g, 2.0)
    assert (x, y) == pytest.approx((0, 0), abs=1)
    assert (w, h) == pytest.approx((400, 400), abs=1)


# ---------------------------------------------------------------------------
# 重複除去
# ---------------------------------------------------------------------------
def test_既定では間引かない():
    dets = [{"bbox_xyxy": [0, 0, 10, 10], "confidence": 0.9},
            {"bbox_xyxy": [1, 1, 11, 11], "confidence": 0.8}]
    kept, dropped = dedup_detections(dets, 0.0)
    assert len(kept) == 2 and dropped == []


def test_近すぎるものを間引く():
    dets = [{"bbox_xyxy": [0, 0, 100, 100], "confidence": 0.9},
            {"bbox_xyxy": [5, 5, 105, 105], "confidence": 0.6},
            {"bbox_xyxy": [500, 500, 600, 600], "confidence": 0.7}]
    kept, dropped = dedup_detections(dets, 0.5)
    assert len(kept) == 2 and len(dropped) == 1
    # 信頼度の高いほうが残る
    assert dropped[0]["confidence"] == 0.6


def test_間引きは決定的():
    dets = [{"bbox_xyxy": [0, 0, 100, 100], "confidence": 0.5},
            {"bbox_xyxy": [5, 5, 105, 105], "confidence": 0.5}]
    a, _ = dedup_detections(list(dets), 0.5)
    b, _ = dedup_detections(list(reversed(dets)), 0.5)
    assert a == b, "順序で結果が変わっている"


# ---------------------------------------------------------------------------
# アノテーション用の書き出し
# ---------------------------------------------------------------------------
def test_対象ごとに1枚できる(src, tmp_path):
    out = tmp_path / "crops"
    res = generate_crops(_dets(src), out, out_size=256, max_upscale=10.0)
    assert res["ok"], res["error"]
    assert res["crops"] == 3 and res["images"] == 2

    imgs = sorted(p.name for p in (out / "images").iterdir())
    assert imgs == ["IMG_0_obj00.png", "IMG_0_obj01.png", "IMG_1_obj00.png"]
    assert len(list((out / "meta").iterdir())) == 3


def test_サイドカーに復元用の情報が入る(src, tmp_path):
    out = tmp_path / "crops"
    generate_crops(_dets(src), out, annotation_scale=2.0, target_scale=1.5,
                   out_size=256, max_upscale=10.0,
                   model_info={"model_id": "fruit_v3", "infer_input_size": 640})
    meta = json.loads((out / "meta" / "IMG_0_obj00.json").read_text())

    assert meta["data_type"] == "object_crop"
    assert meta["annotation_status"] == "raw"
    assert meta["source_image"]["width"] == 800
    assert meta["source_image"]["sha1"]
    assert meta["bbox_model"]["model_id"] == "fruit_v3"
    assert meta["target_object"]["confidence"] == 0.9
    g = meta["crop_geometry"]
    for k in ("crop_rect_in_source", "padding", "resize_ratio",
              "target_rect_in_crop", "annotation_scale", "target_scale"):
        assert k in g, k


def test_同じクロップに写る他の対象を記録する(src, tmp_path):
    """密集した対象。どれが主対象かの判別に使う"""
    dets = {str(src / "IMG_0.png"): [
        {"bbox_xyxy": [300, 300, 360, 360], "confidence": 0.9},
        {"bbox_xyxy": [380, 300, 440, 360], "confidence": 0.8},
    ]}
    out = tmp_path / "crops"
    generate_crops(dets, out, annotation_scale=3.0, out_size=256,
                   max_upscale=10.0)
    meta = json.loads((out / "meta" / "IMG_0_obj00.json").read_text())
    assert len(meta["others_in_crop"]) == 1


def test_離れた対象は写り込みに入れない(src, tmp_path):
    dets = {str(src / "IMG_0.png"): [
        {"bbox_xyxy": [10, 10, 50, 50], "confidence": 0.9},
        {"bbox_xyxy": [700, 500, 760, 560], "confidence": 0.8},
    ]}
    out = tmp_path / "crops"
    generate_crops(dets, out, annotation_scale=2.0, out_size=256,
                   max_upscale=10.0)
    meta = json.loads((out / "meta" / "IMG_0_obj00.json").read_text())
    assert meta["others_in_crop"] == []


def test_マニフェストが1クロップ1行(src, tmp_path):
    out = tmp_path / "crops"
    generate_crops(_dets(src), out, out_size=256, max_upscale=10.0)
    rows = read_manifest(out)
    assert len(rows) == 3
    for r in rows:
        for k in ("crop_image", "meta", "data_type", "annotation_status",
                  "source_image", "confidence"):
            assert k in r, k


def test_出力先が既にあれば中断する(src, tmp_path):
    out = tmp_path / "crops"
    out.mkdir()
    (out / "既存.txt").write_text("消されたら困る")
    res = generate_crops(_dets(src), out)
    assert not res["ok"] and "既にあります" in res["error"]
    assert (out / "既存.txt").exists()


def test_検出が無ければ何もしない(tmp_path):
    res = generate_crops({}, tmp_path / "crops")
    assert not res["ok"] and "対象となる検出がありません" in res["error"]


def test_読めない画像は飛ばして続ける(src, tmp_path):
    bad = src / "broken.png"
    bad.write_text("画像ではない")
    dets = {**_dets(src), str(bad): [{"bbox_xyxy": [1, 1, 5, 5],
                                      "confidence": 0.9}]}
    out = tmp_path / "crops"
    res = generate_crops(dets, out, out_size=256, max_upscale=10.0)
    assert res["crops"] == 3
    assert any("読めません" in why for _, why in res["skipped"])


def test_jpgでも書ける(src, tmp_path):
    out = tmp_path / "crops"
    res = generate_crops(_dets(src), out, out_format="jpg", jpg_quality=90,
                         out_size=256, max_upscale=10.0)
    assert res["ok"]
    assert all(p.suffix == ".jpg" for p in (out / "images").iterdir())


def test_元画像には触らない(src, tmp_path):
    before = {p.name: p.read_bytes() for p in src.glob("*.png")}
    generate_crops(_dets(src), tmp_path / "crops", out_size=256,
                   max_upscale=10.0)
    for p in src.glob("*.png"):
        assert p.read_bytes() == before[p.name]


# ---------------------------------------------------------------------------
# 背景タイル
# ---------------------------------------------------------------------------
def test_背景をタイルに分ける(src, tmp_path):
    out = tmp_path / "bg"
    res = generate_background_tiles(sorted(src.glob("*.png")), out,
                                    tile_size=400)
    assert res["ok"]
    # 800x600 を 400 で割ると 2x2 = 4 枚（端の細切れは捨てる）
    assert res["tiles"] == 8       # 画像 2 枚 × 4
    assert len(list((out / "meta").iterdir())) == 8


def test_タイルのメタに位置が入る(src, tmp_path):
    out = tmp_path / "bg"
    generate_background_tiles(sorted(src.glob("*.png"))[:1], out, tile_size=400)
    meta = json.loads((out / "meta" / "IMG_0_tile_r0_c1.json").read_text())
    assert meta["data_type"] == "background_tile"
    assert meta["tile"]["x"] == 400 and meta["tile"]["y"] == 0


def test_重なりを付けるとタイルが増える(src, tmp_path):
    a = generate_background_tiles(sorted(src.glob("*.png"))[:1],
                                  tmp_path / "a", tile_size=400)
    b = generate_background_tiles(sorted(src.glob("*.png"))[:1],
                                  tmp_path / "b", tile_size=400,
                                  tile_overlap=0.5)
    assert b["tiles"] > a["tiles"]


def test_空タイルを勝手に捨てない(tmp_path):
    """枝葉の判定は誤りやすいので、採否は人が決める"""
    d = tmp_path / "raw"
    d.mkdir()
    cv2.imwrite(str(d / "blank.png"), np.zeros((800, 800, 3), np.uint8))
    res = generate_background_tiles([d / "blank.png"], tmp_path / "bg",
                                    tile_size=400)
    assert res["tiles"] == 4, "真っ黒でも捨てないこと"


# ---------------------------------------------------------------------------
# 出力サイズを優先する（max_upscale <= 0 = 上限なし）
#
#   学習で使うサイズをそろえたい、という要望に応える経路。
# ---------------------------------------------------------------------------
def test_上限なしなら必ず出力サイズになる():
    img = _img(640, 480)
    crop, g = make_crop(img, [300, 200, 340, 240], scale=2.0,
                        out_size=1024, max_upscale=0.0)
    assert max(crop.shape[:2]) == 1024
    assert g["max_upscale_applied"] is False, "上限なしなのにフラグが立っている"


@pytest.mark.parametrize("scale", [1.5, 2.0, 3.0])
def test_上限なしなら倍率によらず同じサイズ(scale):
    """学習の入力サイズをそろえられること"""
    img = _img(640, 480)
    crop, _ = make_crop(img, [300, 200, 340, 240], scale=scale,
                        out_size=512, max_upscale=0.0)
    assert max(crop.shape[:2]) == 512


def test_上限なしでも座標は往復する():
    img = _img(640, 480)
    _, g = make_crop(img, [300, 200, 340, 240], scale=2.0,
                     out_size=1024, max_upscale=0.0)
    cx, cy = source_to_crop(320.0, 220.0, g)
    bx, by = crop_to_source(cx, cy, g)
    assert (bx, by) == pytest.approx((320.0, 220.0))


def test_倍率を変えると切り出し範囲が変わる():
    """UI で annotation_scale を変えたら結果が変わること"""
    img = _img()
    sizes = []
    for s in (1.5, 2.0, 3.0):
        _, g = make_crop(img, [300, 200, 400, 300], scale=s,
                         out_size=512, max_upscale=0.0)
        sizes.append(g["crop_rect_in_source"][2])
    assert sizes == sorted(sizes) and len(set(sizes)) == 3, sizes


def test_ファイル名の接頭辞を変えられる(src, tmp_path):
    """対象に合わせて名前を変えられること（既定は汎用の obj）"""
    out = tmp_path / "crops"
    generate_crops(_dets(src), out, out_size=256, max_upscale=10.0,
                   name_prefix="fruit")
    assert (out / "images" / "IMG_0_fruit00.png").exists()
    assert (out / "meta" / "IMG_0_fruit00.json").exists()


def test_切り出しサイズの既定はUIと実機で揃っている():
    """UI が 512 で作った学習データに対し、実機が 1024 で切り出すと
    対象の写る大きさが変わって精度が落ちる。既定値は 1 か所に持つこと。"""
    import inspect

    from core.crop import DEFAULT_OUT_SIZE, generate_crops, make_crop

    for fn in (make_crop, generate_crops):
        assert inspect.signature(fn).parameters["out_size"].default == DEFAULT_OUT_SIZE, \
            f"{fn.__name__} の既定が DEFAULT_OUT_SIZE とずれている"

    # UI の初期値とも一致していること
    src = (inspect.getsourcefile(make_crop) or "")
    ui = __import__("pathlib").Path(src).parent.parent / "ui" / "tab_crop.py"
    text = ui.read_text(encoding="utf-8")
    assert f'"cr_out": {DEFAULT_OUT_SIZE}' in text, \
        "UI の初期値 (cr_out) が DEFAULT_OUT_SIZE と違う"


# ---------------------------------------------------------------------------
# 仕様 §5.3 bbox_model / §8 debug_overlay
# ---------------------------------------------------------------------------
def test_モデルの素性に重みのハッシュが入る():
    """モデルを更新しても `models/<run>/weights/best.pt` というパスは変わらない。
    パスだけでは「どの時点の重みで作ったクロップか」を区別できない。"""
    import tempfile
    from pathlib import Path

    from core.crop import build_model_info

    with tempfile.TemporaryDirectory() as d:
        w = Path(d) / "myrun" / "weights" / "best.pt"
        w.parent.mkdir(parents=True)
        w.write_bytes(b"weights-v1")
        info = build_model_info(w, infer_input_size=640, conf_threshold=0.3)

        # 仕様が求める項目
        for k in ("model_id", "weights_sha1", "infer_input_size", "inferred_at"):
            assert info.get(k), f"{k} が無い"
        # best.pt ではなく run 名で識別できること
        assert info["model_id"] == "myrun"
        assert info["conf_threshold"] == 0.3

        before = info["weights_sha1"]
        w.write_bytes(b"weights-v2")           # 重みを差し替える
        assert build_model_info(w)["weights_sha1"] != before, \
            "重みを変えてもハッシュが変わらない"


def test_確認画像は全体に散らして上限内に収まる():
    """先頭から順に出すと 1 枚目の元画像に偏る。総数から確率を決めること。"""
    import shutil
    import tempfile
    from pathlib import Path

    import cv2
    import numpy as np

    from core.crop import generate_crops

    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "src"
        src.mkdir()
        det = {}
        for k in range(20):
            p = src / f"i{k}.png"
            cv2.imwrite(str(p), np.zeros((480, 640, 3), np.uint8))
            det[str(p)] = [{"bbox_xyxy": [50 + j * 60, 100, 110 + j * 60, 160],
                            "confidence": 0.8, "label": "o", "class_id": 0}
                           for j in range(4)]

        out = Path(d) / "out"
        shutil.rmtree(out, ignore_errors=True)
        r = generate_crops(det, out, debug_overlay=True, debug_samples=10, seed=0)

        assert r["crops"] == 80
        assert 0 < r["debug_images"] <= 10, r["debug_images"]
        assert (out / "debug").is_dir()
        # 1 枚目の元画像だけに偏っていないこと
        names = {p.name.split("_")[0] for p in (out / "debug").glob("*.png")}
        assert len(names) > 1, f"確認画像が偏っている: {names}"


def test_確認画像を出さないときはdebugが作られない():
    import tempfile
    from pathlib import Path

    import cv2
    import numpy as np

    from core.crop import generate_crops

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "a.png"
        cv2.imwrite(str(p), np.zeros((480, 640, 3), np.uint8))
        out = Path(d) / "out"
        r = generate_crops(
            {str(p): [{"bbox_xyxy": [100, 100, 160, 160], "confidence": 0.9,
                       "label": "o", "class_id": 0}]}, out)
        assert r["debug_images"] == 0
        assert not (out / "debug").exists()


def test_探索の作業場はモデル一覧に出ない():
    """探索は使い捨ての学習を何度も回す。その中間生成物 (.tuning/train-2 など) が
    本物のモデルに紛れると、選択肢が汚れて取り違えのもとになる。"""
    import tempfile
    from pathlib import Path

    from core.models import model_run_dirs, model_weight_files

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        for rel in ("good_run/weights/best.pt", ".tuning/train-2/weights/best.pt",
                    ".tuning/tune_x/weights/last.pt"):
            f = root / rel
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(b"x")

        got = model_weight_files(root)
        assert len(got) == 1 and got[0].name == "best.pt"
        assert ".tuning" not in str(got[0])
        assert [x.name for x in model_run_dirs(root)] == ["good_run"]
