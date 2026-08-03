# =============================================================================
# モザイク処理のテスト
#
#   **元画像を上書きする処理**なので、次の 3 点を重点的に確かめる:
#     1. 書き換える前に必ず退避されること
#     2. 2 回実行しても原本が失われないこと
#     3. 退避できなければ書き換えないこと
# =============================================================================
from __future__ import annotations

import json

import numpy as np
import pytest

import cv2

from core.mosaic import (
    BACKUP_DIR_NAME, METHODS, apply_mosaic, backup_path_for, backup_root,
    count_backup, dataset_image_paths, has_backup, mosaic_image,
    preview_mosaic, regions_from_cvat_xml, regions_from_fixed,
    regions_from_predictions, restore_mosaic,
)


def _img(w=80, h=60, value=200):
    """一様な色の画像。モザイクがかかると値がばらつくので判定しやすい。"""
    a = np.full((h, w, 3), value, dtype=np.uint8)
    # 中央に模様を入れて、粗くしたときに変化が出るようにする
    a[20:40, 20:60] = np.random.RandomState(0).randint(0, 255, (20, 40, 3))
    return a


@pytest.fixture
def ds(tmp_path):
    """images/train に画像 3 枚を持つデータセット"""
    d = tmp_path / "bg_data"
    (d / "images" / "train").mkdir(parents=True)
    for i in range(3):
        cv2.imwrite(str(d / "images" / "train" / f"img{i}.png"), _img())
    return d


def _paths(ds):
    return sorted((ds / "images" / "train").glob("*.png"))


# ---------------------------------------------------------------------------
# 画像 1 枚への適用
# ---------------------------------------------------------------------------
def test_指定領域だけが変わる():
    img = _img()
    out, applied = mosaic_image(img, [(20, 20, 60, 40)], "pixelate", 8, padding=0)
    assert applied == 1
    assert not np.array_equal(out[20:40, 20:60], img[20:40, 20:60]), "領域が変わっていない"
    assert np.array_equal(out[0:10, 0:10], img[0:10, 0:10]), "領域外まで変わっている"


def test_余白のぶん広く隠す():
    img = _img()
    out, _ = mosaic_image(img, [(30, 25, 50, 35)], "pixelate", 8, padding=10)
    # 余白で 20,15 まで広がる
    assert not np.array_equal(out[15:20, 20:25], img[15:20, 20:25])


def test_画像の外にはみ出しても落ちない():
    img = _img(80, 60)
    out, applied = mosaic_image(img, [(-50, -50, 500, 500)], "pixelate", 8, padding=20)
    assert applied == 1 and out.shape == img.shape


def test_つぶれた領域は飛ばす():
    img = _img()
    _, applied = mosaic_image(img, [(30, 30, 30, 30)], "pixelate", 8, padding=0)
    assert applied == 0


@pytest.mark.parametrize("method", list(METHODS))
def test_どの方式でも動く(method):
    img = _img()
    out, applied = mosaic_image(img, [(20, 20, 60, 40)], method, 9, padding=0)
    assert applied == 1 and out.shape == img.shape


def test_塗りつぶしは黒くなる():
    img = _img()
    out, _ = mosaic_image(img, [(20, 20, 60, 40)], "fill", 8, padding=0)
    assert out[25, 25].tolist() == [0, 0, 0]


# ---------------------------------------------------------------------------
# 領域の集め方
# ---------------------------------------------------------------------------
def test_推論結果から領域を集める(tmp_path):
    jf = tmp_path / "a.json"
    jf.write_text(json.dumps({
        "image_path": "/x/1.png",
        "boxes": [
            {"label": "bell_pepper", "confidence": 0.9, "bbox_xyxy": [1, 2, 3, 4]},
            {"label": "bell_pepper", "confidence": 0.05, "bbox_xyxy": [5, 6, 7, 8]},
            {"label": "leaf", "confidence": 0.9, "bbox_xyxy": [9, 9, 9, 9]},
        ]}))
    got = regions_from_predictions([jf], conf=0.10)
    assert got == {"/x/1.png": [(1.0, 2.0, 3.0, 4.0), (9.0, 9.0, 9.0, 9.0)]}

    only = regions_from_predictions([jf], labels=["bell_pepper"], conf=0.10)
    assert only == {"/x/1.png": [(1.0, 2.0, 3.0, 4.0)]}


def test_低い信頼度も拾える(tmp_path):
    """隠す目的では取りこぼしのほうが高くつくので、しきい値を下げられること"""
    jf = tmp_path / "a.json"
    jf.write_text(json.dumps({"image_path": "/x/1.png", "boxes": [
        {"label": "p", "confidence": 0.02, "bbox_xyxy": [1, 1, 2, 2]}]}))
    assert regions_from_predictions([jf], conf=0.5) == {}
    assert regions_from_predictions([jf], conf=0.01) != {}


def test_壊れたJSONは飛ばす(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{ 壊れている")
    assert regions_from_predictions([bad]) == {}


def test_固定領域は画像サイズに合わせて画素座標になる(ds):
    paths = _paths(ds)
    got = regions_from_fixed(paths, (0.0, 0.5, 0.5, 1.0))
    # 画像は 80x60
    assert got[str(paths[0])] == [(0.0, 30.0, 40.0, 60.0)]
    assert len(got) == 3


def test_固定領域の指定が逆でも直す(ds):
    paths = _paths(ds)
    got = regions_from_fixed(paths[:1], (0.5, 1.0, 0.0, 0.5))
    assert got[str(paths[0])] == [(0.0, 30.0, 40.0, 60.0)]


def test_CVATのXMLからマスク領域を取る(ds, tmp_path):
    xml = tmp_path / "annotations.xml"
    xml.write_text("""<?xml version="1.0"?>
<annotations>
  <image id="0" name="img0.png" width="80" height="60">
    <box label="mask" xtl="10" ytl="10" xbr="30" ybr="30"/>
    <box label="bell_pepper" xtl="1" ytl="1" xbr="5" ybr="5"/>
    <polygon label="mask" points="40,10;60,10;60,25;40,25"/>
  </image>
</annotations>""", encoding="utf-8")

    got = regions_from_cvat_xml(xml, ["mask"], [ds / "images" / "train"])
    key = str(ds / "images" / "train" / "img0.png")
    assert key in got
    # box はそのまま、polygon は外接矩形になる
    assert (10.0, 10.0, 30.0, 30.0) in got[key]
    assert (40.0, 10.0, 60.0, 25.0) in got[key]
    # 学習対象のラベルは拾わない
    assert (1.0, 1.0, 5.0, 5.0) not in got[key]


def test_CVATの画像が見つからなければ飛ばす(tmp_path):
    xml = tmp_path / "a.xml"
    xml.write_text('<annotations><image name="ない.png">'
                   '<box label="mask" xtl="1" ytl="1" xbr="2" ybr="2"/>'
                   '</image></annotations>', encoding="utf-8")
    assert regions_from_cvat_xml(xml, ["mask"], [tmp_path]) == {}


# ---------------------------------------------------------------------------
# 退避（いちばん大事）
# ---------------------------------------------------------------------------
def test_書き換える前に退避される(ds):
    paths = _paths(ds)
    before = cv2.imread(str(paths[0])).copy()

    res = apply_mosaic(ds, {str(paths[0]): [(20, 20, 60, 40)]}, padding=0)
    assert res["ok"], res["error"]
    assert res["backed_up"] == 1

    bak = backup_path_for(paths[0], ds)
    assert bak.exists(), "退避されていない"
    assert np.array_equal(cv2.imread(str(bak)), before), "退避の中身が原本と違う"
    assert not np.array_equal(cv2.imread(str(paths[0])), before), "上書きされていない"


def test_2回実行しても原本を失わない(ds):
    """2 回目に「モザイク済みの画像」で退避を上書きすると原本が消える"""
    paths = _paths(ds)
    original = cv2.imread(str(paths[0])).copy()
    regions = {str(paths[0]): [(20, 20, 60, 40)]}

    apply_mosaic(ds, regions, padding=0)
    res2 = apply_mosaic(ds, regions, padding=0)

    assert res2["backed_up"] == 0, "2 回目に退避を上書きしている"
    bak = backup_path_for(paths[0], ds)
    assert np.array_equal(cv2.imread(str(bak)), original), \
        "退避が上書きされ、原本が失われている"


def test_退避から戻せる(ds):
    paths = _paths(ds)
    originals = [cv2.imread(str(p)).copy() for p in paths]
    regions = {str(p): [(20, 20, 60, 40)] for p in paths}

    apply_mosaic(ds, regions, padding=0)
    assert not np.array_equal(cv2.imread(str(paths[0])), originals[0])

    res = restore_mosaic(ds)
    assert res["ok"] and res["restored"] == 3
    for p, orig in zip(paths, originals):
        assert np.array_equal(cv2.imread(str(p)), orig), "戻せていない"


def test_退避が無ければ戻せないと言う(ds):
    res = restore_mosaic(ds)
    assert not res["ok"] and "ありません" in res["error"]


def test_データセットの外の画像は書き換えない(ds, tmp_path):
    outside = tmp_path / "よそ.png"
    cv2.imwrite(str(outside), _img())
    before = cv2.imread(str(outside)).copy()

    res = apply_mosaic(ds, {str(outside): [(10, 10, 30, 30)]}, padding=0)
    assert res["images"] == 0
    assert any("外にある" in why for _, why in res["skipped"])
    assert np.array_equal(cv2.imread(str(outside)), before), "外の画像を書き換えた"


def test_退避先の場所(ds):
    p = _paths(ds)[0]
    bak = backup_path_for(p, ds)
    assert bak == ds / BACKUP_DIR_NAME / "images" / "train" / "img0.png"


def test_退避先の画像は対象に含めない(ds):
    paths = _paths(ds)
    apply_mosaic(ds, {str(paths[0]): [(20, 20, 60, 40)]}, padding=0)
    found = dataset_image_paths(ds)
    assert all(BACKUP_DIR_NAME not in p.parts for p in found)
    assert len(found) == 3


def test_退避の有無と件数(ds):
    assert not has_backup(ds) and count_backup(ds) == 0
    apply_mosaic(ds, {str(_paths(ds)[0]): [(20, 20, 60, 40)]}, padding=0)
    assert has_backup(ds) and count_backup(ds) == 1


# ---------------------------------------------------------------------------
# 下見と拒否条件
# ---------------------------------------------------------------------------
def test_下見では何も書き換えない(ds):
    paths = _paths(ds)
    before = [cv2.imread(str(p)).copy() for p in paths]

    res = apply_mosaic(ds, {str(p): [(20, 20, 60, 40)] for p in paths},
                       dry_run=True)
    assert res["images"] == 3 and res["regions"] == 3
    assert not has_backup(ds), "下見なのに退避が作られている"
    for p, b in zip(paths, before):
        assert np.array_equal(cv2.imread(str(p)), b), "下見なのに書き換わっている"


def test_プレビューは元と処理後を返す(ds):
    p = _paths(ds)[0]
    before_file = cv2.imread(str(p)).copy()
    got = preview_mosaic(p, [(20, 20, 60, 40)], padding=0)
    assert got is not None
    orig, masked, applied = got
    assert applied == 1
    assert not np.array_equal(orig, masked)
    assert np.array_equal(cv2.imread(str(p)), before_file), "プレビューで書き換わっている"


def test_対象が無ければ何もしない(ds):
    res = apply_mosaic(ds, {})
    assert not res["ok"] and "領域がありません" in res["error"]


def test_知らない方式は拒否する(ds):
    res = apply_mosaic(ds, {str(_paths(ds)[0]): [(1, 1, 2, 2)]}, method="変な方式")
    assert not res["ok"] and "知らない方式" in res["error"]
    assert not has_backup(ds)


def test_存在しないデータセットは拒否する(tmp_path):
    res = apply_mosaic(tmp_path / "ない", {"/x/1.png": [(1, 1, 2, 2)]})
    assert not res["ok"] and "ありません" in res["error"]


def test_読めない画像は飛ばして続ける(ds):
    paths = _paths(ds)
    broken = ds / "images" / "train" / "broken.png"
    broken.write_text("これは画像ではない")

    res = apply_mosaic(ds, {
        str(broken): [(1, 1, 2, 2)],
        str(paths[0]): [(20, 20, 60, 40)],
    }, padding=0)
    assert res["images"] == 1, "読めるほうまで止まっている"
    assert any("読めません" in why for _, why in res["skipped"])
