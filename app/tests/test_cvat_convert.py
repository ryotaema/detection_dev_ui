# =============================================================================
# CVAT for images 1.1 (XML) → YOLO 変換のテスト
#
#   実際に踏むと「エラーは出ないのに学習データが静かに壊れる」ものを中心に見る。
#     - 別フォルダの同名画像でラベルが上書きされる
#     - 複数タスクをまとめると 2 タスク目以降が消える
#     - pose にキーポイントが入らない / kpt_shape が無い
#     - mask・楕円・回転 box が無視される
# =============================================================================
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from core import cvat_convert as cc
from core.cvat import parse_cvat_xml
from core.dataset import generate_yolo_dataset

META = """
<meta><task><labels>
  <label><name>car</name><type>rectangle</type></label>
  <label><name>person</name><type>skeleton</type></label>
  <label><name>left_hand</name><type>points</type><parent>person</parent></label>
  <label><name>right_hand</name><type>points</type><parent>person</parent></label>
  <label><name>head</name><type>points</type><parent>person</parent></label>
</labels></task></meta>
"""


def _write_task(raw: Path, images: dict[str, str], size=(100, 100)) -> Path:
    """raw/annotations.xml と raw/images/<name> を作る。images は {名前: 画像内の要素}"""
    import cv2
    w, h = size
    body = []
    for i, (name, inner) in enumerate(images.items()):
        p = raw / "images" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p), np.full((h, w, 3), i * 10, np.uint8))
        body.append(f'<image id="{i}" name="{name}" width="{w}" height="{h}">{inner}</image>')
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "annotations.xml").write_text(
        f'<?xml version="1.0"?><annotations><version>1.1</version>{META}{"".join(body)}</annotations>')
    return raw


def _box(label="car", x1=10, y1=20, x2=30, y2=60, rot=None):
    r = f' rotation="{rot}"' if rot is not None else ""
    return f'<box label="{label}" xtl="{x1}" ytl="{y1}" xbr="{x2}" ybr="{y2}"{r}></box>'


def _labels(ds: Path) -> dict[str, list[str]]:
    out = {}
    for p in sorted(ds.glob("labels/*/*.txt")):
        out[p.stem] = p.read_text().strip().splitlines()
    return out


# ---------------------------------------------------------------------------
# 名前の衝突
# ---------------------------------------------------------------------------
def test_別フォルダの同名画像を潰さない(tmp_path):
    raw = _write_task(tmp_path / "raw", {
        "cam1/0001.jpg": _box(x1=0, x2=10),
        "cam2/0001.jpg": _box(x1=50, x2=90),
    })
    info = parse_cvat_xml(raw)
    out = generate_yolo_dataset(raw, info, ["car"], "detect", tmp_path / "ds")
    labels = _labels(out)
    assert set(labels) == {"cam1__0001", "cam2__0001"}
    # 画像とラベルが取り違えられていない
    assert labels["cam1__0001"][0].startswith("0 0.050000")
    assert labels["cam2__0001"][0].startswith("0 0.700000")
    assert len(list(out.glob("images/*/*.jpg"))) == 2


def test_複数タスクをまとめても2タスク目が消えない(tmp_path):
    parent = tmp_path / "export"
    r1 = _write_task(parent / "task_1" / "raw", {"frame_000000.jpg": _box(x1=0, x2=10)})
    r2 = _write_task(parent / "task_2" / "raw", {"frame_000000.jpg": _box(x1=50, x2=90)})
    info = parse_cvat_xml([r1, r2])
    assert len(info["xml_paths"]) == 2
    assert info["image_count"] == 2
    out = generate_yolo_dataset(parent, info, ["car"], "detect", tmp_path / "ds")
    labels = _labels(out)
    assert set(labels) == {"task_1__frame_000000", "task_2__frame_000000"}
    assert labels["task_1__frame_000000"][0].startswith("0 0.050000")
    assert labels["task_2__frame_000000"][0].startswith("0 0.700000")


def test_名前の割り当ては重なれば連番を付ける():
    na = cc.NameAllocator()
    assert na.allocate("a/b.jpg") == "a__b"
    assert na.allocate("a/b.png") == "a__b_1"
    assert na.allocate("../x.jpg") == "x"


def test_同名画像が複数あるときはファイル名だけで探さない(tmp_path):
    base = tmp_path / "raw"
    for d in ("a", "b"):
        (base / "other" / d).mkdir(parents=True)
        (base / "other" / d / "x.jpg").write_bytes(b"x")
    (base / "annotations.xml").write_text("<annotations/>")
    loc = cc.ImageLocator(base / "annotations.xml")
    assert loc.find("x.jpg") is None


# ---------------------------------------------------------------------------
# 形状
# ---------------------------------------------------------------------------
def test_回転boxはdetectで外接矩形になる(tmp_path):
    raw = _write_task(tmp_path / "raw", {"a.jpg": _box(x1=40, y1=45, x2=60, y2=55, rot=90)})
    out = generate_yolo_dataset(raw, parse_cvat_xml(raw), ["car"], "detect", tmp_path / "ds")
    cls, cx, cy, w, h = _labels(out)["a"][0].split()
    # 20x10 の箱を 90 度回すと 10x20
    assert float(w) == pytest.approx(0.10, abs=1e-4)
    assert float(h) == pytest.approx(0.20, abs=1e-4)


def test_はみ出した座標は画像内に収める(tmp_path):
    raw = _write_task(tmp_path / "raw", {"a.jpg": _box(x1=-10, y1=-10, x2=50, y2=50)})
    out = generate_yolo_dataset(raw, parse_cvat_xml(raw), ["car"], "detect", tmp_path / "ds")
    _, cx, cy, w, h = map(float, _labels(out)["a"][0].split())
    assert (cx, cy, w, h) == pytest.approx((0.25, 0.25, 0.5, 0.5))


def test_maskのRLEを戻せる():
    # 3x2 の範囲: 0 0 1 / 1 1 0 → counts = 2,3,1
    el = cc.ET.fromstring('<mask rle="2,3,1" left="5" top="7" width="3" height="2"/>')
    m, left, top = cc.decode_mask(el)
    assert (left, top) == (5, 7)
    assert m.tolist() == [[0, 0, 1], [1, 1, 0]]


def test_maskはsegmentのポリゴンになる(tmp_path):
    # 10x10 の正方形マスク（左上 20,30）
    rle = "0,100"
    raw = _write_task(tmp_path / "raw", {
        "a.jpg": f'<mask label="car" rle="{rle}" left="20" top="30" width="10" height="10"></mask>'})
    info = parse_cvat_xml(raw)
    assert "mask" in info["annotation_types"]
    out = generate_yolo_dataset(raw, info, ["car"], "segment", tmp_path / "ds")
    vals = list(map(float, _labels(out)["a"][0].split()[1:]))
    xs, ys = vals[0::2], vals[1::2]
    assert min(xs) == pytest.approx(0.20) and max(xs) == pytest.approx(0.29)
    assert min(ys) == pytest.approx(0.30) and max(ys) == pytest.approx(0.39)


def test_楕円はsegmentでポリゴンに_detectで矩形になる(tmp_path):
    el = '<ellipse label="car" cx="50" cy="50" rx="20" ry="10"></ellipse>'
    raw = _write_task(tmp_path / "raw", {"a.jpg": el})
    info = parse_cvat_xml(raw)
    seg = generate_yolo_dataset(raw, info, ["car"], "segment", tmp_path / "seg")
    assert len(_labels(seg)["a"][0].split()) == 1 + 32 * 2
    det = generate_yolo_dataset(raw, info, ["car"], "detect", tmp_path / "det")
    _, cx, cy, w, h = map(float, _labels(det)["a"][0].split())
    assert (cx, cy, w, h) == pytest.approx((0.5, 0.5, 0.4, 0.2), abs=1e-3)


def test_obbは多角形を最小外接回転矩形にする():
    pts = [(0, 0), (10, 0), (10, 10), (5, 12), (0, 10)]
    assert len(cc.obb_corners(pts)) == 4
    assert cc.obb_corners([(0, 0), (1, 0), (1, 1), (0, 1)]) == [(0, 0), (1, 0), (1, 1), (0, 1)]


# ---------------------------------------------------------------------------
# pose
# ---------------------------------------------------------------------------
def _skeleton(occluded_head=False, outside_right=False):
    return (
        '<skeleton label="person">'
        '<points label="left_hand" outside="0" occluded="0" points="20,40"></points>'
        f'<points label="right_hand" outside="{int(outside_right)}" occluded="0" points="80,40"></points>'
        f'<points label="head" outside="0" occluded="{int(occluded_head)}" points="50,10"></points>'
        '</skeleton>')


def test_skeletonのキーポイント名はクラス候補に出さない(tmp_path):
    raw = _write_task(tmp_path / "raw", {"a.jpg": _skeleton()})
    info = parse_cvat_xml(raw)
    assert info["labels"] == ["car", "person"]
    assert info["skeletons"]["person"] == ["left_hand", "right_hand", "head"]
    assert "skeleton" in info["annotation_types"]


def test_poseはキーポイントとkpt_shapeを書き出す(tmp_path):
    raw = _write_task(tmp_path / "raw", {
        "a.jpg": _skeleton(occluded_head=True),
        "b.jpg": _skeleton(outside_right=True),
    })
    out = generate_yolo_dataset(raw, parse_cvat_xml(raw), ["person"], "pose", tmp_path / "ds")
    cfg = yaml.safe_load((out / "data.yaml").read_text())
    assert cfg["kpt_shape"] == [3, 3]
    # left_hand ↔ right_hand を入れ替え、head はそのまま
    assert cfg["flip_idx"] == [1, 0, 2]

    labels = _labels(out)
    a = labels["a"][0].split()
    assert len(a) == 1 + 4 + 3 * 3
    assert a[5:8] == ["0.200000", "0.400000", "2"]
    assert a[11:14] == ["0.500000", "0.100000", "1"]     # 隠れている
    b = labels["b"][0].split()
    assert b[8:11] == ["0.000000", "0.000000", "0"]      # 画面外
    # ボックスは見えている点の外接矩形（right_hand を除く）
    assert float(b[3]) == pytest.approx(0.30)


def test_キーポイント数が違うラベルを混ぜたらpose生成を止める(tmp_path):
    info = {"skeletons": {"a": ["x", "y"], "b": ["x"]}}
    from core.dataset import _pose_keypoint_names
    names, err = _pose_keypoint_names(info, ["a", "b"], [])
    assert names is None and "キーポイントの数" in err


def test_左右の対を名前から推定する():
    assert cc.infer_flip_idx(["nose", "left_eye", "right_eye"]) == [0, 2, 1]
    assert cc.infer_flip_idx(["l_hand", "r_hand"]) == [1, 0]
    assert cc.infer_flip_idx(["hand_L", "hand_R"]) == [1, 0]
    assert cc.infer_flip_idx(["a", "b"]) is None


def test_Ultralyticsがposeのdata_yamlを読める(tmp_path):
    """kpt_shape / flip_idx / kpt_names の形が Ultralytics の検査を通ること"""
    pytest.importorskip("ultralytics")
    from ultralytics.data.utils import check_det_dataset
    raw = _write_task(tmp_path / "raw", {f"{i}.jpg": _skeleton() for i in range(5)})
    out = generate_yolo_dataset(raw, parse_cvat_xml(raw), ["person"], "pose", tmp_path / "ds")
    data = check_det_dataset(str(out / "data.yaml"))
    assert data["kpt_shape"] == [3, 3]


# ---------------------------------------------------------------------------
# 分割・背景画像
# ---------------------------------------------------------------------------
def test_フォルダごとに分けると同じカメラは片側に入る(tmp_path):
    imgs = {f"cam{c}/{i:03d}.jpg": _box() for c in range(4) for i in range(5)}
    raw = _write_task(tmp_path / "raw", imgs)
    out = generate_yolo_dataset(raw, parse_cvat_xml(raw), ["car"], "detect",
                                tmp_path / "ds", val_ratio=0.25, split_mode="folder")
    tr = {p.stem.split("__")[0] for p in (out / "labels" / "train").iterdir()}
    va = {p.stem.split("__")[0] for p in (out / "labels" / "val").iterdir()}
    assert va and tr.isdisjoint(va)


def test_同じシードなら何度作っても同じ分け方(tmp_path):
    raw = _write_task(tmp_path / "raw", {f"{i:03d}.jpg": _box() for i in range(20)})
    info = parse_cvat_xml(raw)
    a = generate_yolo_dataset(raw, info, ["car"], "detect", tmp_path / "a", seed=3)
    b = generate_yolo_dataset(raw, info, ["car"], "detect", tmp_path / "b", seed=3)
    assert (sorted(p.name for p in (a / "labels" / "val").iterdir())
            == sorted(p.name for p in (b / "labels" / "val").iterdir()))


def test_背景画像は選んだときだけ空ラベルで入る(tmp_path):
    raw = _write_task(tmp_path / "raw", {"a.jpg": _box(), "empty.jpg": ""})
    info = parse_cvat_xml(raw)
    no = generate_yolo_dataset(raw, info, ["car"], "detect", tmp_path / "no")
    assert "empty" not in _labels(no)
    yes = generate_yolo_dataset(raw, info, ["car"], "detect", tmp_path / "yes",
                                include_background=True)
    assert _labels(yes)["empty"] == []
    assert len(list(yes.glob("images/*/*.jpg"))) == 2
