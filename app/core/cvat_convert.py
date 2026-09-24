# =============================================================================
# CVAT for images 1.1 (XML) → YOLO 形式への変換部品
#
#   画面を持たない純粋な処理だけを置く（テストしやすくするため）。
#   dataset.generate_yolo_dataset() と cvat.parse_cvat_xml() から使う。
#
#   ここで扱う落とし穴:
#     - 複数タスクをまとめると、XML も画像名（frame_000000.jpg など）も衝突する
#       → XML ごとに読み、出力名は重ならないように振り直す
#     - 別フォルダの同名画像（cam1/0001.jpg と cam2/0001.jpg）
#       → 出力名にフォルダを含める
#     - box の rotation、ellipse、mask（RLE）も形状として拾う
#     - pose は skeleton（キーポイント）を書き出し、data.yaml に kpt_shape を入れる
# =============================================================================
from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import numpy as np

__all__ = [
    "SHAPE_TAGS", "find_cvat_xmls", "read_label_meta", "summarize_cvat_xmls",
    "parse_points", "box_corners", "ellipse_points", "decode_mask", "mask_polygon",
    "shape_polygon", "fmt_bbox", "fmt_polygon", "obb_corners",
    "skeleton_keypoints", "fmt_pose", "infer_flip_idx",
    "ImageLocator", "NameAllocator",
]

# 画像ごとの形状として数えるもの（tag は画像単位のラベル）
SHAPE_TAGS = ("box", "polygon", "polyline", "points", "ellipse", "mask", "skeleton", "tag")


# ---------------------------------------------------------------------------
# XML の場所とメタ情報
# ---------------------------------------------------------------------------
def find_cvat_xmls(raw_dir: Path) -> list[Path]:
    """raw_dir 以下の CVAT の XML を返す（複数タスクをまとめたときは複数）"""
    out = []
    for p in sorted(Path(raw_dir).rglob("*.xml")):
        try:
            # 先頭だけ見て CVAT の XML か確かめる（無関係な XML を拾わない）
            with open(p, "rb") as f:
                head = f.read(4096)
            if b"<annotations" in head:
                out.append(p)
        except OSError:
            continue
    return out


def read_label_meta(root: ET.Element) -> tuple[list[str], dict[str, list[str]]]:
    """メタ情報から (トップレベルのラベル, {skeleton ラベル: キーポイント名の並び}) を返す。

    skeleton のキーポイントは `<parent>` を持つ別ラベルとして並ぶ。
    これらは学習のクラスではないので、ラベル一覧からは外す。
    タスクの XML は meta/task、プロジェクトの XML は meta/project の下にある。
    """
    labels: list[str] = []
    skeletons: dict[str, list[str]] = {}
    seen: set[str] = set()
    for lbl in root.iterfind("./meta//labels/label"):
        name = (lbl.findtext("name") or "").strip()
        if not name:
            continue
        parent = (lbl.findtext("parent") or "").strip()
        if parent:
            skeletons.setdefault(parent, [])
            if name not in skeletons[parent]:
                skeletons[parent].append(name)
            continue
        if name not in seen:
            seen.add(name)
            labels.append(name)
    return labels, skeletons


def summarize_cvat_xmls(xml_paths: list[Path]) -> dict:
    """複数の XML をまとめて、UI に出す統計を返す"""
    labels: list[str] = []
    skeletons: dict[str, list[str]] = {}
    types: set[str] = set()
    image_count = annotated = 0

    for xp in xml_paths:
        root = ET.parse(xp).getroot()
        lbls, skels = read_label_meta(root)
        for n in lbls:
            if n not in labels:
                labels.append(n)
        for k, v in skels.items():
            skeletons.setdefault(k, v)
        for img in root.iter("image"):
            image_count += 1
            has = False
            for child in img:
                if child.tag in SHAPE_TAGS:
                    types.add(child.tag)
                    has = True
                    # meta に無い skeleton でも、実物から並びを拾っておく
                    if child.tag == "skeleton":
                        lab = child.get("label", "")
                        if lab and lab not in skeletons:
                            skeletons[lab] = [p.get("label", "") for p in child.findall("points")]
            if has:
                annotated += 1

    # skeleton のキーポイント名（子ラベル）はクラス候補から外す
    children = {c for v in skeletons.values() for c in v}
    labels = [n for n in labels if n not in children]
    for k in skeletons:
        if k not in labels:
            labels.append(k)

    return {
        "xml_path": str(xml_paths[0]) if xml_paths else "",
        "xml_paths": [str(p) for p in xml_paths],
        "labels": labels,
        "skeletons": skeletons,
        "annotation_types": sorted(types),
        "image_count": image_count,
        "annotated_count": annotated,
    }


# ---------------------------------------------------------------------------
# 形状 → 座標
# ---------------------------------------------------------------------------
def parse_points(s: str) -> list[tuple[float, float]]:
    pts = []
    for pt in (s or "").split(";"):
        pt = pt.strip()
        if "," in pt:
            x, y = pt.split(",")[:2]
            pts.append((float(x), float(y)))
    return pts


def box_corners(elem: ET.Element) -> list[tuple[float, float]]:
    """box の 4 隅（rotation 属性があれば中心周りに回した位置）"""
    xtl, ytl = float(elem.get("xtl", 0)), float(elem.get("ytl", 0))
    xbr, ybr = float(elem.get("xbr", 0)), float(elem.get("ybr", 0))
    corners = [(xtl, ytl), (xbr, ytl), (xbr, ybr), (xtl, ybr)]
    rot = float(elem.get("rotation", 0) or 0)
    if rot:
        cx, cy = (xtl + xbr) / 2, (ytl + ybr) / 2
        a = math.radians(rot)
        ca, sa = math.cos(a), math.sin(a)
        corners = [(cx + (x - cx) * ca - (y - cy) * sa,
                    cy + (x - cx) * sa + (y - cy) * ca) for x, y in corners]
    return corners


def ellipse_points(elem: ET.Element, n: int = 32) -> list[tuple[float, float]]:
    """ellipse を n 角形で近似する"""
    cx, cy = float(elem.get("cx", 0)), float(elem.get("cy", 0))
    rx, ry = float(elem.get("rx", 0)), float(elem.get("ry", 0))
    a = math.radians(float(elem.get("rotation", 0) or 0))
    ca, sa = math.cos(a), math.sin(a)
    out = []
    for i in range(n):
        t = 2 * math.pi * i / n
        x, y = rx * math.cos(t), ry * math.sin(t)
        out.append((cx + x * ca - y * sa, cy + x * sa + y * ca))
    return out


def decode_mask(elem: ET.Element) -> Optional[tuple[np.ndarray, int, int]]:
    """CVAT の mask（RLE）を (2値画像, left, top) に戻す。

    rle は「0 の個数, 1 の個数, 0 の個数, ...」を行優先で並べたもの。
    範囲は left/top/width/height の矩形の中だけ。
    """
    try:
        counts = [int(float(c)) for c in (elem.get("rle") or "").split(",") if c.strip()]
        left, top = int(float(elem.get("left", 0))), int(float(elem.get("top", 0)))
        width, height = int(float(elem.get("width", 0))), int(float(elem.get("height", 0)))
    except ValueError:
        return None
    if width <= 0 or height <= 0 or not counts:
        return None
    flat = np.zeros(width * height, np.uint8)
    pos, val = 0, 0
    for c in counts:
        if val:
            flat[pos:pos + c] = 1
        pos += c
        val ^= 1
    return flat.reshape(height, width), left, top


def mask_polygon(elem: ET.Element) -> list[tuple[float, float]]:
    """mask を輪郭のポリゴンにする（YOLO は 1 物体 1 ポリゴンなので最大の輪郭を使う）"""
    dec = decode_mask(elem)
    if dec is None:
        return []
    import cv2
    m, left, top = dec
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    c = max(contours, key=cv2.contourArea)
    if len(c) < 3:
        # 細すぎて輪郭が線になったときは外接矩形で代用する
        x, y, w, h = cv2.boundingRect(c)
        return [(left + x, top + y), (left + x + w, top + y),
                (left + x + w, top + y + h), (left + x, top + y + h)]
    return [(float(left + p[0][0]), float(top + p[0][1])) for p in c]


def shape_polygon(elem: ET.Element) -> list[tuple[float, float]]:
    """box / polygon / ellipse / mask を絶対座標のポリゴンにそろえる"""
    if elem.tag == "box":
        return box_corners(elem)
    if elem.tag == "polygon":
        return parse_points(elem.get("points", ""))
    if elem.tag == "ellipse":
        return ellipse_points(elem)
    if elem.tag == "mask":
        return mask_polygon(elem)
    return []


def _c(v: float) -> float:
    return min(max(v, 0.0), 1.0)


def fmt_bbox(pts, w: int, h: int) -> Optional[str]:
    """点列の外接矩形を YOLO の cx cy w h（正規化・画像内に収める）にする"""
    if not pts:
        return None
    xs = [_c(x / w) for x, _ in pts]
    ys = [_c(y / h) for _, y in pts]
    x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
    if x2 - x1 <= 0 or y2 - y1 <= 0:
        return None
    return f"{(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}"


def fmt_polygon(pts, w: int, h: int) -> Optional[str]:
    if len(pts) < 3:
        return None
    return " ".join(f"{_c(x / w):.6f} {_c(y / h):.6f}" for x, y in pts)


def obb_corners(pts) -> list[tuple[float, float]]:
    """4 点ならそのまま、それ以外は最小外接回転矩形にする"""
    if len(pts) == 4:
        return list(pts)
    if len(pts) < 3:
        return []
    import cv2
    rect = cv2.minAreaRect(np.array(pts, np.float32))
    return [(float(x), float(y)) for x, y in cv2.boxPoints(rect)]


# ---------------------------------------------------------------------------
# pose（キーポイント）
# ---------------------------------------------------------------------------
def skeleton_keypoints(elem: ET.Element, kpt_names: list[str]) -> list[tuple[float, float, int]]:
    """skeleton を kpt_names の順の (x, y, v) にする。

    v は YOLO の可視性: 0=無し（outside）/ 1=隠れている（occluded）/ 2=見えている
    """
    by_name = {p.get("label", ""): p for p in elem.findall("points")}
    out = []
    for n in kpt_names:
        p = by_name.get(n)
        pts = parse_points(p.get("points", "")) if p is not None else []
        if p is None or not pts or p.get("outside", "0") == "1":
            out.append((0.0, 0.0, 0))
            continue
        v = 1 if p.get("occluded", "0") == "1" else 2
        out.append((pts[0][0], pts[0][1], v))
    return out


def fmt_pose(kpts: list[tuple[float, float, int]], w: int, h: int) -> Optional[str]:
    """キーポイントから YOLO pose の 1 行（cx cy w h + x y v ...）を作る。
    ボックスは見えているキーポイントの外接矩形にする（CVAT 公式の変換と同じ）。"""
    vis = [(x, y) for x, y, v in kpts if v > 0]
    box = fmt_bbox(vis, w, h)
    if box is None:
        return None
    kp = " ".join(
        f"{_c(x / w):.6f} {_c(y / h):.6f} {v}" if v > 0 else "0.000000 0.000000 0"
        for x, y, v in kpts)
    return f"{box} {kp}"


_LR = [(r"left", "right"), (r"right", "left"), (r"Left", "Right"), (r"Right", "Left"),
       (r"LEFT", "RIGHT"), (r"RIGHT", "LEFT")]


def infer_flip_idx(kpt_names: list[str]) -> Optional[list[int]]:
    """左右反転したときに入れ替わるキーポイントの並び（YOLO の flip_idx）を名前から推定する。

    left_eye ↔ right_eye、l_hand ↔ r_hand、hand_l ↔ hand_r のような名前を対にする。
    対が 1 つも見つからなければ None（YOLO は左右反転の拡張を切る）。
    """
    idx = {n: i for i, n in enumerate(kpt_names)}

    def partner(n: str) -> Optional[str]:
        for a, b in _LR:
            if a in n:
                return n.replace(a, b)
        m = re.match(r"^([lrLR])([_\-.])(.*)$", n)
        if m:
            sw = {"l": "r", "r": "l", "L": "R", "R": "L"}[m.group(1)]
            return f"{sw}{m.group(2)}{m.group(3)}"
        m = re.match(r"^(.*)([_\-.])([lrLR])$", n)
        if m:
            sw = {"l": "r", "r": "l", "L": "R", "R": "L"}[m.group(3)]
            return f"{m.group(1)}{m.group(2)}{sw}"
        return None

    out, paired = [], False
    for i, n in enumerate(kpt_names):
        p = partner(n)
        if p is not None and p in idx and p != n:
            out.append(idx[p])
            paired = True
        else:
            out.append(i)
    return out if paired else None


# ---------------------------------------------------------------------------
# 画像の場所と出力名
# ---------------------------------------------------------------------------
class ImageLocator:
    """XML 1 つぶんの画像を探す。

    CVAT のエクスポートは `images/<name>`（プロジェクトなら `images/<subset>/<name>`）。
    それで見つからないときだけファイル名で探すが、同名が複数あれば取り違えるので諦める。
    """

    def __init__(self, xml_path: Path):
        self.base = Path(xml_path).parent
        self._by_name: Optional[dict[str, list[Path]]] = None

    def find(self, name: str, subset: str = "") -> Optional[Path]:
        cands = [self.base / "images" / name, self.base / name]
        if subset:
            cands.insert(0, self.base / "images" / subset / name)
        for p in cands:
            if p.is_file():
                return p
        if self._by_name is None:
            self._by_name = {}
            for p in self.base.rglob("*"):
                if p.is_file() and p.suffix.lower() != ".xml":
                    self._by_name.setdefault(p.name, []).append(p)
        hits = self._by_name.get(Path(name).name, [])
        return hits[0] if len(hits) == 1 else None


class NameAllocator:
    """出力先で名前が重ならないようにする。

    画像名にフォルダが含まれていればそれも名前に入れ（cam1/0001.jpg → cam1__0001）、
    それでも重なれば連番を付ける（複数タスクの frame_000000.jpg など）。
    """

    def __init__(self):
        self._used: set[str] = set()

    def allocate(self, name: str, prefix: str = "") -> str:
        p = Path(name)
        parts = [x for x in p.with_suffix("").parts if x not in ("", ".", "..", "/")]
        stem = "__".join(parts) or "image"
        base = f"{prefix}__{stem}" if prefix else stem
        cand, i = base, 1
        while cand.lower() in self._used:
            cand = f"{base}_{i}"
            i += 1
        self._used.add(cand.lower())
        return cand
