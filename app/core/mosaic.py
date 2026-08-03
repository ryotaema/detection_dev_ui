# =============================================================================
# モザイク処理（写り込みを隠す）
#
#   用途は 2 つ:
#     - 背景データに少しだけ写り込んだ対象物（学習で「背景」と覚えられると困る）
#     - プライバシー（顔・ナンバープレート）
#
#   **画像を上書きする処理**なので、安全側に倒した作りにしている:
#     - 書き換える前に必ず `_backup_original/` へ退避する
#     - 退避に失敗した画像は書き換えない
#     - すでに退避済みのものは**上書きしない**（2 回目の実行で原本を失わないため）
#     - 適用前に確認できるよう、プレビューを別に用意している
#     - `restore_mosaic()` でいつでも戻せる
#
#   隠し漏れのほうが隠しすぎより高くつくので、
#   呼び出し側の既定値は「conf 低め・余白あり」にすること。
# =============================================================================
from __future__ import annotations

import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

from .config import IMG_EXTS

BACKUP_DIR_NAME = "_backup_original"

METHODS = {
    "pixelate": "モザイク（画素を粗くする）",
    "blur":     "ぼかし（境目が目立ちにくい）",
    "fill":     "black（塗りつぶし・非推奨）",
}


# ---------------------------------------------------------------------------
# 画像 1 枚への適用
# ---------------------------------------------------------------------------
def mosaic_image(img, regions, method: str = "pixelate",
                 strength: int = 12, padding: int = 0):
    """画像の指定領域を隠す。img は BGR の ndarray。

    regions: [(x1, y1, x2, y2), ...] 画素座標
    strength: pixelate なら「何画素を 1 マスにするか」、blur ならカーネルの大きさ
    padding: 各領域を上下左右へ広げる画素数（隠し漏れを減らすため）
    """
    import cv2

    h, w = img.shape[:2]
    out = img.copy()
    applied = 0

    for x1, y1, x2, y2 in regions:
        # 余白を足してから画像の内側に収める
        x1 = max(0, int(round(x1)) - padding)
        y1 = max(0, int(round(y1)) - padding)
        x2 = min(w, int(round(x2)) + padding)
        y2 = min(h, int(round(y2)) + padding)
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue

        roi = out[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        if method == "blur":
            k = max(3, int(strength) | 1)          # カーネルは奇数
            out[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)
        elif method == "fill":
            out[y1:y2, x1:x2] = 0
        else:                                       # pixelate
            block = max(2, int(strength))
            sw = max(1, (x2 - x1) // block)
            sh = max(1, (y2 - y1) // block)
            small = cv2.resize(roi, (sw, sh), interpolation=cv2.INTER_LINEAR)
            out[y1:y2, x1:x2] = cv2.resize(
                small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
        applied += 1

    return out, applied


# ---------------------------------------------------------------------------
# 領域の集め方（3 通り）
# ---------------------------------------------------------------------------
def regions_from_predictions(
    pred_jsons,
    labels: Optional[list[str]] = None,
    conf: float = 0.10,
) -> dict[str, list[tuple]]:
    """推論結果から領域を集める。

    隠す目的なので conf の既定は低め。取りこぼすほうが高くつく。
    labels を渡すとそのクラスだけを対象にする。
    """
    out: dict[str, list[tuple]] = {}
    for jf in pred_jsons:
        try:
            data = json.loads(Path(jf).read_text())
        except Exception:
            continue
        img = data.get("image_path")
        if not img:
            continue
        boxes = []
        for b in data.get("boxes") or []:
            if labels and b.get("label") not in labels:
                continue
            if float(b.get("confidence", 0.0)) < conf:
                continue
            xyxy = b.get("bbox_xyxy") or []
            if len(xyxy) == 4:
                boxes.append(tuple(float(v) for v in xyxy))
        if boxes:
            out.setdefault(img, []).extend(boxes)
    return out


def regions_from_fixed(image_paths, rect_norm: tuple) -> dict[str, list[tuple]]:
    """すべての画像に同じ領域を当てる（カメラが固定のとき）。

    rect_norm: (x1, y1, x2, y2) を 0〜1 で指定する。
    画像ごとに大きさが違っても比率で効くようにするため正規化で受ける。
    """
    x1n, y1n, x2n, y2n = rect_norm
    x1n, x2n = sorted((max(0.0, x1n), min(1.0, x2n)))
    y1n, y2n = sorted((max(0.0, y1n), min(1.0, y2n)))

    out: dict[str, list[tuple]] = {}
    for p in image_paths:
        p = Path(p)
        size = _image_size(p)
        if size is None:
            continue
        w, h = size
        out[str(p)] = [(x1n * w, y1n * h, x2n * w, y2n * h)]
    return out


def regions_from_cvat_xml(
    xml_path: Path,
    mask_labels: list[str],
    image_dirs: Optional[list[Path]] = None,
) -> dict[str, list[tuple]]:
    """CVAT の XML から、指定ラベルの領域を集める。

    box はそのまま、polygon / polyline は外接矩形にする
    （モザイクは矩形でかけるため）。
    XML には画像の名前しか入っていないので、image_dirs から実体を探す。
    """
    out: dict[str, list[tuple]] = {}
    try:
        root = ET.parse(Path(xml_path)).getroot()
    except Exception:
        return out

    wanted = set(mask_labels)
    for img_el in root.findall("image"):
        name = img_el.get("name") or ""
        if not name:
            continue
        boxes: list[tuple] = []

        for el in img_el:
            if el.get("label") not in wanted:
                continue
            if el.tag == "box":
                try:
                    boxes.append((float(el.get("xtl")), float(el.get("ytl")),
                                  float(el.get("xbr")), float(el.get("ybr"))))
                except (TypeError, ValueError):
                    continue
            elif el.tag in ("polygon", "polyline"):
                pts = _parse_points(el.get("points") or "")
                if pts:
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    boxes.append((min(xs), min(ys), max(xs), max(ys)))

        if not boxes:
            continue
        resolved = _resolve_image(name, image_dirs or [])
        if resolved:
            out.setdefault(str(resolved), []).extend(boxes)
    return out


def _parse_points(raw: str) -> list[tuple]:
    pts = []
    for pair in raw.split(";"):
        if not pair.strip():
            continue
        try:
            x, y = pair.split(",")
            pts.append((float(x), float(y)))
        except ValueError:
            continue
    return pts


def _resolve_image(name: str, image_dirs: list[Path]) -> Optional[Path]:
    """CVAT の画像名から実ファイルを探す。サブディレクトリ付きの名前にも対応。"""
    base = Path(name).name
    for d in image_dirs:
        d = Path(d)
        if not d.exists():
            continue
        cand = d / name
        if cand.exists():
            return cand
        hits = list(d.rglob(base))
        if hits:
            return hits[0]
    return None


def _image_size(path: Path) -> Optional[tuple]:
    try:
        from PIL import Image
        with Image.open(path) as im:
            return im.size          # (w, h)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# バックアップ
# ---------------------------------------------------------------------------
def backup_root(dataset_dir: Path) -> Path:
    return Path(dataset_dir) / BACKUP_DIR_NAME


def backup_path_for(image_path: Path, dataset_dir: Path) -> Optional[Path]:
    """データセット内の相対位置を保ったまま退避先を決める"""
    try:
        rel = Path(image_path).resolve().relative_to(Path(dataset_dir).resolve())
    except ValueError:
        return None                 # データセットの外にある画像は扱わない
    return backup_root(dataset_dir) / rel


def has_backup(dataset_dir: Path) -> bool:
    root = backup_root(dataset_dir)
    return root.exists() and any(root.rglob("*"))


def count_backup(dataset_dir: Path) -> int:
    root = backup_root(dataset_dir)
    if not root.exists():
        return 0
    return sum(1 for p in root.rglob("*")
               if p.is_file() and p.suffix.lower() in IMG_EXTS)


def restore_mosaic(dataset_dir: Path) -> dict:
    """退避しておいた原本を書き戻す。退避先はそのまま残す。"""
    root = backup_root(dataset_dir)
    if not root.exists():
        return {"ok": False, "restored": 0, "errors": [],
                "error": "退避された原本がありません"}

    restored, errors = 0, []
    for src in sorted(root.rglob("*")):
        if not src.is_file() or src.suffix.lower() not in IMG_EXTS:
            continue
        dst = Path(dataset_dir) / src.relative_to(root)
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            restored += 1
        except Exception as e:
            errors.append((str(dst), str(e)))
    return {"ok": not errors, "restored": restored, "errors": errors,
            "error": f"{len(errors)} 件を戻せませんでした" if errors else ""}


# ---------------------------------------------------------------------------
# 適用
# ---------------------------------------------------------------------------
def apply_mosaic(
    dataset_dir: Path,
    regions_by_image: dict,
    method: str = "pixelate",
    strength: int = 12,
    padding: int = 8,
    dry_run: bool = False,
    on_progress=None,
) -> dict:
    """データセット内の画像にモザイクをかけ、**元の場所へ上書きする**。

    書き換える前に必ず `_backup_original/` へ退避する。
    すでに退避済みのものは上書きしない（2 回目の実行で原本を失わないため）。

    dry_run=True なら何も書かず、対象の件数だけ数える。
    """
    import cv2

    ds = Path(dataset_dir)
    result = {
        "ok": False, "error": "", "images": 0, "regions": 0,
        "backed_up": 0, "skipped": [], "errors": [], "dry_run": dry_run,
    }
    if not ds.exists():
        result["error"] = f"データセットがありません: {ds}"
        return result
    if method not in METHODS:
        result["error"] = f"知らない方式です: {method}"
        return result

    targets = [(Path(p), r) for p, r in regions_by_image.items() if r]
    total = len(targets)
    if total == 0:
        result["error"] = "対象となる領域がありませんでした"
        return result

    for i, (img_path, regions) in enumerate(sorted(targets), 1):
        if on_progress:
            on_progress(i, total)

        if not img_path.exists():
            result["skipped"].append((str(img_path), "画像が見つかりません"))
            continue

        bak = backup_path_for(img_path, ds)
        if bak is None:
            # データセットの外を書き換えると戻せなくなるので触らない
            result["skipped"].append(
                (str(img_path), "このデータセットの外にある画像です"))
            continue

        if dry_run:
            result["images"] += 1
            result["regions"] += len(regions)
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            result["skipped"].append((str(img_path), "画像を読めません"))
            continue

        # ── 退避（すでにあれば触らない）──
        if not bak.exists():
            try:
                bak.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(img_path, bak)
                result["backed_up"] += 1
            except Exception as e:
                # 退避できないものは書き換えない（戻せなくなるため）
                result["errors"].append((str(img_path), f"退避に失敗: {e}"))
                continue

        try:
            masked, applied = mosaic_image(img, regions, method, strength, padding)
            if not cv2.imwrite(str(img_path), masked):
                raise RuntimeError("書き込みに失敗しました")
        except Exception as e:
            result["errors"].append((str(img_path), str(e)))
            continue

        result["images"] += 1
        result["regions"] += applied

    result["ok"] = not result["errors"]
    if result["errors"]:
        result["error"] = f"{len(result['errors'])} 件で問題が起きました"
    return result


def preview_mosaic(image_path: Path, regions, method: str = "pixelate",
                   strength: int = 12, padding: int = 8):
    """適用前に見せる 1 枚。(処理前 RGB, 処理後 RGB, 領域数) を返す。

    実際に書き換える前に必ずこれで確認させること。
    """
    import cv2

    img = cv2.imread(str(image_path))
    if img is None:
        return None
    masked, applied = mosaic_image(img, regions, method, strength, padding)
    return (cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
            cv2.cvtColor(masked, cv2.COLOR_BGR2RGB),
            applied)


def dataset_image_paths(dataset_dir: Path) -> list[Path]:
    """データセット内の画像を集める（退避先は除く）"""
    ds = Path(dataset_dir)
    if not ds.exists():
        return []
    return sorted(
        p for p in ds.rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXTS
        and BACKUP_DIR_NAME not in p.parts
    )


# ---------------------------------------------------------------------------
# 顔検出（YuNet / Haar）
#
#   プライバシー目的で顔を隠すための検出。Ultralytics（AGPL）を通さず、
#   OpenCV だけで完結させている。配布や実機への搭載を考えると
#   ライセンスの制約が無いほうが後々やりやすいため。
#
#   YuNet … MIT。ONNX 1 ファイルだけで動く。こちらが主力
#   Haar  … OpenCV 同梱で追加取得が不要。2001 年の手法で取りこぼしが多いが、
#           ネットワークが使えない環境や、YuNet の保険として併用できる
# ---------------------------------------------------------------------------
FACE_MODELS_DIR = Path(__file__).resolve().parent.parent / ".face_models"

# opencv_zoo（MIT ライセンス）の配布物
YUNET_FILENAME = "face_detection_yunet_2023mar.onnx"
YUNET_URLS = (
    "https://raw.githubusercontent.com/opencv/opencv_zoo/main/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "https://huggingface.co/opencv/face_detection_yunet/resolve/main/"
    "face_detection_yunet_2023mar.onnx",
)

FACE_DETECTORS = {
    "yunet": {
        "label": "YuNet（推奨・MIT）",
        "desc": "OpenCV Zoo の軽量モデル。精度が高く、ライセンスの制約が無い",
    },
    "haar": {
        "label": "Haar カスケード（OpenCV 同梱）",
        "desc": "追加の取得が不要。正面向き以外は取りこぼしやすい",
    },
}


def yunet_model_path() -> Path:
    return FACE_MODELS_DIR / YUNET_FILENAME


def yunet_available() -> bool:
    p = yunet_model_path()
    return p.exists() and p.stat().st_size > 100_000


def download_yunet() -> dict:
    """YuNet の ONNX を取得する。取得済みなら何もしない。

    ネットワークに出るので、利用者が押したときだけ呼ぶこと。
    """
    dst = yunet_model_path()
    if yunet_available():
        return {"ok": True, "path": str(dst), "error": "", "skipped": True}

    import urllib.request

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".part")
    errors = []
    for url in YUNET_URLS:
        try:
            urllib.request.urlretrieve(url, tmp)
            if tmp.stat().st_size < 100_000:
                raise RuntimeError(f"取得したファイルが小さすぎます ({tmp.stat().st_size} バイト)")
            tmp.replace(dst)
            return {"ok": True, "path": str(dst), "error": "", "skipped": False}
        except Exception as e:
            errors.append(f"{url.split('/')[2]}: {e}")
            tmp.unlink(missing_ok=True)
    return {"ok": False, "path": str(dst),
            "error": "取得できませんでした（" + " / ".join(errors) + "）"}


def _detect_faces_yunet(img, conf: float) -> list[tuple]:
    import cv2

    h, w = img.shape[:2]
    det = cv2.FaceDetectorYN.create(
        str(yunet_model_path()), "", (w, h),
        score_threshold=float(conf), nms_threshold=0.3, top_k=5000)
    det.setInputSize((w, h))
    _, faces = det.detect(img)
    if faces is None:
        return []
    # 戻りは [x, y, w, h, 5点のランドマーク..., score]
    return [(float(f[0]), float(f[1]), float(f[0] + f[2]), float(f[1] + f[3]))
            for f in faces]


def _detect_faces_haar(img, min_size: int = 30) -> list[tuple]:
    import cv2

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    out: list[tuple] = []
    # 正面と横顔の両方を当てる（Haar は向きに弱いため）
    for name in ("haarcascade_frontalface_default.xml",
                 "haarcascade_profileface.xml"):
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + name)
        if cascade.empty():
            continue
        for (x, y, w, h) in cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=4,
                minSize=(min_size, min_size)):
            out.append((float(x), float(y), float(x + w), float(y + h)))
    return out


def regions_from_faces(
    image_paths,
    detector: str = "yunet",
    conf: float = 0.6,
    on_progress=None,
) -> dict[str, list[tuple]]:
    """顔を検出して領域を返す。

    conf は YuNet のみ有効（Haar にはスコアの概念が無い）。
    隠す用途なので、呼び出し側は低めの既定値にすること。
    """
    import cv2

    if detector == "yunet" and not yunet_available():
        return {}

    out: dict[str, list[tuple]] = {}
    paths = list(image_paths)
    for i, p in enumerate(paths, 1):
        if on_progress:
            on_progress(i, len(paths))
        img = cv2.imread(str(p))
        if img is None:
            continue
        try:
            boxes = (_detect_faces_yunet(img, conf) if detector == "yunet"
                     else _detect_faces_haar(img))
        except Exception:
            continue
        if boxes:
            out[str(p)] = boxes
    return out


# ---------------------------------------------------------------------------
# アノテーションとの重なり検出
#
#   顔だと判定した場所が、実はアノテーション済みの対象物と重なっていることがある。
#   そのまま隠すと**学習データを壊す**ので、必ず人に確認させる。
#   起きる原因は 2 つ:
#     - 対象物を顔と誤検出した（果実の模様など）
#     - 本当に人と対象物が重なって写っている
#   どちらであっても、機械が勝手に決めてよい話ではない。
# ---------------------------------------------------------------------------
def label_path_for(image_path: Path, dataset_dir: Path) -> Optional[Path]:
    """画像に対応する YOLO ラベル txt を探す。

    `images/train/x.jpg` → `labels/train/x.txt` のほか、
    画像と同じ場所に置く形にも対応する。
    """
    img = Path(image_path)
    ds = Path(dataset_dir)

    # images/... を labels/... に読み替える
    try:
        rel = img.resolve().relative_to(ds.resolve())
        parts = list(rel.parts)
        if "images" in parts:
            parts[parts.index("images")] = "labels"
            cand = ds.joinpath(*parts).with_suffix(".txt")
            if cand.exists():
                return cand
    except ValueError:
        pass

    same = img.with_suffix(".txt")
    return same if same.exists() else None


def overlap_ratio(region: tuple, box: tuple) -> float:
    """region のうち box と重なっている割合（0〜1）。

    IoU ではなく「隠す側が相手をどれだけ覆うか」で見る。
    小さなアノテーションが大きな顔領域に丸ごと飲まれる場合、
    IoU は小さく出てしまい見逃すため。
    """
    ix1, iy1 = max(region[0], box[0]), max(region[1], box[1])
    ix2, iy2 = min(region[2], box[2]), min(region[3], box[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    box_area = max(1e-9, (box[2] - box[0]) * (box[3] - box[1]))
    return inter / box_area


def find_annotation_conflicts(
    regions_by_image: dict,
    dataset_dir: Path,
    class_names: Optional[list[str]] = None,
    min_overlap: float = 0.01,
    padding: int = 0,
) -> list[dict]:
    """隠そうとしている領域が、アノテーション済みの物体と重なっていないか調べる。

    戻り値は重なった組の一覧。空なら衝突なし。
    padding には実際に適用する余白と同じ値を渡すこと
    （余白ぶん広がったところで初めて重なる場合があるため）。
    """
    from .dataset import _yolo_txt_to_xyxy

    ds = Path(dataset_dir)
    names = class_names or []
    conflicts: list[dict] = []

    for img_str, regions in regions_by_image.items():
        img_path = Path(img_str)
        lbl = label_path_for(img_path, ds)
        if lbl is None:
            continue
        size = _image_size(img_path)
        if size is None:
            continue
        w, h = size

        anns = _yolo_txt_to_xyxy(lbl, w, h, names)
        if not anns:
            continue

        for region in regions:
            padded = (region[0] - padding, region[1] - padding,
                      region[2] + padding, region[3] + padding)
            for ann in anns:
                box = ann.get("bbox_xyxy") or []
                if len(box) != 4:
                    continue
                ratio = overlap_ratio(padded, tuple(box))
                if ratio >= min_overlap:
                    conflicts.append({
                        "image": str(img_path),
                        "region": tuple(float(v) for v in region),
                        "annotation": tuple(float(v) for v in box),
                        "label": ann.get("label") or "（クラス不明）",
                        "covered": ratio,
                    })
    return conflicts


def merge_regions(*region_maps) -> dict[str, list[tuple]]:
    """複数の領域の集まりを 1 つにまとめる（顔と対象物を同時に隠す場合など）"""
    out: dict[str, list[tuple]] = {}
    for m in region_maps:
        for img, regions in (m or {}).items():
            for r in regions:
                out.setdefault(img, [])
                if r not in out[img]:
                    out[img].append(r)
    return out


def applied_previews(dataset_dir: Path, limit: int = 12) -> list[dict]:
    """適用済みの結果を確認するための一覧を作る。

    退避してある原本と、いまの（隠したあとの）画像を対にして返す。
    隠し漏れが無いかを後から見直すためのもの。
    """
    ds = Path(dataset_dir)
    root = backup_root(ds)
    if not root.exists():
        return []

    out: list[dict] = []
    for bak in sorted(root.rglob("*")):
        if not bak.is_file() or bak.suffix.lower() not in IMG_EXTS:
            continue
        cur = ds / bak.relative_to(root)
        if not cur.exists():
            continue
        out.append({"name": cur.name, "current": str(cur), "original": str(bak)})
        if len(out) >= limit:
            break
    return out
