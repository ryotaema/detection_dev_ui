# =============================================================================
# MLOps 統合UI - main.py
# Streamlit + CVAT API + YOLO学習 + ClearML + FiftyOne
# =============================================================================
from __future__ import annotations

import json
import os
import threading
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import streamlit as st


@st.cache_resource
def _get_train_shared() -> tuple[dict, threading.Lock]:
    """st.rerun() をまたいで同一オブジェクトを保持する共有状態。
    Streamlit はスクリプトを再実行するたびにモジュール変数を再初期化するため、
    st.cache_resource でキャッシュして常に同一インスタンスを返す。
    """
    return (
        {"log": [], "progress": 0, "running": False, "error": None, "model_path": None},
        threading.Lock(),
    )

# ---------------------------------------------------------------------------
# 定数・環境変数
# ---------------------------------------------------------------------------
DATA_DIR       = Path(os.getenv("DATA_DIR",       "/workspace/data"))
MODELS_DIR     = Path(os.getenv("MODELS_DIR",     "/workspace/models"))
PREDICTIONS_DIR= Path(os.getenv("PREDICTIONS_DIR","/workspace/predictions"))
CVAT_HOST      = os.getenv("CVAT_HOST",  "http://cvat-server:8080")
CVAT_USER      = os.getenv("CVAT_USERNAME","admin")
CVAT_PASS      = os.getenv("CVAT_PASSWORD","admin")
CLEARML_API    = os.getenv("CLEARML_API_HOST","http://clearml_apiserver:8008")
CLEARML_WEB    = os.getenv("CLEARML_WEB_HOST","http://localhost:8082")  # Fix: 8080→8082
FIFTYONE_PORT  = int(os.getenv("FIFTYONE_PORT","5151"))

for d in [DATA_DIR, MODELS_DIR, PREDICTIONS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Streamlit ページ設定
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="MLOps Pipeline",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# st.set_page_config() の後で取得（cache_resource の初回呼び出しが安全なタイミング）
_train_state, _train_log_lock = _get_train_shared()

# ---------------------------------------------------------------------------
# カスタム CSS (ダークテーマ / インダストリアル)
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}
code, pre, .stCode { font-family: 'JetBrains Mono', monospace; }

.stApp { background: #0d0f14; }

/* サイドバー */
[data-testid="stSidebar"] {
    background: #12151c;
    border-right: 1px solid #1e2330;
}

/* カード */
.pipeline-card {
    background: #161b26;
    border: 1px solid #1e2330;
    border-radius: 8px;
    padding: 20px;
    margin: 10px 0;
}
.pipeline-card h3 { color: #7ecff4; margin-top: 0; }

/* ステータスバッジ */
.badge-ok   { background:#1a3a2a; color:#4caf7d; border:1px solid #2d6b47;
                padding:2px 10px; border-radius:4px; font-size:.78rem; }
.badge-warn { background:#3a2a10; color:#f0a830; border:1px solid #7a5520;
                padding:2px 10px; border-radius:4px; font-size:.78rem; }
.badge-err  { background:#3a1a1a; color:#f06060; border:1px solid #7a3030;
                padding:2px 10px; border-radius:4px; font-size:.78rem; }

/* ログエリア */
.log-area {
    background: #0a0c10;
    border: 1px solid #1e2330;
    border-radius: 6px;
    padding: 12px;
    font-family: 'JetBrains Mono', monospace;
    font-size: .78rem;
    color: #8fb8d0;
    max-height: 520px;
    overflow-y: auto;
    white-space: pre-wrap;
    word-break: break-all;
}

/* プログレスバー */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #2d7dd2, #7ecff4);
}

/* ボタン */
.stButton > button {
    background: #1a2540;
    color: #7ecff4;
    border: 1px solid #2d4a80;
    border-radius: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    letter-spacing: .05em;
    transition: all .2s;
}
.stButton > button:hover {
    background: #2d4a80;
    border-color: #7ecff4;
    color: #fff;
}
</style>
""", unsafe_allow_html=True)

# ===========================================================================
# セッションステート初期化
# ===========================================================================
defaults = {
    "training_log": [],
    "training_running": False,
    "training_progress": 0,
    "training_error": None,
    "fiftyone_session": None,
    "fiftyone_port": None,
    "last_model_path": None,
    "cvat_tasks": [],
    "cvat_xml_info": None,   # 解析済みXMLメタ情報（ラベル・アノテーション種別）
    "cvat_raw_dir": None,    # CVATエクスポートRAWデータのディレクトリ
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ===========================================================================
# ヘルパー関数群
# ===========================================================================

# ---------------------------------------------------------------------------
# CVAT API クライアント
# ---------------------------------------------------------------------------
def get_cvat_client():
    """cvat-sdk の CvatClient を返す（接続失敗時は None）"""
    try:
        from cvat_sdk import make_client
        client = make_client(
            host=CVAT_HOST,
            credentials=(CVAT_USER, CVAT_PASS),
        )
        return client
    except Exception as e:
        st.error(f"CVAT接続エラー: {e}")
        return None


def fetch_cvat_tasks() -> list[dict]:
    """CVATのタスク一覧を取得する"""
    client = get_cvat_client()
    if not client:
        return []
    try:
        tasks = client.tasks.list()
        return [{"id": t.id, "name": t.name, "size": t.size,
                "status": t.status} for t in tasks]
    except Exception as e:
        st.error(f"タスク取得エラー: {e}")
        return []


def export_cvat_task_raw(task_id: int, out_dir: Path) -> Optional[Path]:
    """指定タスクを「CVAT for images 1.1」(XML形式) でエクスポートし、
    out_dir/raw/ に解凍したパスを返す。
    CVAT v2.64.0 の非同期エクスポート API (POST→ポーリング→ダウンロード) を使用。
    """
    import requests as _requests

    raw_dir  = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / "cvat_export.zip"
    session  = _requests.Session()

    try:
        login = session.post(
            f"{CVAT_HOST}/api/auth/login",
            json={"username": CVAT_USER, "password": CVAT_PASS},
        )
        login.raise_for_status()
        token = login.json().get("key")
        session.headers.update({"Authorization": f"Token {token}"})

        export = session.post(
            f"{CVAT_HOST}/api/tasks/{task_id}/dataset/export",
            params={
                "save_images": "True",
                "format": "CVAT for images 1.1",
            },
        )
        export.raise_for_status()
        rq_id = export.json().get("rq_id")
        if not rq_id:
            st.error("エクスポートジョブID が取得できませんでした")
            return None

        result_url = None
        for _ in range(180):
            status_resp = session.get(f"{CVAT_HOST}/api/requests/{rq_id}")
            status_resp.raise_for_status()
            data = status_resp.json()
            status = data.get("status")
            if status == "finished":
                result_url = data.get("result_url")
                break
            elif status == "failed":
                st.error(f"エクスポートに失敗しました: {data}")
                return None
            time.sleep(1)
        else:
            st.error("エクスポートがタイムアウトしました（180秒）")
            return None

        dl = session.get(result_url, stream=True)
        dl.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in dl.iter_content(chunk_size=8192):
                f.write(chunk)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(raw_dir)
        zip_path.unlink()
        return raw_dir

    except Exception as e:
        st.error(f"CVATからのエクスポート中にエラーが発生しました: {e}")
        return None


def parse_cvat_xml(raw_dir: Path) -> Optional[dict]:
    """CVAT for images 1.1 のXMLを解析してメタ情報を返す。

    Returns:
        {
          "xml_path": str,
          "labels": [str, ...],           # タスク定義のラベル一覧
          "annotation_types": [str, ...], # 実際に使われている種別 (box/polygon/points)
          "image_count": int,
          "annotated_count": int,         # 1件以上アノテーション付きの画像数
        }
    """
    import xml.etree.ElementTree as ET

    xml_candidates = list(raw_dir.glob("**/*.xml"))
    if not xml_candidates:
        st.error("XMLファイルが見つかりません")
        return None

    xml_path = xml_candidates[0]
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # タスク定義のラベル一覧
        labels: list[str] = []
        for lbl in root.findall(".//meta/task/labels/label"):
            name = lbl.find("name")
            if name is not None and name.text:
                labels.append(name.text.strip())

        # 画像・アノテーション統計
        annotation_types: set[str] = set()
        image_count = 0
        annotated_count = 0
        for img in root.findall("image"):
            image_count += 1
            has_annot = False
            for child in img:
                if child.tag in ("box", "polygon", "polyline", "points", "ellipse"):
                    annotation_types.add(child.tag)
                    has_annot = True
            if has_annot:
                annotated_count += 1

        return {
            "xml_path": str(xml_path),
            "labels": labels,
            "annotation_types": sorted(annotation_types),
            "image_count": image_count,
            "annotated_count": annotated_count,
        }
    except Exception as e:
        st.error(f"XML解析エラー: {e}")
        return None


def generate_yolo_dataset(
    raw_dir: Path,
    xml_info: dict,
    selected_labels: list[str],
    task_type: str,
    out_dir: Path,
    val_ratio: float = 0.2,
) -> Optional[Path]:
    """選択ラベル × タスク種別で YOLO 形式データセットを生成する。

    - 画像は raw_dir 内から探してシンボリックリンクを張る（大容量でもコピー不要）
    - train/val は annotated サンプルを 80/20 でランダム分割
    - data.yaml は絶対パス + names リスト形式で生成
    """
    import xml.etree.ElementTree as ET
    import random
    import yaml

    label2id = {lbl: i for i, lbl in enumerate(selected_labels)}
    xml_path = Path(xml_info["xml_path"])

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 画像ディレクトリを探す（ZIPの構造に依存するため複数候補）
    img_roots: list[Path] = []
    for d in raw_dir.rglob("*"):
        if d.is_dir() and d.name in ("images", "train", "val"):
            img_roots.append(d)
    if not img_roots:
        img_roots = [raw_dir]

    def _find_image(name: str) -> Optional[Path]:
        for base in img_roots:
            p = base / name
            if p.exists():
                return p
        for p in raw_dir.rglob(Path(name).name):
            if p.is_file():
                return p
        return None

    def _box_to_detect(box, w: int, h: int) -> str:
        xtl, ytl = float(box.get("xtl", 0)), float(box.get("ytl", 0))
        xbr, ybr = float(box.get("xbr", 0)), float(box.get("ybr", 0))
        cx = (xtl + xbr) / 2 / w
        cy = (ytl + ybr) / 2 / h
        bw = (xbr - xtl) / w
        bh = (ybr - ytl) / h
        return f"{cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"

    def _polygon_to_segment(polygon, w: int, h: int) -> str:
        pts = []
        for pt in polygon.get("points", "").split(";"):
            pt = pt.strip()
            if "," in pt:
                x, y = pt.split(",")
                pts.append(f"{float(x)/w:.6f} {float(y)/h:.6f}")
        return " ".join(pts)

    samples: list[dict] = []
    for img_elem in root.findall("image"):
        img_name = img_elem.get("name", "")
        w = int(img_elem.get("width", 1))
        h = int(img_elem.get("height", 1))
        lines: list[str] = []

        if task_type == "detect":
            for box in img_elem.findall("box"):
                lbl = box.get("label", "")
                if lbl not in label2id:
                    continue
                lines.append(f"{label2id[lbl]} {_box_to_detect(box, w, h)}")

        elif task_type == "segment":
            for polygon in img_elem.findall("polygon"):
                lbl = polygon.get("label", "")
                if lbl not in label2id:
                    continue
                seg = _polygon_to_segment(polygon, w, h)
                if seg:
                    lines.append(f"{label2id[lbl]} {seg}")
            for box in img_elem.findall("box"):
                lbl = box.get("label", "")
                if lbl not in label2id:
                    continue
                xtl, ytl = float(box.get("xtl", 0)), float(box.get("ytl", 0))
                xbr, ybr = float(box.get("xbr", 0)), float(box.get("ybr", 0))
                pts = " ".join([
                    f"{xtl/w:.6f} {ytl/h:.6f}",
                    f"{xbr/w:.6f} {ytl/h:.6f}",
                    f"{xbr/w:.6f} {ybr/h:.6f}",
                    f"{xtl/w:.6f} {ybr/h:.6f}",
                ])
                lines.append(f"{label2id[lbl]} {pts}")

        elif task_type == "pose":
            for box in img_elem.findall("box"):
                lbl = box.get("label", "")
                if lbl not in label2id:
                    continue
                lines.append(f"{label2id[lbl]} {_box_to_detect(box, w, h)}")

        if lines:
            samples.append({"name": img_name, "lines": lines})

    if not samples:
        st.error("選択したラベルにマッチするアノテーションがありません")
        return None

    random.shuffle(samples)
    split = max(1, int(len(samples) * (1 - val_ratio)))
    splits = {"train": samples[:split], "val": samples[split:]}

    for sp in ("train", "val"):
        (out_dir / "images" / sp).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / sp).mkdir(parents=True, exist_ok=True)

    for sp, sp_samples in splits.items():
        for s in sp_samples:
            img_src = _find_image(s["name"])
            if img_src is None:
                continue
            stem = Path(s["name"]).stem
            img_dst = out_dir / "images" / sp / img_src.name
            lbl_dst = out_dir / "labels" / sp / f"{stem}.txt"
            if not img_dst.exists():
                try:
                    img_dst.symlink_to(img_src.resolve())
                except Exception:
                    import shutil
                    shutil.copy2(img_src, img_dst)
            with open(lbl_dst, "w") as f:
                f.write("\n".join(s["lines"]))

    cfg = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(selected_labels),
        "names": selected_labels,
    }
    with open(out_dir / "data.yaml", "w") as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)

    return out_dir


# ---------------------------------------------------------------------------
# ClearML 設定
# ---------------------------------------------------------------------------
def init_clearml(project_name: str, task_name: str):
    """ClearML Task を初期化して返す。失敗時は None（学習は継続）。
    バックグラウンドスレッドから呼ばれるため st.* は使用しない。
    ACCESS_KEY が未設定の場合はスキップ（接続待ちで詰まるのを防ぐ）。
    """
    access_key = os.getenv("CLEARML_API_ACCESS_KEY", "").strip()
    if not access_key:
        print("[ClearML] ACCESS_KEY 未設定のためスキップ")
        return None
    try:
        from clearml import Task
        os.environ["CLEARML_API_HOST"] = CLEARML_API
        os.environ["CLEARML_WEB_HOST"] = CLEARML_WEB
        task = Task.init(
            project_name=project_name,
            task_name=task_name,
            reuse_last_task_id=False,
        )
        return task
    except Exception as e:
        print(f"[ClearML] 初期化エラー（実験追跡なし）: {e}")
        return None


# ---------------------------------------------------------------------------
# YOLO 学習ワーカー (別スレッドで実行)
# ---------------------------------------------------------------------------

class _StdoutCapture:
    """sys.stdout を乗っ取り、YOLO の print 出力を _train_state["log"] に転送する。
    元の stdout にも同時に書くので docker logs でも確認できる。
    """
    def __init__(self, original, lock: threading.Lock, state: dict) -> None:
        self._orig  = original
        self._lock  = lock
        self._state = state
        self._buf   = ""

    def write(self, text: str) -> int:
        self._orig.write(text)
        self._buf += text
        # 改行単位で確定させる
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            line = line.rstrip()
            if line:
                with self._lock:
                    self._state["log"].append(line)
        return len(text)

    def flush(self) -> None:
        self._orig.flush()

    def fileno(self) -> int:
        return self._orig.fileno()


def _train_worker(
    data_yaml: str,
    model_name: str,
    epochs: int,
    batch_size: int,
    project_name: str,
    run_name: str,
    train_kwargs: dict,
):
    """バックグラウンドスレッドで YOLO 学習を実行する。
    sys.stdout を _StdoutCapture に差し替えて全 print 出力を UI に転送する。
    st.session_state はスレッド外から参照不可のため、_train_state 経由で通信する。
    train_kwargs は model.train() に **kwargs として渡す追加パラメータ。
    """
    import sys

    def _log(msg: str) -> None:
        with _train_log_lock:
            _train_state["log"].append(msg)

    def _on_epoch_end(trainer) -> None:
        cur   = trainer.epoch + 1
        total = trainer.epochs
        with _train_log_lock:
            _train_state["progress"] = int(cur / total * 95)

    _orig_stdout = sys.stdout
    sys.stdout   = _StdoutCapture(_orig_stdout, _train_log_lock, _train_state)

    try:
        clearml_task = init_clearml(project_name, run_name)
        if clearml_task:
            clearml_task.connect({
                "model": model_name,
                "epochs": epochs,
                "batch_size": batch_size,
                "data_yaml": data_yaml,
                **train_kwargs,
            })
            _log(f"[ClearML] タスク開始: {clearml_task.id}")
        else:
            _log("[ClearML] スキップ（実験追跡なし）")

        from ultralytics import YOLO

        model = YOLO(model_name)
        model.add_callback("on_train_epoch_end", _on_epoch_end)

        results = model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            project=str(MODELS_DIR),
            name=run_name,
            exist_ok=True,
            **train_kwargs,
        )

        best_model = Path(results.save_dir) / "weights" / "best.pt"
        with _train_log_lock:
            _train_state["model_path"] = str(best_model)
            _train_state["progress"]   = 100
        _log(f"[完了] best.pt: {best_model}")

        if clearml_task:
            clearml_task.upload_artifact("best_model", str(best_model))
            clearml_task.close()

    except Exception as e:
        _log(f"[ERROR] {e}")
        with _train_log_lock:
            _train_state["error"] = str(e)

    finally:
        sys.stdout = _orig_stdout
        with _train_log_lock:
            _train_state["running"] = False


# ---------------------------------------------------------------------------
# 画像ディレクトリスキャン
# ---------------------------------------------------------------------------
def _find_image_dirs(base_dir: Path, max_depth: int = 4) -> list[Path]:
    """base_dir 以下で画像ファイルが1件以上あるディレクトリを返す。
    シンボリックリンク先も辿る。深さは max_depth で制限。
    """
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    result: list[Path] = []
    base_depth = len(base_dir.parts)
    for root, dirs, files in os.walk(str(base_dir), followlinks=True):
        root_path = Path(root)
        if len(root_path.parts) - base_depth > max_depth:
            dirs.clear()
            continue
        dirs.sort()
        if any(Path(f).suffix.lower() in img_exts for f in files):
            result.append(root_path)
    return result


# ---------------------------------------------------------------------------
# FiftyOne セッション管理
# ---------------------------------------------------------------------------
def launch_fiftyone(dataset_name: str, predictions_dir: Path) -> Optional[int]:
    """
    FiftyOne データセットを作成し、Appを起動してポート番号を返す。
    既存のセッションがあれば再利用。

    Fix: remote=True → remote=False, address="0.0.0.0"
        コンテナ内で 0.0.0.0:5151 でListenさせてホストブラウザからアクセス可能にする。
    """
    try:
        import fiftyone as fo

        # 既存データセットをリセット
        if fo.dataset_exists(dataset_name):
            fo.delete_dataset(dataset_name)

        dataset = fo.Dataset(name=dataset_name)

        # predictions_dir の JSON ファイルを読み込んでサンプル追加
        json_files = list(predictions_dir.glob("*.json"))
        if not json_files:
            st.warning("predictions/ に結果JSONがありません。先に推論を実行してください。")
            return None

        samples = []
        for jf in json_files:
            with open(jf) as f:
                pred = json.load(f)

            img_path = pred.get("image_path", "")
            detections = []
            for box in pred.get("boxes", []):
                detections.append(
                    fo.Detection(
                        label=box["label"],
                        bounding_box=box["bbox_xywhn"],  # [x, y, w, h] 正規化済
                        confidence=box.get("confidence", 1.0),
                    )
                )
            sample = fo.Sample(filepath=img_path)
            sample["predictions"] = fo.Detections(detections=detections)
            samples.append(sample)

        dataset.add_samples(samples)

        # 既存セッションを閉じる
        if st.session_state.fiftyone_session:
            try:
                st.session_state.fiftyone_session.close()
            except Exception:
                pass

        # Fix: remote=False, address="0.0.0.0" でコンテナ外から直接アクセス可能に
        session = fo.launch_app(
            dataset,
            port=FIFTYONE_PORT,
            address="0.0.0.0",
            remote=False,
        )
        st.session_state.fiftyone_session = session
        st.session_state.fiftyone_port = FIFTYONE_PORT
        return FIFTYONE_PORT

    except Exception as e:
        st.error(f"FiftyOne エラー: {e}")
        return None


# ---------------------------------------------------------------------------
# YOLO 推論
# ---------------------------------------------------------------------------
def run_inference(
    model_path: str,
    image_dir: Path,
    out_dir: Path,
    conf_threshold: float = 0.25,
) -> list[Path]:
    """指定モデルで画像フォルダを推論し、結果JSONを out_dir に保存して返す"""
    try:
        from ultralytics import YOLO

        model = YOLO(model_path)
        results_list = model.predict(
            source=str(image_dir),
            conf=conf_threshold,
            save=False,
        )
        saved_jsons = []
        for res in results_list:
            img_path = res.path
            boxes = []
            if res.boxes:
                for box in res.boxes:
                    xyxy   = box.xyxy[0].tolist()
                    xywhn  = box.xywhn[0].tolist()
                    cls_id = int(box.cls[0])
                    conf   = float(box.conf[0])
                    label  = res.names[cls_id]
                    boxes.append({
                        "label": label,
                        "confidence": round(conf, 4),
                        "bbox_xyxy": [round(v, 2) for v in xyxy],
                        "bbox_xywhn": [round(v, 6) for v in xywhn],
                    })

            out_json = out_dir / (Path(img_path).stem + ".json")
            with open(out_json, "w") as f:
                json.dump({"image_path": img_path, "boxes": boxes}, f, indent=2, ensure_ascii=False)
            saved_jsons.append(out_json)

        return saved_jsons
    except Exception as e:
        st.error(f"推論エラー: {e}")
        return []


# ===========================================================================
# UI ヘルパー：ポップオーバー付きウィジェット
# ===========================================================================
_DOC_TRAIN = "https://docs.ultralytics.com/modes/train/#train-settings"
_DOC_AUG   = "https://docs.ultralytics.com/modes/train/#augmentation-settings-and-hyperparameters"


def _ph(name: str, desc: str, url: str) -> None:
    """ポップオーバー形式のパラメータヘルプボタン（❓）を描画する。"""
    with st.popover("❓", use_container_width=True):
        st.markdown(f"**`{name}`**\n\n{desc}")
        st.markdown(f"[📖 Ultralytics ドキュメント]({url})")


def _sw(label: str, lo: float, hi: float, val: float, step: float,
        name: str, desc: str, url: str = _DOC_AUG, **kw) -> float:
    """slider + ❓ popover を [5,1] カラムで横並び表示して値を返す。"""
    c, h = st.columns([5, 1])
    with c:
        v = st.slider(label, lo, hi, val, step=step, **kw)
    with h:
        st.markdown('<div style="margin-top:28px"></div>', unsafe_allow_html=True)
        _ph(name, desc, url)
    return v


def _nw(label: str, lo: float, hi: float, val: float,
        name: str, desc: str, url: str = _DOC_TRAIN, **kw):
    """number_input + ❓ popover を [5,1] カラムで横並び表示して値を返す。"""
    c, h = st.columns([5, 1])
    with c:
        v = st.number_input(label, lo, hi, val, **kw)
    with h:
        st.markdown('<div style="margin-top:28px"></div>', unsafe_allow_html=True)
        _ph(name, desc, url)
    return v


def _selw(label: str, options: list, idx: int,
          name: str, desc: str, url: str = _DOC_TRAIN, **kw) -> str:
    """selectbox + ❓ popover を [5,1] カラムで横並び表示して値を返す。"""
    c, h = st.columns([5, 1])
    with c:
        v = st.selectbox(label, options, index=idx, **kw)
    with h:
        st.markdown('<div style="margin-top:28px"></div>', unsafe_allow_html=True)
        _ph(name, desc, url)
    return v


def _ckw(label: str, val: bool,
         name: str, desc: str, url: str = _DOC_TRAIN, **kw) -> bool:
    """checkbox + ❓ popover を [5,1] カラムで横並び表示して値を返す。"""
    c, h = st.columns([5, 1])
    with c:
        v = st.checkbox(label, value=val, **kw)
    with h:
        _ph(name, desc, url)
    return v


# ===========================================================================
# UI レイアウト
# ===========================================================================

# --- ヘッダー ---
st.markdown("""
<div style="border-bottom:1px solid #1e2330; padding-bottom:16px; margin-bottom:24px;">
  <h1 style="color:#7ecff4; font-family:'JetBrains Mono',monospace; font-size:1.6rem; margin:0;">
    🔬 MLOps Pipeline
  </h1>
  <p style="color:#4a6080; font-size:.85rem; margin:4px 0 0;">
    CVAT → YOLO → ClearML → FiftyOne 統合ダッシュボード
  </p>
</div>
""", unsafe_allow_html=True)

# --- サイドバー: サービス接続状態 ---
with st.sidebar:
    st.markdown("### 🖥 サービス状態")

    def check_service(url: str, name: str):
        import requests
        try:
            r = requests.get(url, timeout=3)
            ok = r.status_code < 500
        except Exception:
            ok = False
        badge = "badge-ok" if ok else "badge-err"
        status = "ONLINE" if ok else "OFFLINE"
        st.markdown(
            f'<div style="margin:4px 0">{name} '
            f'<span class="{badge}">{status}</span></div>',
            unsafe_allow_html=True,
        )

    check_service(f"{CVAT_HOST}/api/server/about", "CVAT")
    check_service(f"{CLEARML_API}/debug.ping", "ClearML API")
    check_service(f"{CLEARML_WEB}", "ClearML WebUI")

    st.markdown("---")
    st.markdown("#### 📁 ディレクトリ")
    st.code(f"data/        {DATA_DIR}\nmodels/      {MODELS_DIR}\npredictions/ {PREDICTIONS_DIR}", language="text")

    st.markdown("---")
    st.markdown("#### 🔗 クイックリンク")
    st.markdown(f"[📝 CVAT UI]({CVAT_HOST})", unsafe_allow_html=False)
    st.markdown(f"[📊 ClearML UI]({CLEARML_WEB})", unsafe_allow_html=False)
    if st.session_state.fiftyone_port:
        fo_url = f"http://localhost:{st.session_state.fiftyone_port}"
        st.markdown(f"[🔭 FiftyOne App]({fo_url})", unsafe_allow_html=False)

# ---------------------------------------------------------------------------
# タブ構成
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "① CVAT エクスポート",
    "② YOLO 学習",
    "③ 推論 & 可視化",
    "④ データ管理",
])

# ===========================================================================
# タブ1: CVAT エクスポート
# ===========================================================================
with tab1:
    st.markdown('<div class="pipeline-card"><h3>📤 CVATタスクエクスポート</h3>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🔄 タスク一覧を取得", use_container_width=True):
            with st.spinner("CVATからタスクを取得中…"):
                st.session_state.cvat_tasks = fetch_cvat_tasks()

    tasks = st.session_state.cvat_tasks
    if not tasks:
        st.info("「タスク一覧を取得」ボタンを押してCVATに接続してください。")
    else:
        task_options = {f"[{t['id']}] {t['name']} ({t['size']} items)": t["id"] for t in tasks}
        selected_label = st.selectbox("エクスポートするタスクを選択", list(task_options.keys()))
        selected_id = task_options[selected_label]

        export_dir_name = st.text_input(
            "エクスポート先サブディレクトリ名",
            value=f"dataset_{selected_id}_{datetime.now():%Y%m%d}",
        )

        # ─── Step 1: CVAT for images 1.1 エクスポート ───────────────────────
        st.markdown("#### Step 1: CVATエクスポート")
        if st.button("⬇️ エクスポート実行 (CVAT for images 1.1)", type="primary",
                     use_container_width=True):
            out_dir = DATA_DIR / export_dir_name
            out_dir.mkdir(parents=True, exist_ok=True)
            with st.spinner("エクスポート中…（最大3分）"):
                raw_dir = export_cvat_task_raw(selected_id, out_dir)
            if raw_dir:
                st.session_state.cvat_raw_dir = str(raw_dir)
                st.session_state.cvat_xml_info = None
                st.success(f"✅ エクスポート完了: `{raw_dir}`")
                xml_info = parse_cvat_xml(raw_dir)
                if xml_info:
                    st.session_state.cvat_xml_info = xml_info

        # ─── Step 2: ラベル・タスク種別の設定 ───────────────────────────────
        if st.session_state.cvat_raw_dir and st.session_state.cvat_xml_info:
            xml_info = st.session_state.cvat_xml_info
            st.markdown("---")
            st.markdown("#### Step 2: ラベルとタスク種別の設定")

            col_stat1, col_stat2, col_stat3 = st.columns(3)
            with col_stat1:
                st.metric("総画像数", xml_info["image_count"])
            with col_stat2:
                st.metric("アノテーション付き", xml_info["annotated_count"])
            with col_stat3:
                ann_type_str = ", ".join(xml_info["annotation_types"]) or "なし"
                st.metric("アノテーション種別", ann_type_str)

            selected_labels = st.multiselect(
                "学習に使用するラベルを選択（順番がクラスID順になります）",
                options=xml_info["labels"],
                default=xml_info["labels"],
            )

            ann_types = set(xml_info.get("annotation_types", []))
            task_type_options = ["detect"]
            if "polygon" in ann_types:
                task_type_options.append("segment")
            if "points" in ann_types:
                task_type_options.append("pose")

            col_task, col_val = st.columns(2)
            with col_task:
                task_type = st.selectbox(
                    "タスク種別",
                    task_type_options,
                    help="detect: バウンディングボックス / segment: ポリゴン（box→矩形ポリゴンに変換） / pose: キーポイント",
                )
            with col_val:
                val_ratio = st.slider("バリデーション割合", 0.05, 0.40, 0.20, step=0.05)

            # ─── Step 3: データセット生成 ────────────────────────────────────
            st.markdown("---")
            st.markdown("#### Step 3: データセット生成")

            gen_dir_name = st.text_input(
                "生成先ディレクトリ名",
                value=f"yolo_{task_type}_{datetime.now():%Y%m%d_%H%M}",
            )

            if not selected_labels:
                st.warning("少なくとも1つ以上のラベルを選択してください。")
            else:
                if st.button("⚙️ データセット生成", type="primary", use_container_width=True):
                    raw_dir_path = Path(st.session_state.cvat_raw_dir)
                    gen_dir = DATA_DIR / gen_dir_name
                    gen_dir.mkdir(parents=True, exist_ok=True)
                    with st.spinner("YOLOデータセットを生成中…"):
                        result = generate_yolo_dataset(
                            raw_dir=raw_dir_path,
                            xml_info=xml_info,
                            selected_labels=selected_labels,
                            task_type=task_type,
                            out_dir=gen_dir,
                            val_ratio=val_ratio,
                        )
                    if result:
                        yaml_path = result / "data.yaml"
                        st.success("✅ データセット生成完了！")
                        st.info(
                            f"🗂 data.yaml パス（タブ②にコピーして使用）:\n`{yaml_path}`"
                        )
                        st.code(str(yaml_path), language="text")

    st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("💡 既にRAWデータがある場合（再解析）"):
        manual_raw = st.text_input(
            "既存のraw_dirパス（コンテナ内）",
            placeholder="/workspace/data/dataset_11_20260512/raw",
        )
        if st.button("🔍 XMLを解析", use_container_width=True) and manual_raw:
            raw_p = Path(manual_raw)
            if raw_p.exists():
                xml_info = parse_cvat_xml(raw_p)
                if xml_info:
                    st.session_state.cvat_raw_dir = str(raw_p)
                    st.session_state.cvat_xml_info = xml_info
                    st.success("解析完了。Step 2が表示されます。")
                    st.rerun()
            else:
                st.error(f"ディレクトリが存在しません: {raw_p}")


# ===========================================================================
# タブ2: YOLO 学習
# ===========================================================================
with tab2:
    st.markdown('<div class="pipeline-card"><h3>🚀 YOLO 学習設定</h3>', unsafe_allow_html=True)

    # ── 基本設定 ────────────────────────────────────────────────────────────
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        _model_preset = st.selectbox(
            "モデル",
            ["yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x",
             "yolo11n-seg", "yolo11s-seg", "yolo11m-seg", "yolo11l-seg", "yolo11x-seg",
             "yolo11n-pose", "yolo11s-pose", "yolo11m-pose", "yolo11l-pose", "yolo11x-pose",
             "カスタム入力"],
            index=1,
        )
    with col_b:
        epochs = st.number_input("エポック数", min_value=1, max_value=5000, value=100, step=10)
    with col_c:
        batch_size = st.select_slider(
            "バッチサイズ",
            options=[-1, 4, 8, 16, 32, 64, 128],
            value=8,
            help="-1 = AutoBatch",
        )

    if _model_preset == "カスタム入力":
        model_name = st.text_input("モデルファイル名 (.pt)", value="yolo11x.pt",
                                   help="例: yolo11x.pt, rtdetr-x.pt")
    else:
        model_name = f"{_model_preset}.pt"
        st.code(f"モデル: {model_name}", language="text")

    # data.yaml 選択（最終更新順に列挙）
    _yaml_candidates = sorted(
        DATA_DIR.rglob("data.yaml"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    _yaml_labels = [str(p.relative_to(DATA_DIR)) for p in _yaml_candidates]
    _YAML_MANUAL = "（手動入力）"
    _yaml_options = _yaml_labels + [_YAML_MANUAL]
    _yaml_sel = st.selectbox(
        "data.yaml",
        _yaml_options,
        index=0 if _yaml_labels else len(_yaml_options) - 1,
    )
    if _yaml_sel == _YAML_MANUAL:
        data_yaml_path = st.text_input(
            "data.yaml パスを直接入力 (コンテナ内絶対パス)",
            value=str(DATA_DIR / "dataset/data.yaml"),
        )
    else:
        data_yaml_path = str(DATA_DIR / _yaml_sel)
        st.code(data_yaml_path, language="text")

    col_p, col_q = st.columns(2)
    with col_p:
        clearml_project = st.text_input("ClearML プロジェクト名", value="YOLO-Detection")
    with col_q:
        run_name = st.text_input(
            "ラン名",
            value=f"{_model_preset.replace('カスタム入力','custom')}_ep{epochs}_{datetime.now():%H%M}",
        )

    # ── 学習設定（最適化・正則化）────────────────────────────────────────────
    with st.expander("⚙️ 学習設定（最適化・正則化）", expanded=False):
        _oc1, _oc2, _oc3, _oc4 = st.columns(4)
        with _oc1:
            imgsz       = _nw("imgsz", 128, 1280, 640, step=32,
                               name="imgsz",
                               desc="学習・推論時の画像サイズ（ピクセル）。大きいほど精度が上がるが計算コストが増加する。")
            patience    = _nw("patience（0=無効）", 0, 1000, 50, step=10,
                               name="patience",
                               desc="EarlyStopping の待機エポック数。N エポック間 val metrics が改善しなければ自動終了。0 で無効。")
            save_period = _nw("save_period（0=無効）", 0, 500, 0, step=10,
                               name="save_period",
                               desc="N エポックごとにチェックポイントを保存する間隔。0 で無効。長期学習での途中確認に便利。")
            workers     = _nw("workers", 0, 32, 8, step=1,
                               name="workers",
                               desc="DataLoader の CPU ワーカースレッド数。多すぎるとメモリ不足になることがある。")
        with _oc2:
            optimizer   = _selw("optimizer", ["auto","SGD","Adam","AdamW","NAdam","RAdam"], 0,
                                 name="optimizer",
                                 desc="`auto` はモデルに応じて自動選択。細かく制御する場合は SGD または AdamW 推奨。")
            lr0         = _nw("lr0（初期学習率）", 1e-5, 1.0, 0.01, format="%.5f", step=0.001,
                               name="lr0",
                               desc="初期学習率。SGD では 0.01、Adam/AdamW では 0.001 が一般的な推奨値。")
            lrf         = _nw("lrf（最終LR係数）", 1e-4, 1.0, 0.01, format="%.4f", step=0.001,
                               name="lrf",
                               desc="学習率スケジューラの終端係数。最終学習率 = `lr0 × lrf`。")
            cos_lr      = _ckw("cos_lr（コサイン学習率）", False,
                                name="cos_lr",
                                desc="True でコサイン学習率スケジューラを使用。学習後半を滑らかに減衰させる。")
        with _oc3:
            momentum    = _nw("momentum（SGD/Adam β1）", 0.5, 0.999, 0.937, format="%.3f", step=0.01,
                               name="momentum",
                               desc="SGD のモメンタム係数、または Adam 系の β1 パラメータ。")
            warmup_epochs = _nw("warmup_epochs", 0, 50, 3, step=1,
                                 name="warmup_epochs",
                                 desc="ウォームアップのエポック数。最初の N エポックで学習率を 0 から lr0 まで徐々に増加させる。")
            warmup_momentum = _nw("warmup_momentum", 0.0, 1.0, 0.8, format="%.2f", step=0.05,
                                   name="warmup_momentum",
                                   desc="ウォームアップ中の初期モメンタム値。")
            warmup_bias_lr  = _nw("warmup_bias_lr", 0.0, 1.0, 0.1, format="%.3f", step=0.01,
                                   name="warmup_bias_lr",
                                   desc="ウォームアップ中のバイアス層の学習率。")
        with _oc4:
            weight_decay = _nw("weight_decay", 0.0, 0.01, 0.0005, format="%.5f", step=0.0001,
                                name="weight_decay",
                                desc="L2 正則化（重み減衰）の強度。過学習の抑制に効果的。")
            dropout      = _sw("dropout", 0.0, 0.5, 0.0, step=0.05,
                                name="dropout",
                                desc="Dropout の確率。0 で無効。学習時にランダムにユニットを無効化して汎化性を高める。",
                                url=_DOC_TRAIN)
            nbs          = _nw("nbs（損失正規化基準バッチ）", 1, 256, 64, step=8,
                                name="nbs",
                                desc="名目バッチサイズ。実バッチサイズが異なる場合に損失をスケーリングする基準値。")
            amp          = _ckw("AMP（混合精度学習）", True,
                                 name="amp",
                                 desc="True で FP16 演算を混在させ GPU メモリを節約しつつ速度を向上させる。Blackwell GPU では有効推奨。")
            cache        = _ckw("cache（画像キャッシュ）", False,
                                 name="cache",
                                 desc="学習画像を RAM/disk にキャッシュ。繰り返しのディスク読み込みを削減して高速化。大規模データセットでは RAM 不足に注意。")

    # ── データ拡張（Augmentation）────────────────────────────────────────────
    with st.expander("🎨 データ拡張（Augmentation）", expanded=False):
        _mn      = model_name.lower()
        _is_seg  = "-seg"  in _mn
        _is_pose = "-pose" in _mn
        _is_cls  = "-cls"  in _mn
        _is_obb  = "-obb"  in _mn
        _has_box = not _is_cls
        _task_label = ("segment" if _is_seg else "pose" if _is_pose
                       else "classify" if _is_cls else "obb" if _is_obb else "detect")
        st.caption(
            f"推定タスク: **{_task_label}** — タスクに適用されないパラメータはグレーアウトされます"
        )

        # ── 幾何変換 ──────────────────────────────────────────────────────────
        st.markdown("##### 🔁 幾何変換")
        _g1, _g2, _g3, _g4 = st.columns(4)
        with _g1:
            degrees = _sw("degrees（回転 ±°）", 0.0, 180.0, 0.0, step=1.0,
                           name="degrees",
                           desc="画像をランダムに回転させる角度範囲（±degrees°）。0 で無効。ロボット視点など姿勢が変化する環境で有効。",
                           disabled=not _has_box)
            shear   = _sw("shear（せん断 ±°）", 0.0, 10.0, 0.0, step=0.5,
                           name="shear",
                           desc="せん断変形（ずれ歪み）の角度範囲（±degrees°）。画像を平行四辺形状に歪める。",
                           disabled=not _has_box)
        with _g2:
            scale     = _sw("scale（拡大縮小）", 0.0, 0.9, 0.5, step=0.05,
                             name="scale",
                             desc="ランダムスケーリングの変化幅。0.5 なら画像サイズが ×0.5〜×1.5 の範囲で変化。距離・解像度の変動に対応。",
                             disabled=not _has_box)
            translate = _sw("translate（平行移動）", 0.0, 0.9, 0.1, step=0.05,
                             name="translate",
                             desc="水平・垂直方向の平行移動量（画像サイズ比）。物体が画像端にある場合への対応。",
                             disabled=not _has_box)
        with _g3:
            fliplr = _sw("fliplr（左右反転）", 0.0, 1.0, 0.5, step=0.05,
                          name="fliplr",
                          desc="水平（左右）反転の確率。文字・数字など向きが意味を持つタスクでは 0.0 を推奨。")
            flipud = _sw("flipud（上下反転）", 0.0, 1.0, 0.0, step=0.05,
                          name="flipud",
                          desc="垂直（上下）反転の確率。重力方向が重要なタスクでは 0.0 を推奨。")
        with _g4:
            perspective = _nw("perspective（透視変換）", 0.0, 0.001, 0.0,
                               format="%.4f", step=0.0001,
                               name="perspective",
                               desc="透視投影変換の強度（0〜0.001 程度）。平面を斜めから見たような 3D 的歪みを追加。",
                               url=_DOC_AUG, disabled=not _has_box)
            bgr = _sw("bgr（BGR↔RGB 反転確率）", 0.0, 1.0, 0.0, step=0.05,
                       name="bgr",
                       desc="BGR と RGB のチャンネル順をランダムに入れ替える確率。色に依存しない特徴を学習させる。")

        # ── 色調・明度変換 ───────────────────────────────────────────────────
        st.markdown("##### 🌈 色調・明度変換")
        _c1, _c2, _c3 = st.columns(3)
        with _c1:
            hsv_h = _sw("hsv_h（色相変動）", 0.0, 0.10, 0.015, step=0.005,
                         name="hsv_h",
                         desc="HSV 色空間の色相（Hue）の変動量。照明条件の変化や異なる色帯域への汎化に効果的。")
        with _c2:
            hsv_s = _sw("hsv_s（彩度変動）", 0.0, 1.0, 0.7, step=0.05,
                         name="hsv_s",
                         desc="HSV 色空間の彩度（Saturation）の変動量。色の鮮やかさをランダムに変化させる。")
        with _c3:
            hsv_v = _sw("hsv_v（明度変動）", 0.0, 1.0, 0.4, step=0.05,
                         name="hsv_v",
                         desc="HSV 色空間の明度（Value）の変動量。屋内外の照明差や露出変化に対応させる。")

        # ── 合成拡張 ─────────────────────────────────────────────────────────
        st.markdown("##### 🔀 合成拡張")
        _m1, _m2, _m3, _m4 = st.columns(4)
        with _m1:
            mosaic = _sw("mosaic（4 画像合成）", 0.0, 1.0,
                          1.0 if _has_box else 0.0, step=0.05,
                          name="mosaic",
                          desc="4 枚の画像をランダムにモザイク結合する確率。小物体の検出精度向上に非常に効果的。detect/segment/pose 向け。",
                          disabled=not _has_box)
            close_mosaic = _nw("close_mosaic（終盤N エポックOFF）", 0, 200, 10, step=5,
                                name="close_mosaic",
                                desc="最後の N エポックでモザイク拡張を OFF にする。学習終盤に拡張なしの本来の分布で収束させ精度を安定させる。",
                                url=_DOC_AUG, disabled=not _has_box)
        with _m2:
            mixup  = _sw("mixup", 0.0, 1.0, 0.0, step=0.05,
                          name="mixup",
                          desc="2 枚の画像とラベルを α ブレンドで混合する確率。クラス境界付近の汎化性向上に有効。detect/segment 向け。",
                          disabled=not _has_box)
            cutmix = _sw("cutmix", 0.0, 1.0, 0.0, step=0.05,
                          name="cutmix",
                          desc="ランダムに切り抜いた領域を別画像で置き換える確率。MixUp の空間的バリアント。",
                          disabled=not _has_box)
        with _m3:
            copy_paste = _sw("copy_paste（セグのみ）", 0.0, 1.0, 0.0, step=0.05,
                              name="copy_paste",
                              desc="【セグメンテーション専用】別画像のセグメント済みオブジェクトをコピーして貼り付ける確率。クラス不均衡の解消やレアオブジェクト増強に有効。",
                              disabled=not _is_seg)
            _cp_c, _cp_h = st.columns([5, 1])
            with _cp_c:
                copy_paste_mode = st.selectbox(
                    "copy_paste_mode", ["flip", "mixup"],
                    disabled=not (_is_seg and copy_paste > 0.0),
                )
            with _cp_h:
                st.markdown('<div style="margin-top:28px"></div>', unsafe_allow_html=True)
                _ph("copy_paste_mode",
                    "`flip` は対象を反転して貼り付け、`mixup` はブレンドして貼り付け。copy_paste > 0 のときのみ有効。",
                    _DOC_AUG)
        with _m4:
            erasing = _sw("erasing（ランダム消去）", 0.0, 0.9, 0.4, step=0.05,
                           name="erasing",
                           desc="ランダムな矩形領域を消去する確率（Random Erasing）。オクルージョン（物体が部分的に隠れる）への耐性を向上させる。")

        # ── 分類専用 ─────────────────────────────────────────────────────────
        if _is_cls:
            st.markdown("##### 📋 分類専用")
            _cl1, _cl2 = st.columns(2)
            with _cl1:
                crop_fraction = _sw("crop_fraction（ランダムクロップ割合）",
                                    0.1, 1.0, 1.0, step=0.05,
                                    name="crop_fraction",
                                    desc="分類タスク専用。画像を中心からランダムにクロップする際の最小割合。",
                                    url=_DOC_AUG)
            with _cl2:
                auto_augment = _selw(
                    "auto_augment", ["randaugment", "autoaugment", "augmix"], 0,
                    name="auto_augment",
                    desc="分類タスク専用の自動拡張ポリシー。`randaugment`（ランダム操作）、`autoaugment`（AutoAugment）、`augmix`（AugMix）から選択。",
                    url=_DOC_AUG,
                )
        else:
            crop_fraction = 1.0
            auto_augment  = "randaugment"

    st.markdown("---")

    # ── 学習ボタン ───────────────────────────────────────────────────────────
    btn_col1, btn_col2 = st.columns([2, 1])
    with btn_col1:
        start_btn = st.button(
            "▶ 学習開始",
            type="primary",
            disabled=st.session_state.training_running,
            use_container_width=True,
        )
    with btn_col2:
        if st.session_state.training_running:
            st.markdown('<span class="badge-warn">RUNNING</span>', unsafe_allow_html=True)
        elif st.session_state.training_progress == 100:
            st.markdown('<span class="badge-ok">COMPLETED</span>', unsafe_allow_html=True)

    if start_btn:
        yaml_p = Path(data_yaml_path)
        if not yaml_p.exists():
            st.error(f"data.yaml が見つかりません: {yaml_p}")
        else:
            _train_kwargs: dict = {
                # ── 基本 ──────────────────────────────────────────────────
                "imgsz": int(imgsz),
                "device": 0,
                "workers": int(workers),
                "nbs": int(nbs),
                # ── 最適化 ────────────────────────────────────────────────
                "optimizer": optimizer,
                "lr0": float(lr0),
                "lrf": float(lrf),
                "momentum": float(momentum),
                "warmup_epochs": float(warmup_epochs),
                "warmup_momentum": float(warmup_momentum),
                "warmup_bias_lr": float(warmup_bias_lr),
                "weight_decay": float(weight_decay),
                "dropout": float(dropout),
                "cos_lr": cos_lr,
                "amp": amp,
                "cache": cache,
                # ── 幾何変換 ──────────────────────────────────────────────
                "degrees": float(degrees),
                "scale": float(scale),
                "translate": float(translate),
                "shear": float(shear),
                "perspective": float(perspective),
                "flipud": float(flipud),
                "fliplr": float(fliplr),
                "bgr": float(bgr),
                # ── 色調変換 ──────────────────────────────────────────────
                "hsv_h": float(hsv_h),
                "hsv_s": float(hsv_s),
                "hsv_v": float(hsv_v),
                # ── 合成拡張 ──────────────────────────────────────────────
                "mosaic": float(mosaic),
                "mixup": float(mixup),
                "cutmix": float(cutmix),
                "erasing": float(erasing),
                "close_mosaic": int(close_mosaic),
            }
            # セグメンテーション専用
            if _is_seg:
                _train_kwargs["copy_paste"] = float(copy_paste)
                _train_kwargs["copy_paste_mode"] = copy_paste_mode
            # 分類専用
            if _is_cls:
                _train_kwargs["crop_fraction"] = float(crop_fraction)
                _train_kwargs["auto_augment"] = auto_augment
            # 条件付き
            if patience > 0:
                _train_kwargs["patience"] = int(patience)
            if save_period > 0:
                _train_kwargs["save_period"] = int(save_period)

            with _train_log_lock:
                _train_state["log"] = []
                _train_state["progress"] = 0
                _train_state["running"] = True
                _train_state["error"] = None
                _train_state["model_path"] = None

            t = threading.Thread(
                target=_train_worker,
                args=(data_yaml_path, model_name, epochs, batch_size,
                      clearml_project, run_name, _train_kwargs),
                daemon=True,
            )
            t.start()
            st.rerun()

    # --- _train_state → st.session_state に同期 ---
    with _train_log_lock:
        st.session_state.training_log = list(_train_state["log"])
        st.session_state.training_progress = _train_state["progress"]
        st.session_state.training_running = _train_state["running"]
        if _train_state["error"]:
            st.session_state.training_error = _train_state["error"]
        if _train_state["model_path"]:
            st.session_state.last_model_path = _train_state["model_path"]

    # --- 進捗表示 ---
    if st.session_state.training_running or st.session_state.training_progress > 0:
        prog = st.session_state.training_progress
        st.progress(prog / 100, text=f"進捗: {prog}%")

        log_html = "<br>".join(st.session_state.training_log[-200:])
        st.markdown(
            f'<div class="log-area">{log_html}</div>',
            unsafe_allow_html=True,
        )

        if st.session_state.training_running:
            time.sleep(2)
            st.rerun()

    if st.session_state.training_error:
        st.error(f"学習エラー: {st.session_state.training_error}")

    # --- 完了後: モデル選択 ---
    if st.session_state.last_model_path:
        st.success(f"✅ 最新モデル: `{st.session_state.last_model_path}`")

    st.markdown('</div>', unsafe_allow_html=True)

    # --- 既存モデル選択 ---
    with st.expander("📦 既存の学習済みモデルを選択"):
        existing_models = list(MODELS_DIR.rglob("*.pt"))
        if existing_models:
            model_labels = [str(p.relative_to(MODELS_DIR)) for p in existing_models]
            sel_model = st.selectbox("モデルファイル", model_labels)
            if st.button("このモデルを使用"):
                st.session_state.last_model_path = str(MODELS_DIR / sel_model)
                st.success(f"モデルを設定: {st.session_state.last_model_path}")
        else:
            st.info("models/ ディレクトリに .pt ファイルが見つかりません。")


# ===========================================================================
# タブ3: 推論 & 可視化
# ===========================================================================
with tab3:
    st.markdown('<div class="pipeline-card"><h3>🔭 推論 & FiftyOne 可視化</h3>', unsafe_allow_html=True)

    # --- モデル確認 ---
    current_model = st.session_state.last_model_path or ""
    model_display = current_model if current_model else "（未設定）"
    st.info(f"使用モデル: `{model_display}`\n→ タブ②で学習または既存モデルを選択してください。")

    # --- 推論対象ディレクトリ ---
    _img_dirs = _find_image_dirs(DATA_DIR)
    _dir_labels = [str(d.relative_to(DATA_DIR)) for d in _img_dirs]
    _MANUAL = "（手動入力）"
    _dir_options = _dir_labels + [_MANUAL]

    _sel = st.selectbox(
        "テスト画像ディレクトリを選択",
        _dir_options,
        index=0 if _dir_labels else len(_dir_options) - 1,
        help=f"スキャン元: {DATA_DIR}",
    )
    if _sel == _MANUAL:
        test_image_dir = st.text_input(
            "パスを直接入力 (コンテナ内絶対パス)",
            value=str(DATA_DIR / "test/images"),
        )
    else:
        test_image_dir = str(DATA_DIR / _sel)
        st.code(test_image_dir, language="text")

    inf_conf = st.slider("確信度しきい値", 0.05, 0.95, 0.25, step=0.05, key="inf_conf")

    col_run, col_vis = st.columns(2)

    # --- 推論実行ボタン ---
    with col_run:
        if st.button("▶ 推論実行", type="primary", use_container_width=True,
                    disabled=not current_model):
            img_dir = Path(test_image_dir)
            if not img_dir.exists():
                st.error(f"画像ディレクトリが存在しません: {img_dir}")
            else:
                with st.spinner("推論中…"):
                    saved = run_inference(
                        current_model,
                        img_dir,
                        PREDICTIONS_DIR,
                        conf_threshold=inf_conf,
                    )
                if saved:
                    st.success(f"✅ 推論完了: {len(saved)} 件のJSONを保存")
                    with st.expander("保存されたJSON (先頭1件)"):
                        with open(saved[0]) as f:
                            st.json(json.load(f))

    # --- FiftyOne 起動ボタン ---
    with col_vis:
        fo_dataset_name = st.text_input("FiftyOneデータセット名", value="yolo_predictions")
        if st.button("🔭 FiftyOne で可視化", use_container_width=True):
            with st.spinner("FiftyOne App を起動中…"):
                port = launch_fiftyone(fo_dataset_name, PREDICTIONS_DIR)
            if port:
                fo_url = f"http://localhost:{port}"
                st.success(f"FiftyOne App が起動しました: {fo_url}")
                st.session_state.fiftyone_port = port

    # --- FiftyOne iframe 埋め込み ---
    if st.session_state.fiftyone_port:
        fo_url = f"http://localhost:{st.session_state.fiftyone_port}"
        st.markdown(f"""
<div style="margin-top:16px;">
    <p style="color:#4a6080; font-size:.85rem;">
        FiftyOne App が別ポートで起動中。同一ホストの場合は以下から直接アクセスできます。
    </p>
    <a href="{fo_url}" target="_blank" style="color:#7ecff4; font-family:'JetBrains Mono',monospace;">
        🔗 FiftyOne App を開く → {fo_url}
    </a>
</div>
<iframe src="{fo_url}" width="100%" height="600px"
    style="border:1px solid #1e2330; border-radius:8px; margin-top:12px;"
    allow="fullscreen">
</iframe>
""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # --- 推論結果 JSON ブラウザ ---
    with st.expander("📋 predictions/ の結果ファイル一覧"):
        json_files = sorted(PREDICTIONS_DIR.glob("*.json"))
        if json_files:
            for jf in json_files[:20]:  # 最大20件表示
                with st.container():
                    c1, c2 = st.columns([4, 1])
                    with c1:
                        st.text(jf.name)
                    with c2:
                        if st.button("👁", key=f"view_{jf.name}"):
                            with open(jf) as f:
                                st.json(json.load(f))
        else:
            st.info("predictions/ にJSONファイルがありません。先に推論を実行してください。")

# ===========================================================================
# タブ4: データ管理
# ===========================================================================
with tab4:
    import shutil

    st.markdown('<div class="pipeline-card"><h3>📁 データ管理</h3>', unsafe_allow_html=True)

    # --- data/ データセット一覧 ---
    st.markdown("#### 学習データセット (`data/`)")
    datasets = sorted(DATA_DIR.iterdir()) if DATA_DIR.exists() else []
    datasets = [d for d in datasets if d.is_dir()]
    if not datasets:
        st.info("data/ にデータセットがありません。")
    else:
        for ds in datasets:
            all_files = [f for f in ds.rglob("*") if f.is_file()]
            file_count = len(all_files)
            size_mb = sum(f.stat().st_size for f in all_files) / (1024 * 1024)
            col1, col2, col3 = st.columns([4, 2, 1])
            with col1:
                st.text(ds.name)
            with col2:
                st.text(f"{file_count} files  /  {size_mb:.1f} MB")
            with col3:
                if st.button("🗑", key=f"del_ds_{ds.name}", help=f"{ds.name} を削除"):
                    shutil.rmtree(ds)
                    st.success(f"{ds.name} を削除しました")
                    st.rerun()

    st.markdown("---")

    # --- models/ モデル一覧 ---
    st.markdown("#### 学習済みモデル (`models/`)")
    model_files = sorted(MODELS_DIR.rglob("*.pt")) if MODELS_DIR.exists() else []
    if not model_files:
        st.info("models/ に .pt ファイルがありません。")
    else:
        for mp in model_files:
            size_mb = mp.stat().st_size / (1024 * 1024)
            col1, col2, col3 = st.columns([4, 2, 1])
            with col1:
                st.text(str(mp.relative_to(MODELS_DIR)))
            with col2:
                st.text(f"{size_mb:.1f} MB")
            with col3:
                if st.button("🗑", key=f"del_model_{mp}", help=f"{mp.name} を削除"):
                    mp.unlink()
                    if st.session_state.last_model_path == str(mp):
                        st.session_state.last_model_path = None
                    st.success(f"{mp.name} を削除しました")
                    st.rerun()

    st.markdown("---")

    # --- predictions/ 一括クリア ---
    st.markdown("#### 推論結果 (`predictions/`)")
    pred_files = list(PREDICTIONS_DIR.glob("*.json")) if PREDICTIONS_DIR.exists() else []
    if not pred_files:
        st.info("predictions/ に結果 JSON がありません。")
    else:
        st.text(f"{len(pred_files)} 件の結果ファイル")
        if st.button("🗑 predictions/ をすべてクリア", type="secondary"):
            for jf in pred_files:
                jf.unlink()
            st.success("predictions/ をクリアしました")
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# フッター
# ---------------------------------------------------------------------------
st.markdown("""
<div style="border-top:1px solid #1e2330; margin-top:40px; padding-top:12px;
            text-align:center; color:#2a3a50; font-size:.75rem; font-family:'JetBrains Mono',monospace;">
    MLOps Pipeline v1.0 · CVAT · YOLO · ClearML · FiftyOne · Streamlit
</div>
""", unsafe_allow_html=True)