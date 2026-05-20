# =============================================================================
# MLOps 統合UI - main.py
# Streamlit + CVAT API + YOLO学習 + MLflow + FiftyOne
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
import streamlit.components.v1 as components


@st.cache_resource
def _get_train_shared() -> tuple[dict, threading.Lock]:
    """st.rerun() をまたいで同一オブジェクトを保持する共有状態。
    Streamlit はスクリプトを再実行するたびにモジュール変数を再初期化するため、
    st.cache_resource でキャッシュして常に同一インスタンスを返す。
    """
    return (
        {"log": [], "progress": 0, "running": False, "error": None, "model_path": None, "metrics_history": []},
        threading.Lock(),
    )

# ---------------------------------------------------------------------------
# 定数・環境変数
# ---------------------------------------------------------------------------
DATA_DIR       = Path(os.getenv("DATA_DIR",       "/workspace/data"))
MODELS_DIR     = Path(os.getenv("MODELS_DIR",     "/workspace/models"))
PREDICTIONS_DIR= Path(os.getenv("PREDICTIONS_DIR","/workspace/predictions"))
CVAT_HOST      = os.getenv("CVAT_HOST",     "http://cvat-server:8080")  # コンテナ内通信用
CVAT_WEB       = os.getenv("CVAT_WEB_HOST", "http://localhost:8080")    # ブラウザ表示用
CVAT_USER      = os.getenv("CVAT_USERNAME","admin")
CVAT_PASS      = os.getenv("CVAT_PASSWORD","admin")
MLFLOW_URI     = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MLFLOW_WEB     = os.getenv("MLFLOW_WEB_HOST", "http://localhost:5000")
FIFTYONE_PORT  = int(os.getenv("FIFTYONE_PORT","5151"))

for d in [DATA_DIR, MODELS_DIR, PREDICTIONS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

USER_THEMES_PATH = MODELS_DIR / ".user_themes.json"

# ---------------------------------------------------------------------------
# プリセットテーマ定義
# ---------------------------------------------------------------------------
PRESET_THEMES: dict[str, dict] = {
    "ダーク（デフォルト）": {
        "bg_app": "#0d0f14", "bg_sidebar": "#12151c", "bg_card": "#161b26",
        "bg_card_inner": "#0e1520", "bg_log": "#0a0c10",
        "border": "#1e2330", "border_accent": "#2d4a80",
        "text_primary": "#c8d8e8", "text_secondary": "#6a8aaa", "text_muted": "#4a6080",
        "accent": "#7ecff4", "accent_dark": "#2d7dd2",
        "success": "#4caf7d", "success_bg": "#111f17", "success_border": "#2d6b47",
        "warning": "#f0a830", "warning_bg": "#3a2a10", "warning_border": "#7a5520",
        "error": "#f06060", "error_bg": "#3a1a1a", "error_border": "#7a3030",
        "btn_bg": "#1a2540", "btn_hover": "#2d4a80",
        "chip_bg": "#0e1520", "chip_border": "#1e2d42", "chip_text": "#4a90c4",
    },
    "ダーク カラフル": {
        "bg_app": "#0d0c14", "bg_sidebar": "#13101c", "bg_card": "#1a1628",
        "bg_card_inner": "#120e20", "bg_log": "#0a0810",
        "border": "#28203c", "border_accent": "#6040b0",
        "text_primary": "#ead8ff", "text_secondary": "#9a78d0", "text_muted": "#6040a0",
        "accent": "#c47eff", "accent_dark": "#8040d0",
        "success": "#60e090", "success_bg": "#0d1a14", "success_border": "#306050",
        "warning": "#ffb040", "warning_bg": "#2a1e08", "warning_border": "#705020",
        "error": "#ff7090", "error_bg": "#2a1018", "error_border": "#703040",
        "btn_bg": "#221040", "btn_hover": "#4820a0",
        "chip_bg": "#160e28", "chip_border": "#2e1e50", "chip_text": "#a078e8",
    },
    "ライト シンプル": {
        "bg_app": "#f5f7fa", "bg_sidebar": "#ebeef5", "bg_card": "#ffffff",
        "bg_card_inner": "#eef2f8", "bg_log": "#e4e9f2",
        "border": "#ced6e4", "border_accent": "#4878b8",
        "text_primary": "#1a2038", "text_secondary": "#445878", "text_muted": "#7a8ca0",
        "accent": "#2d6bb8", "accent_dark": "#1a4a90",
        "success": "#2a7848", "success_bg": "#e8f5ee", "success_border": "#4a9868",
        "warning": "#a86800", "warning_bg": "#fff3d8", "warning_border": "#c89040",
        "error": "#c03030", "error_bg": "#fff0f0", "error_border": "#e06060",
        "btn_bg": "#dce8f8", "btn_hover": "#b8d0f0",
        "chip_bg": "#e8f0fa", "chip_border": "#b0c4dc", "chip_text": "#2a5898",
    },
    "ライト カラフル": {
        "bg_app": "#f4f0ff", "bg_sidebar": "#ece4ff", "bg_card": "#ffffff",
        "bg_card_inner": "#ede5ff", "bg_log": "#e4daf8",
        "border": "#d0c4ec", "border_accent": "#8840e8",
        "text_primary": "#1c1030", "text_secondary": "#5830a0", "text_muted": "#8860c0",
        "accent": "#7c30e0", "accent_dark": "#5010b8",
        "success": "#1a8040", "success_bg": "#e8fff0", "success_border": "#40a060",
        "warning": "#c07000", "warning_bg": "#fff8e0", "warning_border": "#d09030",
        "error": "#c02040", "error_bg": "#fff0f4", "error_border": "#e05070",
        "btn_bg": "#e8d8ff", "btn_hover": "#c8a8ff",
        "chip_bg": "#ecdeff", "chip_border": "#c4a0e8", "chip_text": "#6010c0",
    },
}

# ユーザーが色ピッカーで編集できる主要フィールド
THEME_EDIT_FIELDS: list[tuple[str, str]] = [
    ("bg_app",       "背景色（メイン）"),
    ("bg_card",      "カード背景色"),
    ("text_primary", "メインテキスト色"),
    ("accent",       "アクセント/ハイライト色"),
    ("success",      "成功/完了色"),
    ("warning",      "警告色"),
    ("error",        "エラー色"),
]

# ---------------------------------------------------------------------------
# Streamlit ページ設定
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="detection_dev_ui",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# st.set_page_config() の後で取得（cache_resource の初回呼び出しが安全なタイミング）
_train_state, _train_log_lock = _get_train_shared()

# ---------------------------------------------------------------------------
# カスタム CSS（CSS変数ベース・テーマ切替対応）
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

/* ── デフォルト変数（ダークテーマ）── 後から注入されるテーマ変数で上書き */
:root {
  --bg-app:         #0d0f14;
  --bg-sidebar:     #12151c;
  --bg-card:        #161b26;
  --bg-card-inner:  #0e1520;
  --bg-log:         #0a0c10;
  --border:         #1e2330;
  --border-accent:  #2d4a80;
  --text-primary:   #c8d8e8;
  --text-secondary: #6a8aaa;
  --text-muted:     #4a6080;
  --accent:         #7ecff4;
  --accent-dark:    #2d7dd2;
  --success:        #4caf7d;
  --success-bg:     #111f17;
  --success-border: #2d6b47;
  --warning:        #f0a830;
  --warning-bg:     #3a2a10;
  --warning-border: #7a5520;
  --error:          #f06060;
  --error-bg:       #3a1a1a;
  --error-border:   #7a3030;
  --btn-bg:         #1a2540;
  --btn-hover:      #2d4a80;
  --chip-bg:        #0e1520;
  --chip-border:    #1e2d42;
  --chip-text:      #4a90c4;
}

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
code, pre, .stCode { font-family: 'JetBrains Mono', monospace; }

.stApp { background: var(--bg-app); }

[data-testid="stSidebar"] {
    background: var(--bg-sidebar);
    border-right: 1px solid var(--border);
}

.pipeline-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 20px;
    margin: 10px 0;
}
.pipeline-card h3 { color: var(--accent); margin-top: 0; }

.badge-ok   { background: var(--success-bg);  color: var(--success);  border: 1px solid var(--success-border);
              padding:2px 10px; border-radius:4px; font-size:.78rem; }
.badge-warn { background: var(--warning-bg);  color: var(--warning);  border: 1px solid var(--warning-border);
              padding:2px 10px; border-radius:4px; font-size:.78rem; }
.badge-err  { background: var(--error-bg);    color: var(--error);    border: 1px solid var(--error-border);
              padding:2px 10px; border-radius:4px; font-size:.78rem; }

.log-area {
    background: var(--bg-log);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 12px;
    font-family: 'JetBrains Mono', monospace;
    font-size: .78rem;
    color: var(--text-secondary);
    max-height: 520px;
    overflow-y: auto;
    white-space: pre-wrap;
    word-break: break-all;
}

.stProgress > div > div > div > div {
    background: linear-gradient(90deg, var(--accent-dark), var(--accent));
}

.stButton > button {
    background: var(--btn-bg);
    color: var(--accent);
    border: 1px solid var(--border-accent);
    border-radius: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    letter-spacing: .05em;
    transition: all .2s;
}
.stButton > button:hover {
    background: var(--btn-hover);
    border-color: var(--accent);
    color: #fff;
}

/* パイプラインフロー */
.pipeline-flow {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 16px 0 8px;
    flex-wrap: wrap;
}
.pf-step {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 8px 14px;
    text-align: center;
    min-width: 110px;
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: .82rem;
}
.pf-step .pf-label { color: var(--text-muted);   font-size:.72rem; margin-bottom:2px; }
.pf-step .pf-name  { color: var(--text-primary);  font-weight:600; }
.pf-step .pf-icon  { font-size: .9rem; }
.pf-step.complete  { border-color: var(--success-border); background: var(--success-bg); }
.pf-step.complete .pf-name { color: var(--success); }
.pf-step.active    { border-color: var(--accent); background: var(--bg-card-inner); animation: pulse-border 2s infinite; }
.pf-step.active .pf-name { color: var(--accent); }
.pf-step.pending   { opacity: .45; }
.pf-arrow          { color: var(--text-muted); font-size: 1.1rem; }
@keyframes pulse-border {
    0%,100% { border-color: var(--accent); }
    50%      { border-color: var(--accent-dark); }
}

/* ステップバナー */
.step-banner {
    background: var(--bg-card-inner);
    border-left: 3px solid var(--accent);
    border-radius: 0 6px 6px 0;
    padding: 10px 16px;
    margin-bottom: 16px;
}
.step-banner .sb-title { color: var(--accent);         font-size: .95rem; font-weight:600;
                          font-family:'JetBrains Mono',monospace; }
.step-banner .sb-prev  { color: var(--success);        font-size: .78rem; margin-top:4px; }
.step-banner .sb-desc  { color: var(--text-secondary); font-size: .78rem; margin-top:2px; }

/* サイドバー サマリー */
.sidebar-stat {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 4px 0;
    font-size: .82rem;
}
.sidebar-stat .ss-label { color: var(--text-muted); }
.sidebar-stat .ss-value { color: var(--accent); font-family:'JetBrains Mono',monospace; font-weight:700; }

/* トピックスタブ */
.topic-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 18px 20px;
    margin: 8px 0;
}
.topic-card .tc-icon  { font-size: 1.6rem; margin-bottom: 6px; }
.topic-card .tc-title { color: var(--accent); font-size: .95rem; font-weight:700;
                         font-family:'JetBrains Mono',monospace; margin-bottom:6px; }
.topic-card .tc-body  { color: var(--text-secondary); font-size: .82rem; line-height:1.6; }
.topic-card .tc-sub   { color: var(--text-muted); font-size:.72rem; text-transform:uppercase;
                         letter-spacing:.05em; margin: 10px 0 4px; }
.tc-chip {
    display: inline-block;
    background: var(--chip-bg);
    border: 1px solid var(--chip-border);
    color: var(--chip-text);
    font-size: .72rem;
    padding: 2px 8px;
    border-radius: 12px;
    margin: 2px;
    font-family: 'JetBrains Mono', monospace;
}
.link-card {
    background: var(--bg-card-inner);
    border: 1px solid var(--chip-border);
    border-radius: 6px;
    padding: 10px 14px;
    margin: 6px 0;
}
.link-card .lc-title { color: var(--text-primary); font-size:.85rem; font-weight:600; }
.link-card .lc-desc  { color: var(--text-muted);   font-size:.75rem; margin-top:2px; }
.link-card a { color: var(--accent); text-decoration:none; }
.link-card a:hover { text-decoration:underline; }

/* テーマプレビュースウォッチ */
.theme-swatch {
    display: inline-block;
    width: 14px; height: 14px;
    border-radius: 3px;
    border: 1px solid rgba(128,128,128,.35);
    vertical-align: middle;
    margin-right: 3px;
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
    "training_metrics_history": [],
    "training_notified": True,
    "fiftyone_session": None,
    "fiftyone_port": None,
    "last_model_path": None,
    "cvat_tasks": [],
    "cvat_xml_info": None,
    "cvat_raw_dir": None,
    "theme_name": "ライト シンプル",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------------------------------------------------------------------------
# テーマ永続化・注入
# ---------------------------------------------------------------------------
def _load_user_themes() -> dict:
    if USER_THEMES_PATH.exists():
        try:
            return json.loads(USER_THEMES_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_user_themes(themes: dict) -> None:
    USER_THEMES_PATH.write_text(json.dumps(themes, ensure_ascii=False, indent=2), encoding="utf-8")


def _get_active_theme() -> dict:
    name = st.session_state.get("theme_name", "ダーク（デフォルト）")
    if name in PRESET_THEMES:
        return PRESET_THEMES[name]
    user_themes = _load_user_themes()
    return user_themes.get(name, PRESET_THEMES["ライト シンプル"])


def _build_theme_vars(t: dict) -> str:
    return f"""<style>
:root {{
  --bg-app:         {t['bg_app']};
  --bg-sidebar:     {t['bg_sidebar']};
  --bg-card:        {t['bg_card']};
  --bg-card-inner:  {t['bg_card_inner']};
  --bg-log:         {t['bg_log']};
  --border:         {t['border']};
  --border-accent:  {t['border_accent']};
  --text-primary:   {t['text_primary']};
  --text-secondary: {t['text_secondary']};
  --text-muted:     {t['text_muted']};
  --accent:         {t['accent']};
  --accent-dark:    {t['accent_dark']};
  --success:        {t['success']};
  --success-bg:     {t['success_bg']};
  --success-border: {t['success_border']};
  --warning:        {t['warning']};
  --warning-bg:     {t['warning_bg']};
  --warning-border: {t['warning_border']};
  --error:          {t['error']};
  --error-bg:       {t['error_bg']};
  --error-border:   {t['error_border']};
  --btn-bg:         {t['btn_bg']};
  --btn-hover:      {t['btn_hover']};
  --chip-bg:        {t['chip_bg']};
  --chip-border:    {t['chip_border']};
  --chip-text:      {t['chip_text']};
}}
.stApp {{ background: {t['bg_app']}; }}
[data-testid="stSidebar"] {{
  background: {t['bg_sidebar']};
  border-right: 1px solid {t['border']};
}}
</style>"""


# テーマ変数を注入（デフォルトCSS変数を上書き）
st.markdown(_build_theme_vars(_get_active_theme()), unsafe_allow_html=True)

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
    try:
        client = get_cvat_client()
        if not client:
            return []
        tasks = client.tasks.list()
        result = []
        for t in tasks:
            assignee_name = ""
            if hasattr(t, "assignee") and t.assignee:
                assignee_name = getattr(t.assignee, "username", "") or getattr(t.assignee, "email", "")
            result.append({
                "id": t.id,
                "name": t.name,
                "size": t.size,
                "status": t.status,
                "assignee": assignee_name,
            })
        return result
    except Exception as e:
        st.error(f"CVATタスク取得エラー: {e}")
        return []


def fetch_cvat_task_labels(task_ids: list[int]) -> dict[str, list[str]]:
    """複数タスクIDからラベル名リストを返す {タスク名(ID:xx): [label, ...]}"""
    try:
        client = get_cvat_client()
        if not client:
            return {}
        result = {}
        for tid in task_ids:
            try:
                task = client.tasks.retrieve(tid)
                labels = task.get_labels()
                result[f"{task.name}  (ID: {tid})"] = [lb.name for lb in labels]
            except Exception as e:
                st.warning(f"タスクID {tid} のラベル取得失敗: {e}")
        return result
    except Exception as e:
        st.error(f"ラベル取得エラー: {e}")
        return {}


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
# MLflow 設定
# ---------------------------------------------------------------------------
def init_mlflow(project_name: str, run_name: str) -> bool:
    """MLflow サーバーへの接続確認と環境変数設定。
    Ultralytics の MLflow コールバックが自動でメトリクス・モデルを記録する。
    """
    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.tracking.MlflowClient().search_experiments()  # 接続テスト
        os.environ["MLFLOW_TRACKING_URI"]   = MLFLOW_URI
        os.environ["MLFLOW_EXPERIMENT_NAME"] = project_name
        os.environ["MLFLOW_RUN"]             = run_name
        print(f"[MLflow] 接続OK: {MLFLOW_URI} / {project_name} / {run_name}")
        return True
    except Exception as e:
        print(f"[MLflow] 接続エラー（実験追跡なし）: {e}")
        return False


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

    def _on_fit_epoch_end(trainer) -> None:
        row: dict = {"epoch": trainer.epoch + 1}
        if hasattr(trainer, "metrics") and trainer.metrics:
            for k, v in trainer.metrics.items():
                try:
                    row[k] = float(v)
                except (TypeError, ValueError):
                    pass
        with _train_log_lock:
            _train_state["metrics_history"].append(row)

    _orig_stdout = sys.stdout
    sys.stdout   = _StdoutCapture(_orig_stdout, _train_log_lock, _train_state)

    try:
        mlflow_ok = init_mlflow(project_name, run_name)
        if mlflow_ok:
            _log(f"[MLflow] 実験追跡: {project_name} / {run_name}")
        else:
            _log("[MLflow] スキップ（実験追跡なし）")

        from ultralytics import YOLO

        model = YOLO(model_name)
        model.add_callback("on_train_epoch_end", _on_epoch_end)
        model.add_callback("on_fit_epoch_end", _on_fit_epoch_end)

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

        if mlflow_ok:
            try:
                import mlflow
                # Ultralytics callback がすでに run を close している場合に備えて、
                # 最後の run を取得して model を登録する
                runs = mlflow.search_runs(
                    experiment_names=[project_name],
                    filter_string=f"tags.mlflow.runName = '{run_name}'",
                    max_results=1,
                )
                if not runs.empty:
                    run_id = runs.iloc[0]["run_id"]
                    mv = mlflow.register_model(
                        f"runs:/{run_id}/weights",
                        project_name,
                    )
                    _log(f"[MLflow] モデル登録: {project_name} v{mv.version}")
            except Exception as e:
                _log(f"[MLflow] モデル登録スキップ: {e}")

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
# 推論結果プレビュー描画
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _draw_predictions(json_path: Path):
    """JSONを読み込みバウンディングボックスを描画した RGB 画像配列を返す。失敗時は None。"""
    import cv2

    _COLORS = [
        (78, 207, 244), (244, 168, 78), (126, 207, 78),
        (207, 78, 126), (168, 78, 244), (78, 168, 207),
    ]
    try:
        with open(json_path) as f:
            pred = json.load(f)
        img_path = pred.get("image_path", "")
        if not img_path or not Path(img_path).exists():
            return None
        img = cv2.imread(img_path)
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes = pred.get("boxes", [])
        label_set = list(dict.fromkeys(b["label"] for b in boxes))
        for box in boxes:
            xyxy = box.get("bbox_xyxy", [])
            if len(xyxy) != 4:
                continue
            x1, y1, x2, y2 = [int(v) for v in xyxy]
            label = box["label"]
            conf  = box.get("confidence", 0.0)
            color = _COLORS[label_set.index(label) % len(_COLORS)]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            text = f"{label} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(img, text, (x1 + 2, y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return img, len(boxes), json_path.stem
    except Exception:
        return None


def export_prediction_images(
    out_dir: Path,
    img_format: str = "PNG",
    quality: int = 95,
    target_files: Optional[list[Path]] = None,
    progress_cb=None,
) -> tuple[int, int]:
    """predictions/*.json を描画済み画像として out_dir に書き出す。
    target_files が None のときは predictions/ の全 JSON を対象とする。
    progress_cb(current, total, filename) を渡すと処理ごとに呼ばれる。
    Returns (成功数, スキップ数)
    """
    from PIL import Image as PILImage

    out_dir.mkdir(parents=True, exist_ok=True)
    ext = ".jpg" if img_format == "JPEG" else ".png"
    json_files = target_files if target_files else sorted(PREDICTIONS_DIR.glob("*.json"))
    total = len(json_files)
    success = 0
    skipped = 0
    for i, jf in enumerate(json_files):
        if progress_cb:
            progress_cb(i, total, jf.name)
        result = _draw_predictions(jf)
        if result is None:
            skipped += 1
            continue
        img_arr, _, stem = result
        out_path = out_dir / f"{stem}{ext}"
        pil_img = PILImage.fromarray(img_arr)
        if img_format == "JPEG":
            pil_img.save(out_path, "JPEG", quality=quality)
        else:
            pil_img.save(out_path, "PNG")
        success += 1
    return success, skipped


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
        # key= がある場合は value= を渡さない（session_state が管理するため）
        if "key" in kw:
            v = st.slider(label, lo, hi, step=step, **kw)
        else:
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
        if "key" in kw:
            v = st.number_input(label, lo, hi, **kw)
        else:
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
        if "key" in kw:
            v = st.selectbox(label, options, **kw)
        else:
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
        if "key" in kw:
            v = st.checkbox(label, **kw)
        else:
            v = st.checkbox(label, value=val, **kw)
    with h:
        _ph(name, desc, url)
    return v


# ===========================================================================
# 学習プリセット
# ===========================================================================
_USER_PRESETS_FILE = MODELS_DIR / ".user_presets.json"

_BUILTIN_PRESETS: dict[str, dict] = {
    "🚫 ノーマル (augなし · yolo11s · 100ep · 640px)": {
        "model": "yolo11s", "epochs": 100, "batch": 8,
        "imgsz": 640, "patience": 50, "optimizer": "auto",
        "lr0": 0.01, "cos_lr": False,
        "warmup_epochs": 3, "dropout": 0.0, "weight_decay": 0.0005, "workers": 8,
        "degrees": 0.0, "scale": 0.0, "fliplr": 0.0, "flipud": 0.0,
        "translate": 0.0, "perspective": 0.0,
        "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0,
        "mosaic": 0.0, "mixup": 0.0, "erasing": 0.0, "close_mosaic": 0,
    },
    "⚡ 速度優先 (yolo11n · 50ep · 640px)": {
        "model": "yolo11n", "epochs": 50, "batch": 16,
        "imgsz": 640, "patience": 20, "optimizer": "SGD",
        "lr0": 0.01, "cos_lr": False,
        "mosaic": 0.5, "close_mosaic": 5, "scale": 0.5, "fliplr": 0.5,
    },
    "⚖️ バランス型 (yolo11s · 100ep · 640px)": {
        "model": "yolo11s", "epochs": 100, "batch": 16,
        "imgsz": 640, "patience": 30, "optimizer": "auto",
        "lr0": 0.01, "cos_lr": False,
        "mosaic": 1.0, "close_mosaic": 10, "scale": 0.5, "fliplr": 0.5,
    },
    "🎯 精度優先 (yolo11l · 200ep · 640px)": {
        "model": "yolo11l", "epochs": 200, "batch": 8,
        "imgsz": 640, "patience": 50, "optimizer": "AdamW",
        "lr0": 0.001, "cos_lr": True,
        "mosaic": 1.0, "close_mosaic": 15, "scale": 0.5, "fliplr": 0.5,
    },
    "🔍 小物体向け (yolo11m · 150ep · 640px)": {
        "model": "yolo11m", "epochs": 150, "batch": 8,
        "imgsz": 640, "patience": 30, "optimizer": "AdamW",
        "lr0": 0.001, "cos_lr": True,
        "mosaic": 1.0, "close_mosaic": 10, "scale": 0.3, "fliplr": 0.5,
    },
    "🤖 ロボット視点 (yolo11x · 2000ep · 640px)": {
        "model": "yolo11x", "epochs": 2000, "batch": 8,
        "imgsz": 640, "patience": 50, "optimizer": "auto",
        "lr0": 0.001, "cos_lr": True,
        "warmup_epochs": 10, "dropout": 0.1, "weight_decay": 0.0005, "workers": 8,
        "degrees": 60.0, "scale": 0.5, "fliplr": 0.5, "flipud": 0.1,
        "translate": 0.2, "perspective": 0.0005,
        "hsv_h": 0.02, "hsv_s": 0.7, "hsv_v": 0.7,
        "mosaic": 1.0, "mixup": 0.15, "erasing": 0.2, "close_mosaic": 30,
    },
}

_MODEL_OPTS = [
    "yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x",
    "yolo11n-seg", "yolo11s-seg", "yolo11m-seg", "yolo11l-seg", "yolo11x-seg",
    "yolo11n-pose", "yolo11s-pose", "yolo11m-pose", "yolo11l-pose", "yolo11x-pose",
    "カスタム入力",
]


def _load_user_presets() -> dict:
    try:
        if _USER_PRESETS_FILE.exists():
            return json.loads(_USER_PRESETS_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _save_user_presets(presets: dict) -> None:
    _USER_PRESETS_FILE.parent.mkdir(parents=True, exist_ok=True)
    _USER_PRESETS_FILE.write_text(
        json.dumps(presets, ensure_ascii=False, indent=2), encoding="utf-8"
    )


_PRESET_KEYS: dict[str, tuple] = {
    # (session_state_key, type, default)
    "model":        ("tp_model",        str,   "yolo11s"),
    "epochs":       ("tp_epochs",       int,   100),
    "batch":        ("tp_batch",        int,   8),
    "imgsz":        ("tp_imgsz",        int,   640),
    "patience":     ("tp_patience",     int,   50),
    "optimizer":    ("tp_optimizer",    str,   "auto"),
    "lr0":          ("tp_lr0",          float, 0.01),
    "cos_lr":       ("tp_cos_lr",       bool,  False),
    "warmup_epochs":("tp_warmup_epochs",int,   3),
    "dropout":      ("tp_dropout",      float, 0.0),
    "weight_decay": ("tp_weight_decay", float, 0.0005),
    "workers":      ("tp_workers",      int,   8),
    "degrees":      ("tp_degrees",      float, 0.0),
    "scale":        ("tp_scale",        float, 0.5),
    "fliplr":       ("tp_fliplr",       float, 0.5),
    "flipud":       ("tp_flipud",       float, 0.0),
    "translate":    ("tp_translate",    float, 0.1),
    "perspective":  ("tp_perspective",  float, 0.0),
    "hsv_h":        ("tp_hsv_h",        float, 0.015),
    "hsv_s":        ("tp_hsv_s",        float, 0.7),
    "hsv_v":        ("tp_hsv_v",        float, 0.4),
    "mosaic":       ("tp_mosaic",       float, 1.0),
    "mixup":        ("tp_mixup",        float, 0.0),
    "erasing":      ("tp_erasing",      float, 0.4),
    "close_mosaic": ("tp_close_mosaic", int,   10),
}

# _PRESET_KEYS のデフォルト値を session_state に事前登録（ウィジェット初回表示用）
for _pk, (_pk_ss, _pk_typ, _pk_def) in _PRESET_KEYS.items():
    if _pk_ss not in st.session_state:
        st.session_state[_pk_ss] = _pk_def


def _apply_preset(params: dict) -> None:
    """プリセット値をセッションステート（widget key）に書き込む。"""
    for k, v in params.items():
        if k not in _PRESET_KEYS:
            continue
        ss_key, typ, _ = _PRESET_KEYS[k]
        if k == "model":
            if v in _MODEL_OPTS:
                st.session_state[ss_key] = v
        else:
            st.session_state[ss_key] = typ(v)


def _collect_current_params() -> dict:
    """現在のウィジェット値からプリセット保存用 dict を生成する。"""
    result = {}
    for k, (ss_key, typ, default) in _PRESET_KEYS.items():
        result[k] = typ(st.session_state.get(ss_key, default))
    return result


# ===========================================================================
# UI レイアウト
# ===========================================================================

# ---------------------------------------------------------------------------
# パイプライン状態ヘルパー
# ---------------------------------------------------------------------------
def _get_pipeline_status() -> dict:
    yaml_exists  = len(list(DATA_DIR.rglob("data.yaml"))) > 0
    model_exists = len(list(MODELS_DIR.rglob("*.pt"))) > 0
    pred_exists  = len(list(PREDICTIONS_DIR.glob("*.json"))) > 0
    training_now = _train_state.get("running", False)
    return {
        "step1": "complete",
        "step2": "complete" if yaml_exists  else "pending",
        "step3": "active"   if training_now else ("complete" if model_exists else "pending"),
        "step4": "complete" if pred_exists  else "pending",
    }


def _pf_html(ps: dict) -> str:
    steps = [
        ("step1", "📝", "STEP 1", "アノテーション"),
        ("step2", "📁", "STEP 2", "データ取込"),
        ("step3", "🚀", "STEP 3", "モデル学習"),
        ("step4", "🔭", "STEP 4", "推論・評価"),
    ]
    parts = []
    for i, (key, icon, label, name) in enumerate(steps):
        cls = ps[key]
        icon_str = "✅" if cls == "complete" else ("⏳" if cls == "active" else "○")
        parts.append(
            f'<div class="pf-step {cls}">'
            f'<div class="pf-label">{label}</div>'
            f'<div class="pf-name">{icon} {name}</div>'
            f'<div class="pf-icon">{icon_str}</div>'
            f'</div>'
        )
        if i < len(steps) - 1:
            parts.append('<div class="pf-arrow">→</div>')
    return '<div class="pipeline-flow">' + "".join(parts) + '</div>'


# --- ヘッダー ---
_ps = _get_pipeline_status()
st.markdown(f"""
<div style="border-bottom:1px solid var(--border); padding-bottom:12px; margin-bottom:20px;">
  <h1 style="color:var(--accent); font-family:'JetBrains Mono',monospace; font-size:1.6rem; margin:0 0 4px;">
    🔬 detection_dev_ui
  </h1>
  <p style="color:var(--text-muted); font-size:.82rem; margin:0 0 12px;">
    CVAT → YOLO → MLflow → FiftyOne 統合ダッシュボード
  </p>
  {_pf_html(_ps)}
</div>
""", unsafe_allow_html=True)

# --- サイドバー ---
with st.sidebar:
    st.markdown("### 📊 現在の状態")
    _ds_count  = len([d for d in DATA_DIR.iterdir() if d.is_dir()]) if DATA_DIR.exists() else 0
    _mdl_count = len(list(MODELS_DIR.rglob("*.pt"))) if MODELS_DIR.exists() else 0
    _prd_count = len(list(PREDICTIONS_DIR.glob("*.json"))) if PREDICTIONS_DIR.exists() else 0
    st.markdown(f"""
<div class="sidebar-stat"><span class="ss-label">📂 データセット</span><span class="ss-value">{_ds_count}</span></div>
<div class="sidebar-stat"><span class="ss-label">🤖 学習済みモデル</span><span class="ss-value">{_mdl_count}</span></div>
<div class="sidebar-stat"><span class="ss-label">📋 推論結果</span><span class="ss-value">{_prd_count}</span></div>
""", unsafe_allow_html=True)
    st.markdown("---")
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
    check_service(f"{MLFLOW_URI}/health", "MLflow")

    st.markdown("---")
    st.markdown("#### 📁 ディレクトリ")
    st.code(f"data/        {DATA_DIR}\nmodels/      {MODELS_DIR}\npredictions/ {PREDICTIONS_DIR}", language="text")

    st.markdown("---")
    st.markdown("#### 🔗 クイックリンク")
    st.markdown(f"[📝 CVAT UI]({CVAT_WEB})", unsafe_allow_html=False)
    st.markdown(f"[📊 MLflow UI]({MLFLOW_WEB})", unsafe_allow_html=False)
    if st.session_state.fiftyone_port:
        fo_url = f"http://localhost:{st.session_state.fiftyone_port}"
        st.markdown(f"[🔭 FiftyOne App]({fo_url})", unsafe_allow_html=False)

    st.markdown("---")
    # ── テーマ設定（サイドバー最下部） ──
    with st.expander("🎨 テーマ設定", expanded=False):
        _user_themes_db = _load_user_themes()
        _preset_names   = list(PRESET_THEMES.keys())
        _user_names     = list(_user_themes_db.keys())
        _all_names      = _preset_names + _user_names

        _cur_theme = st.session_state.theme_name
        _cur_idx   = _all_names.index(_cur_theme) if _cur_theme in _all_names else 0

        st.markdown("**プリセット**")
        _selected_preset = st.radio(
            "プリセットテーマ",
            _preset_names,
            index=_cur_idx if _cur_theme in _preset_names else 0,
            key="theme_preset_radio",
            label_visibility="collapsed",
        )
        if _user_names:
            st.markdown("**保存済みカスタム**")
            _selected_user = st.radio(
                "カスタムテーマ",
                _user_names,
                index=(_user_names.index(_cur_theme) if _cur_theme in _user_names else 0),
                key="theme_user_radio",
                label_visibility="collapsed",
            )
        else:
            _selected_user = None

        if "theme_radio_last" not in st.session_state:
            st.session_state.theme_radio_last = "preset"

        _apply_col1, _apply_col2 = st.columns(2)
        with _apply_col1:
            if st.button("▶ プリセット適用", key="apply_preset_btn"):
                st.session_state.theme_name = _selected_preset
                st.session_state.theme_radio_last = "preset"
                st.rerun()
        with _apply_col2:
            if _selected_user and st.button("▶ カスタム適用", key="apply_user_btn"):
                st.session_state.theme_name = _selected_user
                st.session_state.theme_radio_last = "user"
                st.rerun()

        st.markdown("---")
        st.markdown("**カスタムテーマを作成・編集**")

        _base_sel = st.selectbox(
            "ベーステーマ（編集の起点）",
            _preset_names,
            key="theme_base_select",
        )
        _base = PRESET_THEMES[_base_sel].copy()

        _edited: dict = {}
        for _field, _label in THEME_EDIT_FIELDS:
            _edited[_field] = st.color_picker(_label, _base[_field], key=f"cp_{_field}")

        _merged = _base.copy()
        _merged.update(_edited)

        _new_name = st.text_input("カスタムテーマ名", placeholder="例: マイテーマ", key="custom_theme_name")
        _sc1, _sc2 = st.columns(2)
        with _sc1:
            if st.button("💾 保存", key="save_theme_btn"):
                if _new_name.strip():
                    _user_themes_db[_new_name.strip()] = _merged
                    _save_user_themes(_user_themes_db)
                    st.session_state.theme_name = _new_name.strip()
                    st.success(f"保存: {_new_name.strip()}")
                    st.rerun()
                else:
                    st.warning("テーマ名を入力してください")
        with _sc2:
            if _cur_theme in _user_themes_db:
                if st.button("🗑 削除", key="del_theme_btn"):
                    del _user_themes_db[_cur_theme]
                    _save_user_themes(_user_themes_db)
                    st.session_state.theme_name = "ライト シンプル"
                    st.rerun()

        _t = _get_active_theme()
        st.markdown(
            f'<div style="margin-top:8px;font-size:.75rem;color:var(--text-muted);">'
            f'現在: <span style="color:var(--accent);font-family:monospace;">{_cur_theme}</span> '
            f'<span class="theme-swatch" style="background:{_t["bg_app"]};"></span>'
            f'<span class="theme-swatch" style="background:{_t["accent"]};"></span>'
            f'<span class="theme-swatch" style="background:{_t["success"]};"></span>'
            f'</div>',
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# タブ構成
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📤 Step1: データ取込",
    "🚀 Step2: モデル学習",
    "🔭 Step3: 推論・評価",
    "📁 データ管理",
    "📚 トピックス",
])

# ===========================================================================
# タブ1: CVAT エクスポート
# ===========================================================================
with tab1:
    st.markdown("""
<div class="step-banner">
  <div class="sb-title">📤 STEP 1: データ取込</div>
  <div class="sb-prev">← 事前準備: CVATでアノテーションを完了させてください (http://localhost:8080)</div>
  <div class="sb-desc">→ ここでやること: CVATタスクをエクスポート → YOLOデータセット形式に変換</div>
</div>""", unsafe_allow_html=True)
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
        # 進捗テーブル
        if tasks:
            st.markdown("#### 📊 アノテーション進捗")
            import pandas as pd
            df = pd.DataFrame(tasks)[["id","name","status","assignee","size"]]
            df.columns = ["ID","タスク名","ステータス","担当者","画像数"]
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.markdown("---")

        # 複数タスク選択
        task_options = {f"[{t['id']}] {t['name']} ({t['size']} items)": t["id"] for t in tasks}
        selected_labels = st.multiselect(
            "エクスポートするタスクを選択（複数可）",
            list(task_options.keys()),
            default=[list(task_options.keys())[0]] if task_options else [],
        )
        selected_ids = [task_options[lbl] for lbl in selected_labels]

        _first_id = selected_ids[0] if selected_ids else "multi"
        export_dir_name = st.text_input(
            "エクスポート先サブディレクトリ名",
            value=f"dataset_{_first_id}_{datetime.now():%Y%m%d}",
        )

        # ─── Step 1: CVAT for images 1.1 エクスポート ───────────────────────
        st.markdown("#### Step 1: CVATエクスポート")
        if st.button("⬇️ エクスポート実行 (CVAT for images 1.1)", type="primary",
                     use_container_width=True,
                     disabled=len(selected_ids) == 0):
            if not selected_ids:
                st.warning("エクスポートするタスクを選択してください。")
            else:
                out_dir = DATA_DIR / export_dir_name
                out_dir.mkdir(parents=True, exist_ok=True)
                all_raw_dirs = []
                with st.spinner("エクスポート中…（最大3分×タスク数）"):
                    for task_id in selected_ids:
                        task_out = out_dir / f"task_{task_id}"
                        task_out.mkdir(parents=True, exist_ok=True)
                        raw_dir = export_cvat_task_raw(task_id, task_out)
                        if raw_dir:
                            all_raw_dirs.append(raw_dir)
                            st.success(f"✅ タスク {task_id} エクスポート完了: `{raw_dir}`")
                        else:
                            st.error(f"タスク {task_id} のエクスポートに失敗しました")

                if all_raw_dirs:
                    # 複数タスクの場合は最初のrawディレクトリをメインとして設定
                    # マージ: 全rawディレクトリのXMLを統合して最初のrawを基準にする
                    if len(all_raw_dirs) == 1:
                        merged_raw = all_raw_dirs[0]
                    else:
                        import shutil as _shutil
                        merged_raw = out_dir / "merged_raw"
                        merged_raw.mkdir(parents=True, exist_ok=True)
                        for src_raw in all_raw_dirs:
                            for item in src_raw.rglob("*"):
                                if item.is_file():
                                    rel = item.relative_to(src_raw)
                                    dst = merged_raw / rel
                                    dst.parent.mkdir(parents=True, exist_ok=True)
                                    if not dst.exists():
                                        _shutil.copy2(item, dst)
                        st.success(f"✅ {len(all_raw_dirs)} タスクを統合: `{merged_raw}`")

                    st.session_state.cvat_raw_dir = str(merged_raw)
                    st.session_state.cvat_xml_info = None
                    xml_info = parse_cvat_xml(merged_raw)
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

    with st.expander("📁 ローカルからデータを直接追加（CVATなし）"):
        import io as _io_ul
        st.caption(
            "CVATを経由せず、手元の画像やYOLOデータセットZIPを直接 data/ に追加します。"
        )
        _ul_mode = st.radio(
            "アップロード形式",
            ["🗜 ZIPファイル（YOLOデータセット）", "🖼 画像ファイル（複数可）"],
            horizontal=True,
            key="ul_mode",
        )
        _ul_dir_name = st.text_input(
            "保存先ディレクトリ名（data/ 以下に作成）",
            value=f"upload_{datetime.now():%Y%m%d_%H%M}",
            key="ul_dir_name",
        )

        if _ul_mode == "🗜 ZIPファイル（YOLOデータセット）":
            _ul_zip = st.file_uploader(
                "YOLOデータセット ZIP（images/, labels/, data.yaml を含む）",
                type=["zip"],
                key="ul_zip",
            )
            if _ul_zip:
                st.caption(f"選択中: {_ul_zip.name}  ({_ul_zip.size / 1024 / 1024:.1f} MB)")
                if st.button("📤 展開して data/ に保存", key="ul_zip_btn",
                             type="primary", use_container_width=True):
                    _ul_out = DATA_DIR / _ul_dir_name
                    _ul_out.mkdir(parents=True, exist_ok=True)
                    with zipfile.ZipFile(_io_ul.BytesIO(_ul_zip.read()), "r") as _zf:
                        _zf.extractall(_ul_out)
                    st.success(f"✅ 展開完了: `{_ul_out}`")
                    _ul_yamls = list(_ul_out.rglob("data.yaml"))
                    if _ul_yamls:
                        st.info(f"🗂 data.yaml: `{_ul_yamls[0]}`")
                    else:
                        st.warning("data.yaml が見つかりません。Step2で手動入力が必要です。")
        else:
            _ul_imgs = st.file_uploader(
                "画像ファイル（複数選択可）",
                type=["jpg", "jpeg", "png", "bmp", "tiff"],
                accept_multiple_files=True,
                key="ul_imgs",
            )
            _ul_split = st.radio(
                "保存先スプリット", ["train", "val"], horizontal=True, key="ul_split"
            )
            if _ul_imgs:
                st.caption(f"選択中: {len(_ul_imgs)} ファイル")
                _ul_dst_preview = f"data/{_ul_dir_name}/images/{_ul_split}/"
                if st.button(f"📤 {_ul_dst_preview} に保存", key="ul_imgs_btn",
                             type="primary", use_container_width=True):
                    _ul_out = DATA_DIR / _ul_dir_name / "images" / _ul_split
                    _ul_out.mkdir(parents=True, exist_ok=True)
                    for _f in _ul_imgs:
                        (_ul_out / _f.name).write_bytes(_f.getbuffer())
                    st.success(f"✅ {len(_ul_imgs)} ファイルを保存: `{_ul_out}`")
                    st.info("アノテーションを付与する場合は CVATにアップロード後、Step1からエクスポートしてください。")


    # ── +α: チーム共通ラベルエクスポート ──────────────────────────────────────
    st.markdown("---")
    with st.expander("🏷️ チーム共通ラベルのエクスポート（+α）", expanded=False):
        st.caption("複数のCVATタスクからラベルを収集し、チーム内で共有できる形式でダウンロードできます。")

        _le_tasks = st.session_state.cvat_tasks
        if not _le_tasks:
            st.info("先に「タスク一覧を取得」を実行してください。")
        else:
            _le_opts = {f"{t['name']}  (ID: {t['id']})": t["id"] for t in _le_tasks}
            _le_selected = st.multiselect(
                "ラベルを収集するタスクを選択（複数可）",
                options=list(_le_opts.keys()),
                key="le_task_select",
            )

            if _le_selected:
                if st.button("🔍 ラベルを取得", key="le_fetch_btn", use_container_width=True):
                    _le_ids = [_le_opts[k] for k in _le_selected]
                    with st.spinner("ラベル取得中..."):
                        st.session_state["le_labels_by_task"] = fetch_cvat_task_labels(_le_ids)

        if st.session_state.get("le_labels_by_task"):
            _le_by_task: dict = st.session_state["le_labels_by_task"]

            # 全ラベルを重複排除して収集
            _le_all: list[str] = []
            for _lbls in _le_by_task.values():
                for _l in _lbls:
                    if _l not in _le_all:
                        _le_all.append(_l)

            st.markdown("**含めるラベルを選択してください：**")
            # タスク別ラベルは参考表示のみ（チェックボックスは重複排除済みリストで一度だけ描画）
            for _tn, _lbls in _le_by_task.items():
                st.caption(f"📋 {_tn}：{', '.join(_lbls)}")
            st.markdown("---")
            _le_cols = st.columns(3)
            for _ci, _l in enumerate(_le_all):
                with _le_cols[_ci % 3]:
                    if f"le_chk_{_l}" not in st.session_state:
                        st.session_state[f"le_chk_{_l}"] = True
                    st.checkbox(_l, key=f"le_chk_{_l}")

            _le_chosen = [_l for _l in _le_all if st.session_state.get(f"le_chk_{_l}", True)]

            if _le_chosen:
                st.markdown(f"**選択中: {len(_le_chosen)} ラベル** — `{', '.join(_le_chosen)}`")
                _le_c1, _le_c2, _le_c3 = st.columns(3)
                with _le_c1:
                    _le_yaml = "names:\n" + "".join(f"  - {l}\n" for l in _le_chosen)
                    st.download_button(
                        "📥 YAML形式",
                        data=_le_yaml,
                        file_name="labels.yaml",
                        mime="text/yaml",
                        key="le_dl_yaml",
                        use_container_width=True,
                    )
                with _le_c2:
                    st.download_button(
                        "📥 TXT形式",
                        data="\n".join(_le_chosen),
                        file_name="labels.txt",
                        mime="text/plain",
                        key="le_dl_txt",
                        use_container_width=True,
                    )
                with _le_c3:
                    _le_cvat_json = json.dumps(
                        [{"name": l, "attributes": [], "type": "any", "sublabels": []} for l in _le_chosen],
                        ensure_ascii=False, indent=2
                    )
                    st.download_button(
                        "📥 CVAT JSON形式",
                        data=_le_cvat_json,
                        file_name="labels_cvat.json",
                        mime="application/json",
                        key="le_dl_cvat",
                        use_container_width=True,
                    )

                st.markdown("---")
                st.markdown("##### CVATで新規タスクを作成するときの手順")
                st.markdown(f"""
<div class="step-banner">
  <div class="sb-title">📋 ラベルの共有方法</div>
  <div class="sb-desc">ダウンロードした <code>labels_cvat.json</code> を使うと、CVATのラベル設定を一括で読み込めます。</div>
</div>""", unsafe_allow_html=True)
                st.markdown("""
1. `http://localhost:8080` にアクセスしてログイン
2. **Tasks** → **+** ボタンで新規タスク作成画面を開く
3. タスク名・画像等を設定後、**Labels** セクションを開く
4. **Raw** タブをクリックし、ダウンロードした `labels_cvat.json` の内容を貼り付ける
5. **Done** をクリックしてラベルを確定する
""")
                st.info("💡 チームメンバー全員が同じ `labels_cvat.json` を使うことで、ラベル名の表記ゆれを防げます。")
            else:
                st.warning("1つ以上のラベルを選択してください。")

# ===========================================================================
# タブ2: YOLO 学習
# ===========================================================================
with tab2:
    _yaml_count = len(list(DATA_DIR.rglob("data.yaml")))
    _prev_info2 = (f"← 前のステップ: ✅ data.yaml が {_yaml_count} 件あります"
                   if _yaml_count > 0 else "← 前のステップ: ⚠ Step1でデータセットを先に生成してください")
    st.markdown(f"""
<div class="step-banner">
  <div class="sb-title">🚀 STEP 2: モデル学習</div>
  <div class="sb-prev">{_prev_info2}</div>
  <div class="sb-desc">→ ここでやること: モデルサイズ・学習パラメータを設定して学習開始</div>
</div>""", unsafe_allow_html=True)
    st.markdown('<div class="pipeline-card"><h3>🚀 YOLO 学習設定</h3>', unsafe_allow_html=True)

    # ── プリセット ───────────────────────────────────────────────────────────
    _user_presets  = _load_user_presets()
    _all_presets   = {**_BUILTIN_PRESETS,
                      **{f"👤 {k}": v for k, v in _user_presets.items()}}
    _PRESET_NONE   = "（選択してください）"

    st.markdown("##### 📋 学習プリセット")
    _pr1, _pr2, _pr3 = st.columns([4, 1, 2])
    with _pr1:
        _preset_sel = st.selectbox(
            "プリセット",
            [_PRESET_NONE] + list(_all_presets.keys()),
            key="preset_sel",
            label_visibility="collapsed",
        )
    with _pr2:
        if st.button("▶ 適用", key="preset_apply", use_container_width=True,
                     disabled=(_preset_sel == _PRESET_NONE)):
            _apply_preset(_all_presets[_preset_sel])
    with _pr3:
        if st.button("💾 現在の設定を保存", key="preset_save_btn", use_container_width=True):
            st.session_state["preset_save_mode"] = True

    if st.session_state.get("preset_save_mode", False):
        with st.container(border=True):
            st.caption("保存するプリセット名を入力してください")
            _sv1, _sv2, _sv3 = st.columns([4, 1, 1])
            with _sv1:
                _new_pname = st.text_input(
                    "プリセット名", key="preset_new_name",
                    placeholder="例: ペッパー物体検出用", label_visibility="collapsed",
                )
            with _sv2:
                if st.button("✅ 保存", key="preset_save_confirm", use_container_width=True):
                    if _new_pname.strip():
                        _ups = _load_user_presets()
                        _ups[_new_pname.strip()] = _collect_current_params()
                        _save_user_presets(_ups)
                        st.session_state["preset_save_mode"] = False
                        st.toast(f"✅ プリセット「{_new_pname.strip()}」を保存しました", icon="💾")
                        st.rerun()
                    else:
                        st.warning("プリセット名を入力してください")
            with _sv3:
                if st.button("✕ キャンセル", key="preset_save_cancel", use_container_width=True):
                    st.session_state["preset_save_mode"] = False
                    st.rerun()

    if _user_presets:
        with st.expander("🗂 ユーザープリセット管理"):
            _editing = st.session_state.get("preset_editing_name", None)

            for _uname, _uparams in list(_user_presets.items()):
                st.markdown(f"**{_uname}**")
                _up1, _up2, _up3, _up4 = st.columns([1, 1, 1, 1])
                _param_summary = (
                    f"`{_uparams.get('model','?')} · {_uparams.get('epochs','?')}ep · "
                    f"{_uparams.get('imgsz','?')}px`"
                )
                st.caption(_param_summary)
                with _up1:
                    if st.button("▶ 適用", key=f"upr_apply_{_uname}", use_container_width=True):
                        _apply_preset(_uparams)
                with _up2:
                    if st.button("✏️ 編集", key=f"upr_edit_{_uname}", use_container_width=True):
                        st.session_state["preset_editing_name"] = _uname
                        st.session_state["preset_editing_vals"] = dict(_uparams)
                        st.rerun()
                with _up3:
                    if st.button("🗑 削除", key=f"upr_del_{_uname}", use_container_width=True):
                        _ups = _load_user_presets()
                        _ups.pop(_uname, None)
                        _save_user_presets(_ups)
                        if st.session_state.get("preset_editing_name") == _uname:
                            st.session_state.pop("preset_editing_name", None)
                            st.session_state.pop("preset_editing_vals", None)
                        st.rerun()
                st.markdown("---")

            # ── 編集フォーム ──────────────────────────────────────────────────
            if _editing and _editing in _user_presets:
                _ev = st.session_state.get("preset_editing_vals", {})
                st.markdown(f"#### ✏️ 編集中: **{_editing}**")
                _OPTS_OPT = ["auto","SGD","Adam","AdamW","NAdam","RAdam"]
                _ef1, _ef2, _ef3 = st.columns(3)
                with _ef1:
                    _ev["model"]   = st.selectbox("モデル", _MODEL_OPTS,
                        index=_MODEL_OPTS.index(_ev.get("model","yolo11s")) if _ev.get("model","yolo11s") in _MODEL_OPTS else 1,
                        key="pe_model")
                    _ev["epochs"]  = st.number_input("エポック数", 1, 5000, int(_ev.get("epochs",100)), step=10, key="pe_epochs")
                    _ev["batch"]   = st.select_slider("バッチサイズ", [-1,4,8,16,32,64,128],
                        value=_ev.get("batch",8), key="pe_batch")
                with _ef2:
                    _ev["imgsz"]   = st.select_slider("imgsz", [320,416,512,640,768,1024,1280],
                        value=int(_ev.get("imgsz",640)) if int(_ev.get("imgsz",640)) in [320,416,512,640,768,1024,1280] else 640,
                        key="pe_imgsz")
                    _ev["patience"]= st.number_input("patience", 0, 1000, int(_ev.get("patience",50)), step=10, key="pe_patience")
                    _ev["optimizer"]= st.selectbox("optimizer", _OPTS_OPT,
                        index=_OPTS_OPT.index(_ev.get("optimizer","auto")) if _ev.get("optimizer","auto") in _OPTS_OPT else 0,
                        key="pe_optimizer")
                with _ef3:
                    _ev["lr0"]     = st.number_input("lr0", 1e-5, 1.0, float(_ev.get("lr0",0.01)), format="%.5f", step=0.001, key="pe_lr0")
                    _ev["cos_lr"]  = st.checkbox("cos_lr", value=bool(_ev.get("cos_lr",False)), key="pe_cos_lr")
                    _ev["warmup_epochs"] = st.number_input("warmup_epochs", 0, 50, int(_ev.get("warmup_epochs",3)), key="pe_warmup")
                    _ev["dropout"] = st.slider("dropout", 0.0, 0.5, float(_ev.get("dropout",0.0)), step=0.05, key="pe_dropout")

                st.markdown("**拡張設定**")
                _ea1, _ea2, _ea3 = st.columns(3)
                with _ea1:
                    _ev["degrees"]  = st.slider("degrees", 0.0, 180.0, float(_ev.get("degrees",0.0)), step=1.0, key="pe_degrees")
                    _ev["scale"]    = st.slider("scale", 0.0, 0.9, float(_ev.get("scale",0.5)), step=0.05, key="pe_scale")
                    _ev["mosaic"]   = st.slider("mosaic", 0.0, 1.0, float(_ev.get("mosaic",1.0)), step=0.05, key="pe_mosaic")
                with _ea2:
                    _ev["fliplr"]   = st.slider("fliplr", 0.0, 1.0, float(_ev.get("fliplr",0.5)), step=0.05, key="pe_fliplr")
                    _ev["flipud"]   = st.slider("flipud", 0.0, 1.0, float(_ev.get("flipud",0.0)), step=0.05, key="pe_flipud")
                    _ev["mixup"]    = st.slider("mixup", 0.0, 1.0, float(_ev.get("mixup",0.0)), step=0.05, key="pe_mixup")
                with _ea3:
                    _ev["hsv_h"]    = st.slider("hsv_h", 0.0, 0.1, float(_ev.get("hsv_h",0.015)), step=0.005, key="pe_hsv_h")
                    _ev["hsv_s"]    = st.slider("hsv_s", 0.0, 1.0, float(_ev.get("hsv_s",0.7)), step=0.05, key="pe_hsv_s")
                    _ev["hsv_v"]    = st.slider("hsv_v", 0.0, 1.0, float(_ev.get("hsv_v",0.4)), step=0.05, key="pe_hsv_v")
                _eb1, _eb2, _eb3 = st.columns(3)
                with _eb1:
                    _ev["translate"]  = st.slider("translate", 0.0, 0.9, float(_ev.get("translate",0.1)), step=0.05, key="pe_translate")
                with _eb2:
                    _ev["erasing"]    = st.slider("erasing", 0.0, 0.9, float(_ev.get("erasing",0.4)), step=0.05, key="pe_erasing")
                with _eb3:
                    _ev["close_mosaic"]= st.number_input("close_mosaic", 0, 200, int(_ev.get("close_mosaic",10)), step=5, key="pe_close")

                _ec1, _ec2 = st.columns(2)
                with _ec1:
                    if st.button("✅ 変更を保存", key="pe_save", use_container_width=True, type="primary"):
                        _ups = _load_user_presets()
                        _ups[_editing] = dict(_ev)
                        _save_user_presets(_ups)
                        st.session_state.pop("preset_editing_name", None)
                        st.session_state.pop("preset_editing_vals", None)
                        st.toast(f"✅ プリセット「{_editing}」を更新しました", icon="💾")
                        st.rerun()
                with _ec2:
                    if st.button("✕ キャンセル", key="pe_cancel", use_container_width=True):
                        st.session_state.pop("preset_editing_name", None)
                        st.session_state.pop("preset_editing_vals", None)
                        st.rerun()

    st.markdown("---")

    # ── 基本設定 ────────────────────────────────────────────────────────────
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        _model_preset = st.selectbox(
            "モデル",
            _MODEL_OPTS,
            key="tp_model",
        )
    with col_b:
        epochs = st.number_input("エポック数", min_value=1, max_value=5000, step=10,
                                 key="tp_epochs")
    with col_c:
        batch_size = st.select_slider(
            "バッチサイズ",
            options=[-1, 4, 8, 16, 32, 64, 128],
            help="-1 = AutoBatch",
            key="tp_batch",
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
        mlflow_project = st.text_input("MLflow プロジェクト名", value="YOLO-Detection")
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
                               name="imgsz", key="tp_imgsz",
                               desc="学習・推論時の画像サイズ（ピクセル）。大きいほど精度が上がるが計算コストが増加する。")
            patience    = _nw("patience（0=無効）", 0, 1000, 50, step=10,
                               name="patience", key="tp_patience",
                               desc="EarlyStopping の待機エポック数。N エポック間 val metrics が改善しなければ自動終了。0 で無効。")
            save_period = _nw("save_period（0=無効）", 0, 500, 0, step=10,
                               name="save_period",
                               desc="N エポックごとにチェックポイントを保存する間隔。0 で無効。長期学習での途中確認に便利。")
            workers     = _nw("workers", 0, 32, 8, step=1,
                               name="workers", key="tp_workers",
                               desc="DataLoader の CPU ワーカースレッド数。多すぎるとメモリ不足になることがある。")
        with _oc2:
            optimizer   = _selw("optimizer", ["auto","SGD","Adam","AdamW","NAdam","RAdam"], 0,
                                 name="optimizer", key="tp_optimizer",
                                 desc="`auto` はモデルに応じて自動選択。細かく制御する場合は SGD または AdamW 推奨。")
            lr0         = _nw("lr0（初期学習率）", 1e-5, 1.0, 0.01, format="%.5f", step=0.001,
                               name="lr0", key="tp_lr0",
                               desc="初期学習率。SGD では 0.01、Adam/AdamW では 0.001 が一般的な推奨値。")
            lrf         = _nw("lrf（最終LR係数）", 1e-4, 1.0, 0.01, format="%.4f", step=0.001,
                               name="lrf",
                               desc="学習率スケジューラの終端係数。最終学習率 = `lr0 × lrf`。")
            cos_lr      = _ckw("cos_lr（コサイン学習率）", False,
                                name="cos_lr", key="tp_cos_lr",
                                desc="True でコサイン学習率スケジューラを使用。学習後半を滑らかに減衰させる。")
        with _oc3:
            momentum    = _nw("momentum（SGD/Adam β1）", 0.5, 0.999, 0.937, format="%.3f", step=0.01,
                               name="momentum",
                               desc="SGD のモメンタム係数、または Adam 系の β1 パラメータ。")
            warmup_epochs = _nw("warmup_epochs", 0, 50, 3, step=1,
                                 name="warmup_epochs", key="tp_warmup_epochs",
                                 desc="ウォームアップのエポック数。最初の N エポックで学習率を 0 から lr0 まで徐々に増加させる。")
            warmup_momentum = _nw("warmup_momentum", 0.0, 1.0, 0.8, format="%.2f", step=0.05,
                                   name="warmup_momentum",
                                   desc="ウォームアップ中の初期モメンタム値。")
            warmup_bias_lr  = _nw("warmup_bias_lr", 0.0, 1.0, 0.1, format="%.3f", step=0.01,
                                   name="warmup_bias_lr",
                                   desc="ウォームアップ中のバイアス層の学習率。")
        with _oc4:
            weight_decay = _nw("weight_decay", 0.0, 0.01, 0.0005, format="%.5f", step=0.0001,
                                name="weight_decay", key="tp_weight_decay",
                                desc="L2 正則化（重み減衰）の強度。過学習の抑制に効果的。")
            dropout      = _sw("dropout", 0.0, 0.5, 0.0, step=0.05,
                                name="dropout", key="tp_dropout",
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
                           name="degrees", key="tp_degrees",
                           desc="画像をランダムに回転させる角度範囲（±degrees°）。0 で無効。ロボット視点など姿勢が変化する環境で有効。",
                           disabled=not _has_box)
            shear   = _sw("shear（せん断 ±°）", 0.0, 10.0, 0.0, step=0.5,
                           name="shear",
                           desc="せん断変形（ずれ歪み）の角度範囲（±degrees°）。画像を平行四辺形状に歪める。",
                           disabled=not _has_box)
        with _g2:
            scale     = _sw("scale（拡大縮小）", 0.0, 0.9, 0.5, step=0.05,
                             name="scale", key="tp_scale",
                             desc="ランダムスケーリングの変化幅。0.5 なら画像サイズが ×0.5〜×1.5 の範囲で変化。距離・解像度の変動に対応。",
                             disabled=not _has_box)
            translate = _sw("translate（平行移動）", 0.0, 0.9, 0.1, step=0.05,
                             name="translate", key="tp_translate",
                             desc="水平・垂直方向の平行移動量（画像サイズ比）。物体が画像端にある場合への対応。",
                             disabled=not _has_box)
        with _g3:
            fliplr = _sw("fliplr（左右反転）", 0.0, 1.0, 0.5, step=0.05,
                          name="fliplr", key="tp_fliplr",
                          desc="水平（左右）反転の確率。文字・数字など向きが意味を持つタスクでは 0.0 を推奨。")
            flipud = _sw("flipud（上下反転）", 0.0, 1.0, 0.0, step=0.05,
                          name="flipud", key="tp_flipud",
                          desc="垂直（上下）反転の確率。重力方向が重要なタスクでは 0.0 を推奨。")
        with _g4:
            perspective = _nw("perspective（透視変換）", 0.0, 0.001, 0.0,
                               format="%.4f", step=0.0001,
                               name="perspective", key="tp_perspective",
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
                         name="hsv_h", key="tp_hsv_h",
                         desc="HSV 色空間の色相（Hue）の変動量。照明条件の変化や異なる色帯域への汎化に効果的。")
        with _c2:
            hsv_s = _sw("hsv_s（彩度変動）", 0.0, 1.0, 0.7, step=0.05,
                         name="hsv_s", key="tp_hsv_s",
                         desc="HSV 色空間の彩度（Saturation）の変動量。色の鮮やかさをランダムに変化させる。")
        with _c3:
            hsv_v = _sw("hsv_v（明度変動）", 0.0, 1.0, 0.4, step=0.05,
                         name="hsv_v", key="tp_hsv_v",
                         desc="HSV 色空間の明度（Value）の変動量。屋内外の照明差や露出変化に対応させる。")

        # ── 合成拡張 ─────────────────────────────────────────────────────────
        st.markdown("##### 🔀 合成拡張")
        _m1, _m2, _m3, _m4 = st.columns(4)
        with _m1:
            mosaic = _sw("mosaic（4 画像合成）", 0.0, 1.0,
                          1.0 if _has_box else 0.0, step=0.05,
                          name="mosaic", key="tp_mosaic",
                          desc="4 枚の画像をランダムにモザイク結合する確率。小物体の検出精度向上に非常に効果的。detect/segment/pose 向け。",
                          disabled=not _has_box)
            close_mosaic = _nw("close_mosaic（終盤N エポックOFF）", 0, 200, 10, step=5,
                                name="close_mosaic", key="tp_close_mosaic",
                                desc="最後の N エポックでモザイク拡張を OFF にする。学習終盤に拡張なしの本来の分布で収束させ精度を安定させる。",
                                url=_DOC_AUG, disabled=not _has_box)
        with _m2:
            mixup  = _sw("mixup", 0.0, 1.0, 0.0, step=0.05,
                          name="mixup", key="tp_mixup",
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
                           name="erasing", key="tp_erasing",
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
                _train_state["metrics_history"] = []

            t = threading.Thread(
                target=_train_worker,
                args=(data_yaml_path, model_name, epochs, batch_size,
                      mlflow_project, run_name, _train_kwargs),
                daemon=True,
            )
            t.start()
            st.session_state.training_notified = False   # 新規学習開始 → 通知リセット
            st.rerun()

    # --- _train_state → st.session_state に同期 ---
    with _train_log_lock:
        st.session_state.training_log = list(_train_state["log"])
        st.session_state.training_progress = _train_state["progress"]
        st.session_state.training_running = _train_state["running"]
        st.session_state.training_metrics_history = list(_train_state["metrics_history"])
        if _train_state["error"]:
            st.session_state.training_error = _train_state["error"]
        if _train_state["model_path"]:
            st.session_state.last_model_path = _train_state["model_path"]

    # --- 学習完了トースト（1回だけ） ---
    if (st.session_state.training_progress == 100
            and not st.session_state.training_running
            and not st.session_state.training_notified):
        st.toast("🎉 学習が完了しました！", icon="✅")
        st.balloons()
        st.session_state.training_notified = True

    # --- 進捗表示 ---
    if st.session_state.training_running:
        # ── 学習中: プログレスバー＋リアルタイムグラフ＋自動スクロールログ ──
        prog = st.session_state.training_progress
        st.progress(prog / 100, text=f"進捗: {prog}%")

        _mh = st.session_state.training_metrics_history
        if _mh:
            import pandas as pd
            df_live = pd.DataFrame(_mh)
            if "epoch" in df_live.columns:
                df_live = df_live.set_index("epoch")
                _live_cols = [c for c in df_live.columns
                              if any(k in c.lower() for k in ["map50", "loss"])
                              and "95" not in c.lower()]
                if _live_cols:
                    st.markdown("**📊 学習進捗グラフ（リアルタイム）**")
                    st.line_chart(df_live[_live_cols])

        log_lines = st.session_state.training_log[-500:]
        log_text_escaped = "\n".join(log_lines).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
        components.html(f"""
<style>
  body{{margin:0;background:#0e1117;}}
  #log-box{{
    background:#0e1117;color:#e0e6ed;
    font-family:'JetBrains Mono',monospace;font-size:12px;
    height:380px;overflow-y:auto;
    padding:12px;border:1px solid #1e2330;border-radius:8px;
    white-space:pre-wrap;word-break:break-all;
  }}
</style>
<div id="log-box">{log_text_escaped}</div>
<script>
  var box = document.getElementById('log-box');
  var dist = 0;
  try {{ dist = parseInt(window.parent.localStorage.getItem('log_dist_bottom') || '0'); }} catch(e) {{}}
  setTimeout(function() {{
    if (dist > 100) {{
      box.scrollTop = box.scrollHeight - box.clientHeight - dist;
    }} else {{
      box.scrollTop = box.scrollHeight;
    }}
  }}, 0);
  var t = null;
  box.addEventListener('scroll', function() {{
    clearTimeout(t);
    t = setTimeout(function() {{
      var d = Math.max(0, box.scrollHeight - box.scrollTop - box.clientHeight);
      try {{ window.parent.localStorage.setItem('log_dist_bottom', d); }} catch(e) {{}}
    }}, 100);
  }}, {{passive:true}});
</script>
""", height=400)

        time.sleep(2)
        st.rerun()

    elif st.session_state.training_progress == 100:
        # ── 学習完了: プログレスバー＋ログ（expander / 静的表示） ──
        st.progress(1.0, text="進捗: 100% — 完了")
        if st.session_state.training_log:
            with st.expander("📋 学習ログ（完了）", expanded=False):
                st.text("\n".join(st.session_state.training_log[-500:]))

    elif st.session_state.training_progress > 0:
        # ── 途中停止: 止まった時点のプログレスバー＋ログ ──
        prog = st.session_state.training_progress
        st.progress(prog / 100, text=f"進捗: {prog}% — 停止")
        if st.session_state.training_log:
            with st.expander("📋 学習ログ（停止時点）", expanded=False):
                st.text("\n".join(st.session_state.training_log[-500:]))

    if st.session_state.training_error:
        st.error(f"学習エラー: {st.session_state.training_error}")

    # --- 完了後: モデル選択 ---
    if st.session_state.last_model_path:
        st.success(f"✅ 最新モデル: `{st.session_state.last_model_path}`")

    # results.csv の可視化
    if st.session_state.last_model_path:
        results_csv = Path(st.session_state.last_model_path).parent.parent / "results.csv"
        if results_csv.exists():
            import pandas as pd
            st.markdown("#### 📈 学習メトリクス")
            df_r = pd.read_csv(results_csv)
            df_r.columns = [c.strip() for c in df_r.columns]
            metric_cols = [c for c in df_r.columns
                           if any(k in c.lower() for k in ["map","precision","recall","loss"])]
            _last = df_r.iloc[-1]
            _map_col  = next((c for c in df_r.columns if "map50" in c.lower() and "95" not in c.lower()), None)
            _loss_col = next((c for c in df_r.columns if "val" in c.lower() and "loss" in c.lower()), None)
            _mc = st.columns(3)
            if _map_col:  _mc[0].metric("mAP50 (最終)", f"{_last[_map_col]:.4f}")
            if _loss_col: _mc[1].metric("Val Loss (最終)", f"{_last[_loss_col]:.4f}")
            _mc[2].metric("エポック数", len(df_r))
            if metric_cols:
                st.line_chart(df_r[metric_cols])
            with st.expander("📄 生データ（末尾5行）"):
                st.dataframe(df_r.tail(5), use_container_width=True)

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
    _mdl_count3 = len(list(MODELS_DIR.rglob("*.pt")))
    _prev_info3 = (f"← 前のステップ: ✅ 学習済みモデルが {_mdl_count3} 件あります"
                   if _mdl_count3 > 0 else "← 前のステップ: ⚠ Step2でモデルを先に学習してください")
    st.markdown(f"""
<div class="step-banner">
  <div class="sb-title">🔭 STEP 3: 推論・評価</div>
  <div class="sb-prev">{_prev_info3}</div>
  <div class="sb-desc">→ ここでやること: 推論実行 → FiftyOneで結果を可視化・確認</div>
</div>""", unsafe_allow_html=True)
    st.markdown('<div class="pipeline-card"><h3>🔭 推論 & FiftyOne 可視化</h3>', unsafe_allow_html=True)

    # --- モデル確認 ---
    current_model = st.session_state.last_model_path or ""
    model_display = current_model if current_model else "（未設定）"
    st.info(f"メインモデル: `{model_display}`")

    # 複数モデル比較
    _all_models = list(MODELS_DIR.rglob("*.pt"))
    _model_map = {str(p.relative_to(MODELS_DIR)): str(p) for p in _all_models}
    compare_mode = st.checkbox("🔀 複数モデル比較モード", value=False)
    if compare_mode and _model_map:
        selected_compare_models = st.multiselect(
            "比較するモデルを選択",
            list(_model_map.keys()),
            default=list(_model_map.keys())[:min(2, len(_model_map))],
        )
    else:
        selected_compare_models = []

    # --- 推論対象ソース ---
    _infer_src = st.radio(
        "推論対象ソース",
        ["📂 data/ のディレクトリ", "📤 画像をアップロード"],
        horizontal=True,
        key="infer_src_mode",
    )

    if _infer_src == "📤 画像をアップロード":
        import io as _io_infer
        _infer_files = st.file_uploader(
            "推論したい画像ファイル（複数選択可）",
            type=["jpg", "jpeg", "png", "bmp", "tiff"],
            accept_multiple_files=True,
            key="infer_upload_files",
        )
        if _infer_files:
            st.caption(f"✅ {len(_infer_files)} ファイル選択中")
            _tmp_infer = PREDICTIONS_DIR / "_tmp_uploads"
            _tmp_infer.mkdir(exist_ok=True)
            _cur_names = {f.name for f in _infer_files}
            _saved_names = {f.name for f in _tmp_infer.iterdir() if f.is_file()}
            if _cur_names != _saved_names:
                for _tf in list(_tmp_infer.iterdir()):
                    _tf.unlink()
                for _f in _infer_files:
                    (_tmp_infer / _f.name).write_bytes(_f.getbuffer())
            test_image_dir = str(_tmp_infer)
        else:
            test_image_dir = ""
    else:
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
        _infer_disabled = (
            (not current_model and not (compare_mode and selected_compare_models))
            or not test_image_dir
        )
        if st.button("▶ 推論実行", type="primary", use_container_width=True,
                    disabled=_infer_disabled):
            img_dir = Path(test_image_dir)
            if not img_dir.exists():
                st.error(f"画像ディレクトリが存在しません: {img_dir}")
            elif compare_mode and selected_compare_models:
                # 複数モデル比較推論
                compare_results = []
                for model_rel in selected_compare_models:
                    model_abs = _model_map[model_rel]
                    with st.spinner(f"推論中: {model_rel}…"):
                        saved = run_inference(
                            model_abs,
                            img_dir,
                            PREDICTIONS_DIR,
                            conf_threshold=inf_conf,
                        )
                    total_detections = 0
                    total_conf = 0.0
                    conf_count = 0
                    for jf in saved:
                        with open(jf) as f:
                            pred = json.load(f)
                        boxes = pred.get("boxes", [])
                        total_detections += len(boxes)
                        for b in boxes:
                            total_conf += b.get("confidence", 0.0)
                            conf_count += 1
                    avg_conf = total_conf / conf_count if conf_count > 0 else 0.0
                    compare_results.append({
                        "モデル": model_rel,
                        "検出数（合計）": total_detections,
                        "平均信頼度": round(avg_conf, 4),
                        "画像数": len(saved),
                    })
                if compare_results:
                    st.success("✅ 比較推論完了")
                    import pandas as pd
                    df_cmp = pd.DataFrame(compare_results)
                    st.dataframe(df_cmp, use_container_width=True, hide_index=True)
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

    # --- 推論結果 画像プレビュー ---
    _pred_jsons = sorted(PREDICTIONS_DIR.glob("*.json"))
    if _pred_jsons:
        st.markdown("#### 🖼 推論結果プレビュー")
        _preview_jsons = _pred_jsons[:9]
        for _row_start in range(0, len(_preview_jsons), 3):
            _row_files = _preview_jsons[_row_start:_row_start + 3]
            _row_cols = st.columns(3)
            for _col, _jf in zip(_row_cols, _row_files):
                _res = _draw_predictions(_jf)
                if _res:
                    _img, _n_boxes, _stem = _res
                    with _col:
                        st.image(_img, caption=f"{_stem} ({_n_boxes}件検出)",
                                 use_column_width=True)
        if len(_pred_jsons) > 9:
            st.caption(f"（他 {len(_pred_jsons) - 9} 件は省略。全件は下の一覧から確認）")

    # --- 推論結果 画像エクスポート ---
    if _pred_jsons:
        st.markdown("#### 📥 結果画像エクスポート")

        _exp_mode = st.radio(
            "エクスポート範囲",
            ["すべて書き出す", "選択して書き出す"],
            horizontal=True,
            key="exp_mode",
        )

        _exp_target_files: Optional[list[Path]] = None
        if _exp_mode == "選択して書き出す":
            _SEL_PAGE_SIZE = 12

            # 選択状態はウィジェットキーではなく set で管理（ページ切替後も保持）
            if "exp_sel_set" not in st.session_state:
                st.session_state.exp_sel_set = set()
            if "exp_sel_page" not in st.session_state:
                st.session_state.exp_sel_page = 0

            _total_pages = max(1, (len(_pred_jsons) + _SEL_PAGE_SIZE - 1) // _SEL_PAGE_SIZE)
            _cur_page    = min(st.session_state.exp_sel_page, _total_pages - 1)
            _page_jsons  = _pred_jsons[_cur_page * _SEL_PAGE_SIZE : (_cur_page + 1) * _SEL_PAGE_SIZE]
            _sel_count   = len(st.session_state.exp_sel_set)

            # ── ツールバー ──
            _tb1, _tb2, _tb3, _tb4 = st.columns([2, 2, 2, 4])
            with _tb1:
                if st.button("☑ 全件選択", key="exp_sel_all", use_container_width=True):
                    st.session_state.exp_sel_set = {jf.name for jf in _pred_jsons}
                    st.rerun()
            with _tb2:
                if st.button("☑ このページ", key="exp_sel_page_btn", use_container_width=True):
                    st.session_state.exp_sel_set.update(jf.name for jf in _page_jsons)
                    st.rerun()
            with _tb3:
                if st.button("☐ すべて解除", key="exp_desel_all", use_container_width=True):
                    st.session_state.exp_sel_set = set()
                    st.rerun()
            with _tb4:
                st.markdown(
                    f'<div style="padding-top:8px; color:#7ecff4; font-size:.85rem;">'
                    f'選択中: <b>{_sel_count}</b> / {len(_pred_jsons)} 件 &nbsp;|&nbsp; '
                    f'ページ {_cur_page + 1} / {_total_pages}</div>',
                    unsafe_allow_html=True,
                )

            # ── 画像グリッド + チェックボックス ──
            # value= で exp_sel_set から初期状態を復元し、変更を exp_sel_set に同期
            _GRID_COLS = 3
            for _row_start in range(0, len(_page_jsons), _GRID_COLS):
                _row_files = _page_jsons[_row_start : _row_start + _GRID_COLS]
                _row_cols  = st.columns(_GRID_COLS)
                for _col, _jf in zip(_row_cols, _row_files):
                    with _col:
                        _res = _draw_predictions(_jf)
                        if _res:
                            _img, _n_boxes, _stem = _res
                            st.image(_img, caption=f"{_stem} ({_n_boxes}件)", use_column_width=True)
                        else:
                            st.caption(_jf.stem)
                            st.markdown("_(プレビュー不可)_")
                        _chk_result = st.checkbox(
                            "選択",
                            value=(_jf.name in st.session_state.exp_sel_set),
                            key=f"exp_chk_{_cur_page}_{_jf.name}",
                        )
                        if _chk_result:
                            st.session_state.exp_sel_set.add(_jf.name)
                        else:
                            st.session_state.exp_sel_set.discard(_jf.name)

            # ── ページネーション ──
            _pn1, _pn2, _pn3 = st.columns([1, 2, 1])
            with _pn1:
                if st.button("← 前へ", disabled=(_cur_page == 0),
                             key="exp_pg_prev", use_container_width=True):
                    st.session_state.exp_sel_page = _cur_page - 1
                    st.rerun()
            with _pn2:
                st.markdown(
                    f'<div style="text-align:center; padding-top:8px; color:#4a6080; font-size:.82rem;">'
                    f'{_cur_page + 1} / {_total_pages}</div>',
                    unsafe_allow_html=True,
                )
            with _pn3:
                if st.button("次へ →", disabled=(_cur_page == _total_pages - 1),
                             key="exp_pg_next", use_container_width=True):
                    st.session_state.exp_sel_page = _cur_page + 1
                    st.rerun()

            _exp_target_files = [PREDICTIONS_DIR / n for n in st.session_state.exp_sel_set
                                 if (PREDICTIONS_DIR / n).exists()]
            _exp_count = len(_exp_target_files)
        else:
            # モード切替時に選択をリセット
            if "exp_sel_set" in st.session_state:
                st.session_state.exp_sel_set = set()
            _exp_count = len(_pred_jsons)

        # ── フォーマット・品質 ──────────────────────────────────────────────────
        _exp_c1, _exp_c2 = st.columns(2)
        with _exp_c1:
            _exp_fmt = st.selectbox("フォーマット", ["PNG", "JPEG"], key="exp_fmt")
        with _exp_c2:
            _exp_q = st.slider("品質 (JPEGのみ)", 60, 100, 95, step=5,
                               disabled=(_exp_fmt != "JPEG"), key="exp_quality")

        # ── 保存先フォルダ選択（ブラウザUI） ──────────────────────────────────
        st.markdown("**📁 保存先フォルダ**")
        if "exp_dest_dir" not in st.session_state:
            st.session_state.exp_dest_dir = str(PREDICTIONS_DIR / "exports")

        _exp_dest = Path(st.session_state.exp_dest_dir)
        _BROWSE_ROOT = Path("/workspace")

        # 現在パス表示 + 「↑ 上へ」ボタン
        _nav1, _nav2 = st.columns([5, 1])
        with _nav1:
            st.code(str(_exp_dest), language="text")
        with _nav2:
            _can_up = (
                _exp_dest != _BROWSE_ROOT
                and str(_exp_dest).startswith(str(_BROWSE_ROOT))
            )
            if st.button("↑ 上へ", key="exp_nav_up", use_container_width=True,
                         disabled=not _can_up):
                st.session_state.exp_dest_dir = str(_exp_dest.parent)
                st.rerun()

        # サブフォルダ一覧（クリックで移動）
        try:
            _subdirs = sorted(
                [d for d in _exp_dest.iterdir() if d.is_dir()]
            ) if _exp_dest.exists() else []
        except Exception:
            _subdirs = []

        if _subdirs:
            _COLS = 4
            _sd_cols = st.columns(_COLS)
            for _ci, _sd in enumerate(_subdirs[:12]):
                with _sd_cols[_ci % _COLS]:
                    if st.button(f"📁 {_sd.name}", key=f"exp_sd_{_ci}",
                                 use_container_width=True):
                        st.session_state.exp_dest_dir = str(_sd)
                        st.rerun()
        else:
            st.caption("サブフォルダなし")

        # 新規フォルダ作成
        _nf1, _nf2 = st.columns([4, 1])
        with _nf1:
            _exp_new_folder = st.text_input(
                "新しいフォルダ名", key="exp_new_folder",
                placeholder="フォルダ名を入力して ＋ 作成",
                label_visibility="collapsed",
            )
        with _nf2:
            if st.button("＋ 作成", key="exp_mkdir", use_container_width=True):
                if _exp_new_folder.strip():
                    _nd = _exp_dest / _exp_new_folder.strip()
                    _nd.mkdir(parents=True, exist_ok=True)
                    st.session_state.exp_dest_dir = str(_nd)
                    st.rerun()

        # ── 書き出しボタン ──────────────────────────────────────────────────────
        _btn_disabled = (_exp_mode == "選択して書き出す" and _exp_count == 0)
        if st.button(f"📥 {_exp_count} 件を書き出す", use_container_width=True,
                     type="primary", disabled=_btn_disabled):
            _exp_out = _exp_dest
            _prog_bar  = st.progress(0, text="書き出し準備中…")
            _prog_text = st.empty()

            def _on_progress(cur, total, fname):
                _prog_bar.progress(cur / total,
                                   text=f"{cur} / {total} 件処理中")
                _prog_text.caption(f"→ {fname}")

            _ok, _ng = export_prediction_images(
                _exp_out, _exp_fmt, _exp_q, _exp_target_files, _on_progress
            )
            _prog_bar.empty()
            _prog_text.empty()
            if _ok > 0:
                st.success(f"✅ {_ok} 件を保存しました → `{_exp_out}`")
                # ZIPにまとめてブラウザからダウンロード
                import io as _io_exp
                _exp_glob = sorted(_exp_out.glob("*.*"))
                if _exp_glob:
                    _zip_buf = _io_exp.BytesIO()
                    with zipfile.ZipFile(_zip_buf, "w", zipfile.ZIP_DEFLATED) as _zf:
                        for _ef in _exp_glob:
                            _zf.write(_ef, _ef.name)
                    _zip_buf.seek(0)
                    st.download_button(
                        f"⬇️ ZIPでダウンロード ({_ok}件)",
                        _zip_buf.getvalue(),
                        file_name=f"exports_{datetime.now():%Y%m%d_%H%M}.zip",
                        mime="application/zip",
                        use_container_width=True,
                    )
            if _ng > 0:
                st.warning(f"⚠ {_ng} 件スキップ（元画像が見つからないため）")

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
            with st.expander(f"➕ {ds.name} に画像を追加"):
                _add_imgs = st.file_uploader(
                    "追加する画像ファイル（複数選択可）",
                    type=["jpg", "jpeg", "png", "bmp", "tiff"],
                    accept_multiple_files=True,
                    key=f"add_imgs_{ds.name}",
                )
                _add_split = st.radio(
                    "追加先スプリット", ["train", "val"], horizontal=True,
                    key=f"add_split_{ds.name}",
                )
                if _add_imgs:
                    st.caption(f"選択中: {len(_add_imgs)} ファイル")
                    _add_dst = ds / "images" / _add_split
                    if st.button(
                        f"📤 images/{_add_split}/ に追加",
                        key=f"add_btn_{ds.name}",
                        type="primary",
                        use_container_width=True,
                    ):
                        _add_dst.mkdir(parents=True, exist_ok=True)
                        for _f in _add_imgs:
                            (_add_dst / _f.name).write_bytes(_f.getbuffer())
                        st.success(f"✅ {len(_add_imgs)} ファイルを追加しました → `{_add_dst}`")
                        st.rerun()

    st.markdown("---")

    # --- models/ モデル一覧（カード表示） ---
    st.markdown("#### 🤖 学習済みモデル (`models/`)")
    model_files = sorted(
        MODELS_DIR.rglob("*.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    ) if MODELS_DIR.exists() else []
    if not model_files:
        st.info("models/ に .pt ファイルがありません。")
    else:
        import pandas as pd
        for mp in model_files:
            size_mb  = mp.stat().st_size / (1024 * 1024)
            mod_time = datetime.fromtimestamp(mp.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            is_current = (str(mp) == st.session_state.last_model_path)

            # results.csv からmAP50を取得
            _results_csv = mp.parent.parent / "results.csv"
            _map50_val = None
            if _results_csv.exists():
                try:
                    _df_r = pd.read_csv(_results_csv)
                    _df_r.columns = [c.strip() for c in _df_r.columns]
                    _mc = next((c for c in _df_r.columns
                                if "map50" in c.lower() and "95" not in c.lower()), None)
                    if _mc:
                        _map50_val = float(_df_r.iloc[-1][_mc])
                except Exception:
                    pass

            with st.container(border=True):
                _label_col, _size_col, _map_col, _use_col, _del_col = st.columns([4, 2, 2, 2, 1])
                with _label_col:
                    if is_current:
                        st.markdown("⭐ **現在使用中**")
                    st.markdown(f"`{mp.relative_to(MODELS_DIR)}`")
                    st.caption(mod_time)
                with _size_col:
                    st.metric("サイズ", f"{size_mb:.1f} MB")
                with _map_col:
                    if _map50_val is not None:
                        st.metric("mAP50", f"{_map50_val:.4f}")
                    else:
                        st.caption("mAP50: -")
                with _use_col:
                    if st.button("✅ 使用", key=f"use_model_{mp}", use_container_width=True,
                                 type="primary" if not is_current else "secondary"):
                        st.session_state.last_model_path = str(mp)
                        st.rerun()
                with _del_col:
                    if st.button("🗑", key=f"del_model_{mp}", help=f"{mp.name} を削除"):
                        mp.unlink()
                        if st.session_state.last_model_path == str(mp):
                            st.session_state.last_model_path = None
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

    st.markdown("---")
    st.markdown("#### 🔀 データセット統合")
    _ds_dirs = [d for d in sorted(DATA_DIR.iterdir()) if d.is_dir()] if DATA_DIR.exists() else []
    _ds_names = [d.name for d in _ds_dirs]
    if len(_ds_names) < 2:
        st.info("統合するには2つ以上のデータセットが必要です。")
    else:
        merge_targets = st.multiselect("統合するデータセットを選択（2つ以上）", _ds_names)
        merge_out_name = st.text_input("統合先ディレクトリ名", value=f"merged_{datetime.now():%Y%m%d_%H%M}")
        if st.button("🔀 統合実行", disabled=len(merge_targets) < 2):
            import yaml as pyyaml
            out_dir = DATA_DIR / merge_out_name
            all_labels: list[str] = []
            # 各データセットからラベル収集
            for ds_name in merge_targets:
                src = DATA_DIR / ds_name
                yaml_f = src / "data.yaml"
                if yaml_f.exists():
                    with open(yaml_f) as f:
                        ydata = pyyaml.safe_load(f)
                    for lbl in ydata.get("names", []):
                        if lbl not in all_labels:
                            all_labels.append(lbl)
            # 画像・ラベルをコピー
            for split in ("train", "val"):
                for ds_name in merge_targets:
                    src = DATA_DIR / ds_name
                    for kind in ("images", "labels"):
                        src_dir = src / split / kind
                        if not src_dir.exists():
                            continue
                        dst_dir = out_dir / split / kind
                        dst_dir.mkdir(parents=True, exist_ok=True)
                        for f in src_dir.iterdir():
                            dst = dst_dir / f"{ds_name}_{f.name}"
                            shutil.copy2(f, dst)
            # data.yaml 生成
            data_yaml_content = {
                "path": str(out_dir),
                "train": "train/images",
                "val": "val/images",
                "names": all_labels,
                "nc": len(all_labels),
            }
            with open(out_dir / "data.yaml", "w") as f:
                pyyaml.dump(data_yaml_content, f, allow_unicode=True)
            st.success(f"✅ 統合完了: `{out_dir}` (ラベル: {all_labels})")
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# ===========================================================================
# タブ5: トピックス
# ===========================================================================
with tab5:
    st.markdown('<div class="pipeline-card"><h3>📚 ガイド</h3>', unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#6a8aaa;font-size:.85rem;'>物体検出 MLOps の概念・操作ガイドです。GitHub の詳細ドキュメントを参照してください。</p>",
        unsafe_allow_html=True,
    )

    # ── ガイドへのリンク ──────────────────────────────────────────
    st.markdown("""
<div class="step-banner" style="margin-bottom:20px;">
  <div class="sb-title">📖 物体検出 MLOps 学習ガイド</div>
  <div class="sb-desc">アノテーションのコツ・学習パラメータの意味・データ拡張・モデルサイズの選び方などを解説しています。</div>
</div>
""", unsafe_allow_html=True)

    st.markdown(
        "**[→ docs/guide.md をGitHubで開く](https://github.com/ryotaema/detection_dev_ui/blob/main/docs/guide.md)**",
    )
    st.caption("アノテーションのコツ / mAP・IoU・過学習の解説 / 学習パラメータ / データ拡張 / モデルサイズ選択基準 を掲載しています。")

    st.markdown("---")

    # ── 公式ドキュメント ──────────────────────────────────────────
    st.markdown("#### 公式ドキュメント")
    link_col1, link_col2 = st.columns(2)
    with link_col1:
        st.markdown("""
<div class="link-card">
  <div class="lc-title">📝 <a href="https://docs.cvat.ai/" target="_blank">CVAT 公式ドキュメント</a></div>
  <div class="lc-desc">アノテーション操作・プロジェクト管理・エクスポート形式の詳細</div>
</div>
<div class="link-card">
  <div class="lc-title">🚀 <a href="https://docs.ultralytics.com/" target="_blank">Ultralytics YOLO 公式ドキュメント</a></div>
  <div class="lc-desc">モデルの使い方・各学習パラメータの意味・モデルサイズ一覧</div>
</div>
<div class="link-card">
  <div class="lc-title">📊 <a href="https://mlflow.org/docs/latest/index.html" target="_blank">MLflow 公式ドキュメント</a></div>
  <div class="lc-desc">実験管理・モデルレジストリ・比較ビューの使い方</div>
</div>
""", unsafe_allow_html=True)
    with link_col2:
        st.markdown("""
<div class="link-card">
  <div class="lc-title">🔭 <a href="https://docs.voxel51.com/" target="_blank">FiftyOne 公式ドキュメント</a></div>
  <div class="lc-desc">データセット探索・推論結果可視化・フィルタリングの使い方</div>
</div>
<div class="link-card">
  <div class="lc-title">📺 <a href="https://docs.streamlit.io/" target="_blank">Streamlit 公式ドキュメント</a></div>
  <div class="lc-desc">このUIで使用しているフレームワーク。ウィジェット・レイアウトの仕様</div>
</div>
""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# フッター
# ---------------------------------------------------------------------------
st.markdown("""
<div style="border-top:1px solid #1e2330; margin-top:40px; padding-top:12px;
            text-align:center; color:#2a3a50; font-size:.75rem; font-family:'JetBrains Mono',monospace;">
    detection_dev_ui v1.0 · CVAT · YOLO · MLflow · FiftyOne · Streamlit
</div>
""", unsafe_allow_html=True)