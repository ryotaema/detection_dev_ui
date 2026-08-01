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

# ロジック層。画面を持たない処理はすべて core/ 側にある
from core import *  # noqa: F401,F403
from core import (  # アンダースコア始まりは * で入らないので明示的に取り込む
    _box_iou, _collect_prediction_items, _deploy_worker, _DOC_AUG, _DOC_TRAIN,
    _draw_predictions, _eval_worker, _find_image_dirs, _get_deploy_shared,
    _get_eval_shared, _get_train_shared, _iou, _MODEL_OPTS, _nuctl,
    _StdoutCapture, _train_worker, _yolo_txt_to_xyxy,
)

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
    "cvat_jobs": [],       # ジョブ単位の進捗（タスクの status より細かい）
    "cvat_export_tasks": [],   # 直近にエクスポートしたタスク（来歴に残す）
    "cvat_xml_info": None,
    "cvat_raw_dir": None,
    "theme_name": "ライト シンプル",
    "reanno_set": set(),   # 再アノテーション要フラグを立てた JSON ファイル名の集合
}
# データもモデルも無い＝初回起動とみなし、はじめかたガイドを開いた状態にする
if "show_onboarding" not in st.session_state:
    try:
        _first_run = (not any(DATA_DIR.rglob("data.yaml"))
                      and not any(MODELS_DIR.rglob("*.pt")))
    except Exception:
        _first_run = False
    defaults["show_onboarding"] = _first_run

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
# エラー表示（原因の推定と対処をセットで出す）
# ---------------------------------------------------------------------------
def show_error(message: str, prefix: str = "") -> None:
    """エラーを表示し、よくある原因に当てはまれば対処も添える"""
    st.error(f"{prefix}{message}" if prefix else str(message))
    hint = explain_error(str(message))
    if hint:
        st.warning(f"**{hint['title']}**\n\n{hint['hint']}")


# ---------------------------------------------------------------------------
# パイプライン状態ヘルパー
# ---------------------------------------------------------------------------
def _get_pipeline_status() -> dict:
    yaml_exists  = len(list(DATA_DIR.rglob("data.yaml"))) > 0
    model_exists = len(list(MODELS_DIR.rglob("*.pt"))) > 0
    pred_exists  = len(list(PREDICTIONS_DIR.glob("*.json"))) > 0
    training_now = _train_state.get("running", False)

    # STEP1(アノテーション) は取得済みのジョブ進捗から判定する。
    # 未取得のときは判定材料が無いので complete 扱いのままにする。
    jobs = st.session_state.get("cvat_jobs") or []
    if jobs:
        total_f = sum(j.get("frames", 0) for j in jobs)
        done_f  = sum(j.get("frames", 0) for j in jobs if j.get("state") == "completed")
        if total_f and done_f >= total_f:
            step1 = "complete"
        elif done_f > 0:
            step1 = "active"
        else:
            step1 = "pending"
    else:
        step1 = "complete"

    return {
        "step1": step1,
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

    # ── はじめかた / 環境セットアップ ────────────────────────────────────────
    st.markdown("---")
    st.checkbox(
        "📖 はじめかたガイドを表示",
        key="show_onboarding",
        help="初めて使うとき、別の PC に環境を移すときの手順を画面上部に表示します。"
             "いつでもここで切り替えられます。",
    )
    if not st.session_state.get("show_onboarding"):
        st.caption("困ったときはここから表示できます")

# ---------------------------------------------------------------------------
# はじめかたガイド（サイドバーのチェックで表示を切り替える）
# ---------------------------------------------------------------------------
if st.session_state.get("show_onboarding"):
    with st.container(border=True):
        st.markdown("### 📖 はじめかた")

        _ob_t1, _ob_t2, _ob_t3 = st.tabs([
            "① まず何をするか", "② 別のPCで環境を作る", "③ 用語とタスクの選び方",
        ])

        # ── ① 現在の状態と次にやること ──────────────────────────────────
        with _ob_t1:
            _ob_ds     = len(list(DATA_DIR.rglob("data.yaml"))) if DATA_DIR.exists() else 0
            _ob_raw    = len([d for d in DATA_DIR.iterdir() if d.is_dir()]) if DATA_DIR.exists() else 0
            _ob_models = len(list(MODELS_DIR.rglob("*.pt"))) if MODELS_DIR.exists() else 0
            _ob_preds  = len(list(PREDICTIONS_DIR.glob("*.json"))) if PREDICTIONS_DIR.exists() else 0
            # CVAT の疎通はサイドバーに出しているので、ここでは
            # 「取り込むものが既にあるか」で判定する（表示のたびに通信しない）
            _ob_cvat   = bool(st.session_state.get("cvat_tasks")) or _ob_raw > 0

            st.markdown(
                "このツールは **アノテーション → データ取込 → 学習 → 評価** の4ステップで"
                "画像認識モデルを作ります。上のタブがそのまま順番になっています。"
            )

            _ob_steps = [
                ("① CVAT でアノテーション",
                 _ob_cvat,
                 f"CVAT ({CVAT_WEB}) で画像に印を付けます",
                 "🏷 Step1 タブで進捗を確認できます"),
                ("② データセットを作る",
                 _ob_ds > 0,
                 f"CVAT から取り込んで YOLO 形式に変換します（現在 {_ob_ds} 件）",
                 "📤 Step2 タブ。CVAT を使わず ZIP や画像を直接入れることもできます"),
                ("③ 学習する",
                 _ob_models > 0,
                 f"モデルサイズとパラメータを選んで学習します（現在 {_ob_models} 件）",
                 "🚀 Step3 タブ。他の PC で作った .pt を持ち込むこともできます"),
                ("④ 評価・推論する",
                 _ob_preds > 0,
                 f"精度を測り、推論結果を確認します（推論結果 {_ob_preds} 件）",
                 "🔭 Step4 タブ。mAP 比較や正解ラベルとの差分も見られます"),
            ]
            for _title, _done, _desc, _where in _ob_steps:
                _icon = "✅" if _done else "⬜"
                st.markdown(f"**{_icon} {_title}** — {_desc}")
                st.caption(f"　　{_where}")

            st.markdown("---")
            if _ob_ds == 0 and _ob_raw == 0:
                st.info(
                    "**まだデータがありません。** まず CVAT でアノテーションするか、"
                    "「📤 Step2: データ取込」の「📁 ローカルからデータを直接追加」から"
                    "手元の画像や YOLO 形式の ZIP を入れてください。"
                )
            elif _ob_ds == 0:
                st.info(
                    "**データはありますが data.yaml がまだありません。** "
                    "「📤 Step2: データ取込」でデータセットを生成してください。"
                )
            elif _ob_models == 0:
                st.info(
                    "**データセットができています。** 次は「🚀 Step3: モデル学習」で学習します。"
                    "まずは小さいモデル（yolo11n / yolo11s）と少なめのエポックで"
                    "一周させてみるのがおすすめです。"
                )
            else:
                st.success(
                    "**ひととおり揃っています。** 「🔭 Step4: 推論・評価」でモデルの精度を測り、"
                    "「🏷 Step1」でそのモデルを CVAT の自動アノテーションに載せると、"
                    "次のアノテーションが楽になります。"
                )

        # ── ② 別PCへの移行手順 ────────────────────────────────────────
        with _ob_t2:
            st.markdown(
                "**別の PC で同じ環境を作るとき**の手順です。"
                "詳細は README.md に記載しています。"
            )
            st.markdown(
                "**1. 前提を揃える**\n"
                "- Docker / docker compose\n"
                "- GPU を使う場合は nvidia-container-toolkit\n"
            )
            st.code(
                "git clone <このリポジトリ>\n"
                "cd detection_dev_ui", language="bash")
            st.markdown("**2. `.env` を作る**（リポジトリには含まれません）")
            st.code(
                "CVAT_USERNAME=admin\n"
                "CVAT_PASSWORD=<任意のパスワード>\n"
                "CVAT_DB_PASSWORD=<任意>\n"
                "CVAT_IAM_DB_PASSWORD=<任意>\n"
                "NVIDIA_VISIBLE_DEVICES=all\n"
                "COMPOSE_PROJECT_NAME=mlops_workspace", language="bash")
            st.markdown("**3. データベースを初期化する**（初回のみ）")
            st.code(
                "docker compose up -d cvat_db cvat_redis cvat_redis_inmem "
                "cvat_redis_ondisk cvat_iam_db\n"
                "sleep 30\n"
                "docker compose run --rm cvat_server init\n"
                "docker compose run --rm cvat_server bash -c \\\n"
                '  "~/manage.py createsuperuser --username admin --email admin@local.com"',
                language="bash")
            st.markdown("**4. 全サービスを起動する**")
            st.code("docker compose up -d\ndocker compose ps", language="bash")

            st.markdown("---")
            st.markdown("**作業内容を持っていくには**")
            st.markdown(
                "- **データセット** … 「📁 データ管理」の `⬇ 持ち出す` から ZIP で書き出せます"
                "（画像を含めない「ラベルのみ」も選べます）\n"
                "- **モデル** … 同じくデータ管理タブから `.pt` 単体、または"
                "学習ログ・評価結果込みの「一式ZIP」で書き出せます\n"
                "- **持ち込み** … 移行先では「📤 学習済みモデルをアップロード」と"
                "「📁 ローカルからデータを直接追加」から取り込めます\n"
                "- **CVAT のアノテーション** … CVAT 側でタスクをバックアップするか、"
                "「🏷 Step1」でラベル定義を書き出して共有できます"
            )
            st.info(
                "`data/` `models/` `predictions/` はホスト側のディレクトリを"
                "そのままマウントしています。ディレクトリごとコピーしても移行できます。"
            )

        # ── ③ 用語とタスク種別 ────────────────────────────────────────
        with _ob_t3:
            st.markdown("**どのタスク種別を選べばよいか**")
            st.markdown(
                "| やりたいこと | タスク種別 | CVAT で付けるもの |\n"
                "|---|---|---|\n"
                "| 物体の位置を四角で囲みたい | `detect` | 矩形 (box) |\n"
                "| 物体の形を正確に取りたい | `segment` | ポリゴン |\n"
                "| 傾いた物体を囲みたい | `obb` | 回転付き矩形 / 4点ポリゴン |\n"
                "| 画像全体を仕分けたい | `classify` | タグ |\n"
                "| 関節や特徴点を取りたい | `pose` | ポイント |\n"
            )
            st.caption("まず迷ったら `detect` から始めるのが無難です。"
                       "後から別の種別のデータセットを作り直すこともできます。")

            st.markdown("---")
            st.markdown("**最低限おさえる指標**")
            st.markdown(
                "- **Precision（適合率）** … 検出したもののうち、正しかった割合。"
                "低い＝誤検出が多い\n"
                "- **Recall（再現率）** … 実際にあるもののうち、見つけられた割合。"
                "低い＝見逃しが多い\n"
                "- **mAP50** … ざっくりした位置が合っていれば正解とする精度。まずこれを見ます\n"
                "- **mAP50-95** … 位置の正確さまで厳しく見る精度。実用ではこちらが効きます\n"
                "- **top1 accuracy** … 画像分類での正答率\n"
            )
            st.caption("詳しい解説と失敗したときの対処は「📚 トピックス」タブにあります。")

        st.caption("このガイドはサイドバー最下部の「📖 はじめかたガイドを表示」で"
                   "いつでも開閉できます。")

# ---------------------------------------------------------------------------
# タブ構成
# ---------------------------------------------------------------------------
tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏷 Step1: アノテーション",
    "📤 Step2: データ取込",
    "🚀 Step3: モデル学習",
    "🔭 Step4: 推論・評価",
    "📁 データ管理",
    "📚 トピックス",
])

# ===========================================================================
# タブ0: アノテーション（CVAT 自動アノテーション運用 + 進捗）
# ===========================================================================
with tab0:
    st.markdown(f"""
<div class="step-banner">
  <div class="sb-title">🏷 STEP 1: アノテーション</div>
  <div class="sb-prev">← 作業場所: CVAT ({CVAT_WEB}) — ここはその作業を支援する管理画面です</div>
  <div class="sb-desc">→ ここでやること: 学習済みモデルを自動アノテーションに載せる / チームの進捗を把握する</div>
</div>""", unsafe_allow_html=True)

    # ── 自動アノテーションモデル (Nuclio) ──────────────────────────────────
    st.markdown('<div class="pipeline-card"><h3>🤖 自動アノテーションモデル (Nuclio)</h3>',
                unsafe_allow_html=True)
    st.caption(
        "ここにデプロイしたモデルが、CVAT の「Actions → Automatic annotation」の選択肢に現れます。"
        "手作業のアノテーションを、モデルの推論結果を修正する作業に置き換えられます。"
    )

    _sl = serverless_status()
    if not serverless_ready():
        _sl_miss = []
        if not _sl["deploy_sh"]:   _sl_miss.append("`serverless/deploy.sh` が見つかりません")
        if not _sl["nuctl"]:       _sl_miss.append("`serverless/bin/nuctl` が見つかりません")
        if not _sl["docker_sock"]: _sl_miss.append("`/var/run/docker.sock` がマウントされていません")
        if not _sl["docker_cli"]:  _sl_miss.append("コンテナ内に `docker` コマンドがありません")
        st.warning(
            "⚠ このコンテナからは Nuclio へデプロイできません。\n\n"
            + "\n".join(f"- {m}" for m in _sl_miss)
            + "\n\n`docker-compose.yml` の `streamlit_app` に serverless / docker.sock の"
              "マウントを追加し、`docker compose up -d streamlit_app` を実行してください。"
        )
    else:
        _dep_state, _dep_lock = _get_deploy_shared()
        with _dep_lock:
            _dep_running  = _dep_state["running"]
            _dep_log      = list(_dep_state["log"])
            _dep_error    = _dep_state["error"]
            _dep_target   = _dep_state["target"]
            _dep_finished = _dep_state["finished"]

        # --- デプロイ実行中 / 直後のログ表示 ---
        if _dep_running or _dep_finished:
            if _dep_running:
                st.info(f"⏳ デプロイ実行中: `{_dep_target}`  "
                        "（初回はイメージのビルドに数分〜十数分かかります）")
            elif _dep_error:
                show_error(_dep_error, prefix="❌ デプロイ失敗: ")
            else:
                st.success(f"✅ デプロイ完了: `{_dep_target}`")

            st.code("\n".join(_dep_log[-40:]) or "(出力待ち)", language="bash")

            if not _dep_running:
                if st.button("ログを閉じる", key="dep_clear_log"):
                    with _dep_lock:
                        _dep_state["finished"] = False
                        _dep_state["log"] = []
                        _dep_state["error"] = None
                    st.rerun()

        # --- デプロイ済み関数の一覧 ---
        _fns  = cached_nuclio_functions()
        _defs = list_serverless_defs()
        _def_by_fn = {d["function_name"]: d for d in _defs if d["function_name"]}

        st.markdown(f"**デプロイ済み: {len(_fns)} 件**")
        if not _fns:
            st.info("まだ関数がデプロイされていません。下の「新しいモデルをデプロイ」から追加してください。")
        else:
            for _fn in _fns:
                _d = _def_by_fn.get(_fn["name"], {})
                with st.container(border=True):
                    _fc1, _fc2, _fc3, _fc4 = st.columns([5, 2, 2, 2])
                    with _fc1:
                        _badge = {"ready": "🟢", "error": "🔴", "building": "🟡"}.get(
                            _fn["state"], "⚪")
                        st.markdown(f"{_badge} **{_fn['display']}**")
                        st.caption(f"`{_fn['name']}`")
                        st.caption("🏷 ラベル: " + (", ".join(_fn["labels"]) or "—"))
                        if _d.get("model_run"):
                            _mr_ok = "✅" if _d.get("model_exists") else "⚠ 見つかりません"
                            st.caption(f"📦 モデル: `models/{_d['model_run']}` {_mr_ok}")
                    with _fc2:
                        st.metric("状態", _fn["state"] or "—")
                    with _fc3:
                        st.metric("実行", "GPU" if _fn["gpu"] else "CPU")
                    with _fc4:
                        if _d.get("dir"):
                            if st.button("🔄 再デプロイ", key=f"redeploy_{_fn['name']}",
                                         use_container_width=True, disabled=_dep_running,
                                         help="モデルを差し替えた後に実行すると最新の best.pt が反映されます"):
                                start_deploy(_d["dir"], use_gpu=_fn["gpu"])
                                cached_nuclio_functions.clear()
                                st.rerun()
                        if st.button("🗑 削除", key=f"delfn_{_fn['name']}",
                                     use_container_width=True, disabled=_dep_running):
                            _ok, _msg = delete_nuclio_function(_fn["name"])
                            cached_nuclio_functions.clear()
                            if _ok:
                                st.success(f"削除しました: {_fn['name']}")
                            else:
                                st.error(f"削除に失敗: {_msg}")
                            st.rerun()

        # --- 新規デプロイ ---
        st.markdown("---")
        st.markdown("**➕ 新しいモデルをデプロイ**")

        _dep_models = sorted(MODELS_DIR.rglob("*.pt"), key=lambda p: p.stat().st_mtime,
                             reverse=True) if MODELS_DIR.exists() else []
        if not _dep_models:
            st.info("models/ に .pt がありません。Step3で学習するか、データ管理タブから取り込んでください。")
        else:
            _dep_map = {str(p.relative_to(MODELS_DIR)): p for p in _dep_models}
            _dep_sel = st.selectbox("デプロイするモデル", list(_dep_map.keys()), key="dep_model_sel")
            _dep_path = _dep_map[_dep_sel]
            # models/<run>/weights/best.pt の <run> を取り出す（deploy.sh がこの構造を前提とする）
            _dep_run = _dep_path.parent.parent.name if _dep_path.parent.name == "weights" else ""

            if not _dep_run or _dep_path.name != "best.pt":
                st.warning(
                    "⚠ デプロイできるのは `models/<モデル名>/weights/best.pt` の形式のみです"
                    "（`serverless/deploy.sh` がこのパスから重みを読み込みます）。\n\n"
                    f"選択中: `{_dep_sel}`"
                )
            else:
                # クラス名はモデルから取得する（= CVAT に出るラベル定義になる）
                _dep_meta = read_model_meta(_dep_path)
                if _dep_meta is None:
                    with st.spinner("モデルのクラス名を読み込み中…"):
                        _dep_meta = inspect_model_file(_dep_path)
                _dep_classes = _dep_meta.get("names") or []

                if not _dep_meta.get("ok") or not _dep_classes:
                    st.error(
                        "❌ このモデルからクラス名を取得できませんでした。"
                        "ラベル定義を作れないためデプロイできません。\n\n"
                        f"{_dep_meta.get('error') or 'クラス名が空です'}"
                    )
                else:
                    _dep_task = _dep_meta.get("task") or "detect"
                    _dep_shape = "polygon（ポリゴン）" if _dep_task == "segment" \
                        else "rectangle（矩形）"
                    st.success(f"🏷 ラベル定義（モデルのクラス名から自動生成）: "
                               f"**{', '.join(_dep_classes)}**")
                    st.caption(
                        f"タスク種別: `{_dep_task}` → CVAT には **{_dep_shape}** として返します。"
                    )
                    st.caption(
                        "⚠ CVAT タスク側のラベル名がこれと一致していないと、"
                        "自動アノテーションの結果が反映されません。"
                    )

                    _dc1, _dc2 = st.columns(2)
                    with _dc1:
                        _dep_dir = st.text_input(
                            "関数ディレクトリ名 (`serverless/custom/` 以下)",
                            value=slugify_function_name(_dep_run),
                            key="dep_fn_dir",
                        ).strip()
                    with _dc2:
                        _dep_disp = st.text_input(
                            "CVAT に表示する名前",
                            value=f"{_dep_run} (custom)",
                            key="dep_fn_disp",
                        ).strip()

                    _dep_gpu = st.radio(
                        "実行モード", ["GPU", "CPU"], horizontal=True, key="dep_gpu_mode",
                        help="GPU 実行には Docker daemon の default-runtime=nvidia が必要です"
                             "（serverless/README.md 参照）。CPU なら前提なしで動きます。",
                    ) == "GPU"

                    _dep_slug = slugify_function_name(_dep_dir) if _dep_dir else ""
                    if _dep_slug:
                        _exists_def = (SERVERLESS_DIR / "custom" / _dep_dir).exists()
                        st.caption(f"Nuclio 関数名: `custom-{_dep_slug}`"
                                   + ("　⚠ 同名の定義が既にあります（上書きされます）" if _exists_def else ""))

                    if st.button("🚀 CVAT にデプロイ", type="primary", use_container_width=True,
                                 disabled=_dep_running or not _dep_dir, key="dep_run_btn"):
                        _out_dir, _fn_name = generate_function_files(
                            fn_dir=_dep_dir,
                            model_run=_dep_run,
                            class_names=_dep_classes,
                            display_name=_dep_disp,
                            task=_dep_meta.get("task") or "detect",
                        )
                        start_deploy(_dep_dir, use_gpu=_dep_gpu)
                        cached_nuclio_functions.clear()
                        st.rerun()

        if _dep_running:
            time.sleep(2)
            st.rerun()

    st.markdown(
        f'<div style="margin-top:8px"><a href="{NUCLIO_WEB}" target="_blank">'
        f'🔗 Nuclio ダッシュボードで詳細を見る</a></div>',
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # ── 新規タスク作成（アノテーションの入口）──────────────────────────────
    st.markdown('<div class="pipeline-card"><h3>➕ CVAT に新しいタスクを作る</h3>',
                unsafe_allow_html=True)
    st.caption(
        "アノテーションしたい画像から CVAT のタスクを直接作ります。"
        "CVAT の画面を開かずに、ここからアノテーションを始められます。"
    )

    _nt_src = st.radio(
        "画像の取得元",
        ["📤 画像をアップロード", "📂 data/ のディレクトリから"],
        horizontal=True, key="nt_src",
    )

    _nt_images: list[Path] = []
    _nt_tmp = PREDICTIONS_DIR / "_newtask_uploads"

    if _nt_src == "📤 画像をアップロード":
        _nt_files = st.file_uploader(
            "アノテーションする画像（複数選択可）",
            type=["jpg", "jpeg", "png", "bmp", "tiff"],
            accept_multiple_files=True, key="nt_files",
        )
        if _nt_files:
            _nt_tmp.mkdir(parents=True, exist_ok=True)
            _cur = {f.name for f in _nt_files}
            _saved = {f.name for f in _nt_tmp.iterdir() if f.is_file()}
            if _cur != _saved:
                for _f in list(_nt_tmp.iterdir()):
                    _f.unlink()
                for _f in _nt_files:
                    (_nt_tmp / _f.name).write_bytes(_f.getbuffer())
            _nt_images = sorted(p for p in _nt_tmp.iterdir() if p.is_file())
            st.caption(f"✅ {len(_nt_images)} 枚を選択中")
    else:
        _nt_dirs = _find_image_dirs(DATA_DIR) if DATA_DIR.exists() else []
        if not _nt_dirs:
            st.info("data/ に画像が見つかりません。")
        else:
            _nt_dir_sel = st.selectbox(
                "画像ディレクトリ",
                [str(d.relative_to(DATA_DIR)) for d in _nt_dirs], key="nt_dir")
            _nt_dir = DATA_DIR / _nt_dir_sel
            _nt_all = sorted(p for p in _nt_dir.iterdir()
                             if p.is_file() and p.suffix.lower() in IMG_EXTS)
            _nt_limit = st.number_input(
                "使用する枚数（先頭から）", 1, max(len(_nt_all), 1),
                min(len(_nt_all), 100), key="nt_limit",
                help="1タスクが大きすぎると作業しづらいので、分割して作るのがおすすめです")
            _nt_images = _nt_all[:int(_nt_limit)]
            st.caption(f"ディレクトリ内 {len(_nt_all)} 枚中 {len(_nt_images)} 枚を使用")

    _ntc1, _ntc2 = st.columns([2, 1])
    with _ntc1:
        _nt_name = st.text_input(
            "タスク名", value=f"annotate_{datetime.now():%Y%m%d_%H%M}", key="nt_name")
    with _ntc2:
        _nt_shape = st.selectbox(
            "アノテーション形式",
            ["rectangle", "polygon", "points", "tag", "any"], key="nt_shape",
            help="rectangle: 物体検出 / polygon: セグメンテーション / "
                 "points: キーポイント / tag: 画像分類 / any: 何でも",
        )

    # ラベルは既存タスクから引き継げるようにする（表記ゆれを防ぐ）
    _nt_known: list[str] = []
    _nt_prev = st.session_state.get("le_labels_by_task") or {}
    for _lbls in _nt_prev.values():
        for _l in _lbls:
            if _l not in _nt_known:
                _nt_known.append(_l)
    _nt_default = ", ".join(_nt_known) if _nt_known else ""
    _nt_labels_raw = st.text_input(
        "ラベル（カンマ区切り）", value=_nt_default, key="nt_labels",
        help="既存タスクと同じ名前にしてください。"
             "自動アノテーションを使う場合は、モデルのクラス名とも一致させる必要があります。",
    )
    _nt_labels = [s.strip() for s in _nt_labels_raw.split(",") if s.strip()]
    if _nt_known and not _nt_labels_raw.strip():
        st.caption(f"💡 取得済みのラベル: {', '.join(_nt_known)}")

    if st.button(f"➕ CVAT にタスクを作成（{len(_nt_images)} 枚）",
                 type="primary", use_container_width=True,
                 disabled=not _nt_images or not _nt_labels or not _nt_name.strip(),
                 key="nt_create"):
        with st.spinner("CVAT にタスクを作成中…（画像アップロード中）"):
            st.session_state["nt_result"] = create_cvat_task_from_images(
                _nt_name.strip(), _nt_images, _nt_labels, label_type=_nt_shape)

    _nt_res = st.session_state.get("nt_result")
    if _nt_res:
        if _nt_res["ok"]:
            st.success(f"✅ タスクを作成しました（ID: {_nt_res['task_id']} / "
                       f"{_nt_res['n_images']} 枚 / ラベル: {', '.join(_nt_res['labels'])}）")
            st.markdown(f"👉 [CVAT でアノテーションを始める]({_nt_res['url']})")
            st.caption(
                "自動アノテーションモデルをデプロイ済みなら、CVAT の "
                "「Actions → Automatic annotation」で下書きを作れます。"
            )
        else:
            show_error(_nt_res["error"], prefix="❌ 作成に失敗しました: ")

    st.markdown('</div>', unsafe_allow_html=True)

    # ── アノテーション進捗 ────────────────────────────────────────────────
    st.markdown('<div class="pipeline-card"><h3>📊 アノテーション進捗</h3>', unsafe_allow_html=True)

    st.caption(
        "進捗はジョブ単位で集計しています。CVAT はタスクを複数のジョブに分割し、"
        "担当者もジョブ単位で割り当てるため、タスクの status より実態に近い数字が出ます。"
    )
    _pc1, _pc2 = st.columns([3, 1])
    with _pc2:
        if st.button("🔄 CVATから進捗を取得", use_container_width=True, key="anno_fetch_tasks"):
            with st.spinner("CVATからタスク・ジョブを取得中…"):
                st.session_state.cvat_tasks = fetch_cvat_tasks()
                st.session_state.cvat_jobs  = fetch_cvat_jobs()

    _anno_tasks = st.session_state.cvat_tasks
    _anno_jobs  = st.session_state.cvat_jobs
    if not _anno_jobs and not _anno_tasks:
        st.info("「CVATから進捗を取得」を押すと、ジョブ単位の進捗と担当者別の状況を表示します。")
    elif not _anno_jobs:
        st.warning("ジョブ情報を取得できませんでした。もう一度「CVATから進捗を取得」を試してください。")
    else:
        import pandas as _pd_anno

        _df_j = _pd_anno.DataFrame(_anno_jobs)
        _df_j["担当者"] = _df_j["assignee"].replace("", None).fillna("（未割当）")

        _total_f = int(_df_j["frames"].fillna(0).sum())
        _done_f  = int(_df_j.loc[_df_j["state"] == "completed", "frames"].fillna(0).sum())
        _rate    = (_done_f / _total_f * 100) if _total_f else 0.0

        _sm1, _sm2, _sm3, _sm4 = st.columns(4)
        _sm1.metric("タスク数", len(_anno_tasks) or _df_j["task_id"].nunique())
        _sm2.metric("ジョブ数", len(_df_j))
        _sm3.metric("総フレーム数", f"{_total_f:,}")
        _sm4.metric("完了率", f"{_rate:.1f}%")
        st.progress(min(_rate / 100, 1.0),
                    text=f"完了 {_done_f:,} / {_total_f:,} フレーム（{_rate:.1f}%）")

        # state / stage の内訳
        _st1, _st2 = st.columns(2)
        with _st1:
            st.markdown("**進行状態 (state)**")
            _by_state = _df_j.groupby("state").agg(
                ジョブ数=("job_id", "count"), フレーム数=("frames", "sum")
            ).reset_index().rename(columns={"state": "状態"})
            st.dataframe(_by_state, use_container_width=True, hide_index=True)
        with _st2:
            st.markdown("**工程 (stage)**")
            _by_stage = _df_j.groupby("stage").agg(
                ジョブ数=("job_id", "count"), フレーム数=("frames", "sum")
            ).reset_index().rename(columns={"stage": "工程"})
            st.dataframe(_by_stage, use_container_width=True, hide_index=True)

        # 担当者別（ジョブ単位。4人以上で分担するときの主指標）
        st.markdown("**👥 担当者別**")
        _by_user = _df_j.groupby("担当者").agg(
            担当ジョブ数=("job_id", "count"),
            フレーム数=("frames", "sum"),
            完了ジョブ数=("state", lambda s: int((s == "completed").sum())),
            完了フレーム数=("frames", "sum"),   # 後で上書きする
        ).reset_index()
        _done_by_user = (_df_j[_df_j["state"] == "completed"]
                         .groupby("担当者")["frames"].sum())
        _by_user["完了フレーム数"] = _by_user["担当者"].map(_done_by_user).fillna(0).astype(int)
        _by_user["完了率"] = (
            _by_user["完了フレーム数"] / _by_user["フレーム数"].replace(0, 1) * 100
        ).round(1).astype(str) + "%"
        st.dataframe(_by_user.sort_values("フレーム数", ascending=False),
                     use_container_width=True, hide_index=True)

        _unassigned = _df_j[_df_j["担当者"] == "（未割当）"]
        if len(_unassigned) > 0:
            st.warning(
                f"⚠ 未割当のジョブが {len(_unassigned)} 件 "
                f"（{int(_unassigned['frames'].sum()):,} フレーム）あります。"
                f"CVAT 側で担当者を割り当ててください。"
            )

        # タスク別の進捗
        st.markdown("**📋 タスク別の進捗**")
        _task_prog = _df_j.groupby(["task_id", "task_name"]).agg(
            ジョブ数=("job_id", "count"),
            フレーム数=("frames", "sum"),
        ).reset_index()
        _done_by_task = (_df_j[_df_j["state"] == "completed"]
                         .groupby("task_id")["frames"].sum())
        _task_prog["完了フレーム"] = _task_prog["task_id"].map(_done_by_task).fillna(0).astype(int)
        _task_prog["進捗"] = (
            _task_prog["完了フレーム"] / _task_prog["フレーム数"].replace(0, 1)
        ).round(3)
        _task_prog["担当者"] = _task_prog["task_id"].map(
            _df_j.groupby("task_id")["担当者"].agg(lambda s: ", ".join(sorted(set(s))))
        )
        _task_prog = _task_prog.rename(columns={"task_id": "ID", "task_name": "タスク名"})

        _only_incomplete = st.checkbox("未完了のタスクだけ表示", value=False,
                                       key="anno_only_incomplete")
        _tp_show = _task_prog[_task_prog["進捗"] < 1.0] if _only_incomplete else _task_prog
        st.dataframe(
            _tp_show[["ID", "タスク名", "担当者", "ジョブ数", "フレーム数", "完了フレーム", "進捗"]],
            use_container_width=True, hide_index=True,
            column_config={"進捗": st.column_config.ProgressColumn(
                "進捗", min_value=0.0, max_value=1.0, format="%.0f%%")},
        )

        st.caption(
            f"CVAT で作業する → [{CVAT_WEB}]({CVAT_WEB})　"
            "／ アノテーションが終わったタスクは「📤 Step2: データ取込」でエクスポートします。"
        )

    st.markdown('</div>', unsafe_allow_html=True)

# ===========================================================================
# タブ1: CVAT エクスポート
# ===========================================================================
with tab1:
    st.markdown("""
<div class="step-banner">
  <div class="sb-title">📤 STEP 2: データ取込</div>
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

        # ─── 手順① CVAT for images 1.1 エクスポート ─────────────────────────
        st.markdown("#### ① CVATエクスポート")
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
                    # 来歴に残すため、どの CVAT タスクから取り込んだかを覚えておく
                    st.session_state["cvat_export_tasks"] = [
                        {"id": t["id"], "name": t["name"], "size": t.get("size")}
                        for t in tasks if t["id"] in selected_ids
                    ]
                    st.session_state.cvat_xml_info = None
                    xml_info = parse_cvat_xml(merged_raw)
                    if xml_info:
                        st.session_state.cvat_xml_info = xml_info

        # ─── 手順② ラベル・タスク種別の設定 ─────────────────────────────────
        if st.session_state.cvat_raw_dir and st.session_state.cvat_xml_info:
            xml_info = st.session_state.cvat_xml_info
            st.markdown("---")
            st.markdown("#### ② ラベルとタスク種別の設定")

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
            if "tag" in ann_types:
                task_type_options.append("classify")
            if "box" in ann_types or "polygon" in ann_types:
                task_type_options.append("obb")

            col_task, col_val = st.columns(2)
            with col_task:
                task_type = st.selectbox(
                    "タスク種別",
                    task_type_options,
                    help="detect: バウンディングボックス / segment: ポリゴン（box→矩形ポリゴンに変換） / "
                         "pose: キーポイント / classify: 画像分類（CVAT の「タグ」から生成） / "
                         "obb: 回転バウンディングボックス（回転付き box・4点ポリゴンから生成）",
                )
            if "tag" not in ann_types:
                st.caption(
                    "💡 画像分類 (classify) を作るには、CVAT で矩形ではなく"
                    "「タグ（画像単位のラベル）」を付けてエクスポートしてください。"
                )
            with col_val:
                val_ratio = st.slider("バリデーション割合", 0.05, 0.40, 0.20, step=0.05)

            # ─── 手順③ データセット生成 ──────────────────────────────────────
            st.markdown("---")
            st.markdown("#### ③ データセット生成")

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
                            cvat_tasks=st.session_state.get("cvat_export_tasks"),
                        )
                    if result:
                        yaml_path = result / "data.yaml"
                        st.success("✅ データセット生成完了！")
                        st.info(
                            f"🗂 data.yaml パス（Step3: モデル学習 タブで使用）:\n`{yaml_path}`"
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
                    st.success("解析完了。下に②が表示されます。")
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
                    record_dataset_provenance(
                        _ul_out, source="upload_zip",
                        extra={"zip_name": _ul_zip.name,
                               "note": "外部から持ち込んだ YOLO データセット ZIP"},
                    )
                    _ul_yamls = list(_ul_out.rglob("data.yaml"))
                    if _ul_yamls:
                        st.info(f"🗂 data.yaml: `{_ul_yamls[0]}`")
                    else:
                        st.warning("data.yaml が見つかりません。Step3で手動入力が必要です。")
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
                    record_dataset_provenance(
                        DATA_DIR / _ul_dir_name, source="upload_images",
                        extra={"note": f"{_ul_split} に画像 {len(_ul_imgs)} 枚を直接アップロード"},
                    )
                    st.info("アノテーションを付与する場合は CVATにアップロード後、Step2からエクスポートしてください。")


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
                   if _yaml_count > 0 else "← 前のステップ: ⚠ Step2でデータセットを先に生成してください")
    st.markdown(f"""
<div class="step-banner">
  <div class="sb-title">🚀 STEP 3: モデル学習</div>
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
                        index=_MODEL_OPTS.index(_ev.get("model","yolo11s")) if _ev.get("model","yolo11s") in _MODEL_OPTS else _MODEL_OPTS.index("その他"),
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

    if _model_preset == "その他":
        model_name = st.text_input(
            "モデルファイル名 (.pt)",
            value="yolo26n.pt",
            help="例: yolo26n.pt, yolo11x.pt, rtdetr-x.pt",
            key="tp_model_custom",
        )
        if model_name and not model_name.endswith(".pt"):
            model_name = model_name + ".pt"
        if model_name:
            _local_candidates = list(MODELS_DIR.rglob(model_name)) + [Path(model_name)]
            if any(p.exists() for p in _local_candidates):
                st.success(f"✅ ローカルに `{model_name}` が見つかりました — ローカルファイルを使用します")
            else:
                st.info(f"⬇️ 学習開始時に Ultralytics が `{model_name}` を自動ダウンロードします（初回のみ）")
        else:
            model_name = "yolo26n.pt"
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

    _yc1, _yc2 = st.columns([10, 1])
    with _yc1:
        _yaml_sel = st.selectbox(
            "data.yaml",
            _yaml_options,
            index=0 if _yaml_labels else len(_yaml_options) - 1,
        )
    with _yc2:
        st.markdown('<div style="margin-top:24px"></div>', unsafe_allow_html=True)
        if st.button("📂", key="btn_ds_dir", help="ディレクトリ内容を表示 / 非表示"):
            st.session_state["_ds_dir_open"] = not st.session_state.get("_ds_dir_open", False)

    if _yaml_sel == _YAML_MANUAL:
        data_yaml_path = st.text_input(
            "data.yaml パスを直接入力 (コンテナ内絶対パス)",
            value=str(DATA_DIR / "dataset/data.yaml"),
        )
    else:
        data_yaml_path = str(DATA_DIR / _yaml_sel)
        _ds_dir_path = Path(data_yaml_path).parent
        # ── データセット詳細パネル ──
        try:
            import yaml as _yaml_mod
            _yaml_content = _yaml_mod.safe_load(Path(data_yaml_path).read_text())
            _nc     = _yaml_content.get("nc", "?")
            _names  = _yaml_content.get("names", [])
            _tr_dir = _ds_dir_path / "images" / "train"
            _vl_dir = _ds_dir_path / "images" / "val"
            _n_tr   = len(list(_tr_dir.glob("*.*"))) if _tr_dir.exists() else "—"
            _n_vl   = len(list(_vl_dir.glob("*.*"))) if _vl_dir.exists() else "—"
            _nm_str = ", ".join(str(n) for n in _names[:10]) + ("…" if len(_names) > 10 else "")
            st.markdown(f"""
<div style="background:#0e1520;border:1px solid #2d6b47;border-left:4px solid #4caf7d;
     border-radius:6px;padding:10px 16px;margin:6px 0 10px;">
  <div style="color:#4caf7d;font-size:.87rem;font-weight:700;margin-bottom:6px;">
    📁 {_ds_dir_path.name}
  </div>
  <div style="display:flex;gap:24px;flex-wrap:wrap;margin-bottom:4px;">
    <span style="color:#6a8aaa;font-size:.82rem;">クラス数: <b style="color:#c8d8e8">{_nc}</b></span>
    <span style="color:#6a8aaa;font-size:.82rem;">Train 画像: <b style="color:#c8d8e8">{_n_tr}</b></span>
    <span style="color:#6a8aaa;font-size:.82rem;">Val 画像: <b style="color:#c8d8e8">{_n_vl}</b></span>
  </div>
  <div style="color:#6a8aaa;font-size:.8rem;">
    ラベル: <span style="color:#c8d8e8">{_nm_str if _nm_str else "—"}</span>
  </div>
  <div style="color:#4a6080;font-size:.75rem;margin-top:4px;">{data_yaml_path}</div>
</div>""", unsafe_allow_html=True)
        except Exception:
            st.code(data_yaml_path, language="text")

        # ── ディレクトリビューア（📂 ボタンでトグル）──
        if st.session_state.get("_ds_dir_open", False):
            with st.container(border=True):
                st.caption(f"📂 {_ds_dir_path}")
                try:
                    _entries = sorted(
                        _ds_dir_path.iterdir(),
                        key=lambda p: (p.is_file(), p.name),
                    )
                    for _e in _entries:
                        if _e.is_dir():
                            _sub_cnt = len(list(_e.iterdir()))
                            st.markdown(
                                f"📁 **{_e.name}/**"
                                f"<span style='color:#4a6080;font-size:.78rem'> ({_sub_cnt} 件)</span>",
                                unsafe_allow_html=True,
                            )
                        else:
                            _sz = _e.stat().st_size
                            _sz_s = (f"{_sz/1024:.1f} KB" if _sz < 1_048_576
                                     else f"{_sz/1048576:.1f} MB")
                            st.markdown(
                                f"📄 {_e.name}"
                                f"<span style='color:#4a6080;font-size:.78rem'> {_sz_s}</span>",
                                unsafe_allow_html=True,
                            )
                except Exception as _dir_err:
                    st.warning(str(_dir_err))

    col_p, col_q = st.columns(2)
    with col_p:
        mlflow_project = st.text_input("MLflow プロジェクト名", value="YOLO-Detection")
    with col_q:
        run_name = st.text_input(
            "ラン名",
            value=f"{_model_preset.replace('その他','custom')}_ep{epochs}_{datetime.now():%H%M}",
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

    # ── データ拡張のプレビュー ────────────────────────────────────────────────
    with st.expander("👁 データ拡張を目で確認する", expanded=False):
        st.caption(
            "上で設定した拡張が画像に何をするかを、学習前に確認できます。"
            "パラメータの意味を掴むためのものです。"
        )

        _ap_params = {
            "hsv_h": float(hsv_h), "hsv_s": float(hsv_s), "hsv_v": float(hsv_v),
            "degrees": float(degrees), "translate": float(translate),
            "scale": float(scale), "shear": float(shear),
            "fliplr": float(fliplr), "flipud": float(flipud),
            "mosaic": float(mosaic), "erasing": float(erasing),
        }

        _ap_active = describe_augment(_ap_params)
        if not _ap_active:
            st.info("有効な拡張がありません。上の「🎨 データ拡張」で値を設定してください。")
        else:
            st.markdown("**有効になっている拡張**")
            for _label, _val, _desc in _ap_active:
                st.caption(f"・**{_label}** = `{_val}` … {_desc}")

        _ap_ds = Path(data_yaml_path).parent if data_yaml_path else None
        _ap_imgs = list_sample_images(_ap_ds) if _ap_ds and _ap_ds.exists() else []

        if not _ap_imgs:
            st.warning("プレビューに使える画像が見つかりません。"
                       "先にデータセットを選択してください。")
        else:
            _apc1, _apc2 = st.columns([1, 2])
            with _apc1:
                _ap_seed = st.number_input("乱数シード", 0, 9999, 0, key="ap_seed",
                                           help="変えると別のかかり方を試せます")
            with _apc2:
                _ap_n = st.slider("表示するパターン数", 1, 4, 3, key="ap_n")

            if st.button("👁 プレビューを作る", use_container_width=True, key="ap_run"):
                with st.spinner("生成中…"):
                    st.session_state["ap_preview"] = build_augment_preview(
                        _ap_imgs, _ap_params, seed=int(_ap_seed), n_variants=int(_ap_n))

            _ap_res = st.session_state.get("ap_preview")
            if _ap_res:
                _orig, _vars = _ap_res
                if _orig is None:
                    st.error("画像を読み込めませんでした。")
                else:
                    _cols = st.columns(len(_vars) + 1)
                    _cols[0].image(_orig, caption="元画像", use_column_width=True)
                    for _c, (_lbl, _im) in zip(_cols[1:], _vars):
                        _c.image(_im, caption=_lbl, use_column_width=True)
                    st.caption(
                        "⚠ 実際の学習では拡張が**確率的に**適用され、"
                        "ここでは効果が見えるよう必ず適用しています。"
                        "見え方の傾向を掴むための近似表示です。"
                    )

    # ── 学習設定サマリー ──────────────────────────────────────────────────────
    _ds_disp = Path(data_yaml_path).parent.name if data_yaml_path else "—"
    st.markdown("#### 📋 学習設定サマリー")
    _sma, _smb, _smc, _smd, _sme = st.columns(5)
    _sma.metric("モデル", model_name)
    _smb.metric("エポック数", str(epochs))
    _smc.metric("バッチ", str(batch_size) if batch_size != -1 else "Auto")
    _smd.metric("imgsz", str(imgsz))
    _sme.metric("patience", str(patience) if patience > 0 else "OFF")
    _smf, _smg, _smh, _smi, _smj = st.columns(5)
    _smf.metric("optimizer", optimizer)
    _smg.metric("lr0", str(lr0))
    _smh.metric("warmup", str(warmup_epochs))
    _smi.metric("dropout", str(dropout))
    _smj.metric("AMP", "ON" if amp else "OFF")
    st.markdown(
        f'<div style="background:#0d1520;border:1px solid #1e2d40;border-radius:6px;'
        f'padding:8px 14px;margin:8px 0 16px;font-size:.82rem;color:#6a8aaa;">'
        f'📁 データセット: <b style="color:#c8d8e8">{_ds_disp}</b>'
        f'<span style="color:#4a6080;font-size:.75rem"> &nbsp;—&nbsp; {data_yaml_path}</span></div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── 中断した学習の再開 ───────────────────────────────────────────────────
    # last.pt があり、results.csv のエポック数が設定より少ない run を候補にする
    _resume_cands = []
    if MODELS_DIR.exists():
        for _last in sorted(MODELS_DIR.glob("*/weights/last.pt"),
                            key=lambda p: p.stat().st_mtime, reverse=True):
            _run_dir = _last.parent.parent
            _done_ep, _total_ep = None, None
            _rcsv = _run_dir / "results.csv"
            if _rcsv.exists():
                try:
                    import pandas as _pd_rs
                    _done_ep = len(_pd_rs.read_csv(_rcsv))
                except Exception:
                    pass
            _args_y = _run_dir / "args.yaml"
            if _args_y.exists():
                try:
                    import yaml as _yml_rs
                    _total_ep = (_yml_rs.safe_load(_args_y.read_text()) or {}).get("epochs")
                except Exception:
                    pass
            # 完走していれば候補から外す（判定できない場合は候補に残す）
            if _done_ep is not None and _total_ep is not None and _done_ep >= int(_total_ep):
                continue
            _resume_cands.append({
                "run": _run_dir.name, "last": _last,
                "done": _done_ep, "total": _total_ep,
            })

    if _resume_cands:
        with st.expander(f"⏯ 中断した学習を再開する（候補 {len(_resume_cands)} 件）"):
            st.caption(
                "停止・クラッシュなどで途中終了した学習を `last.pt` から続きから再開します。"
                "エポック数や学習率などの設定は中断時のものが引き継がれます"
                "（上で設定した値は使われません）。"
            )
            _rs_labels = [
                f"{c['run']}　"
                + (f"({c['done']}/{c['total']} エポック完了)"
                   if c["done"] is not None and c["total"] is not None
                   else f"({c['done']} エポック完了)" if c["done"] is not None else "")
                for c in _resume_cands
            ]
            _rs_sel = st.selectbox("再開する学習", _rs_labels, key="resume_sel")
            _rs_target = _resume_cands[_rs_labels.index(_rs_sel)]
            st.caption(f"再開元: `{_rs_target['last']}`")

            if st.button("⏯ この学習を再開する", type="primary", use_container_width=True,
                         disabled=st.session_state.training_running, key="resume_btn"):
                with _train_log_lock:
                    _train_state["log"] = []
                    _train_state["progress"] = 0
                    _train_state["running"] = True
                    _train_state["error"] = None
                    _train_state["model_path"] = None
                    _train_state["metrics_history"] = []
                    _train_state["stop_requested"] = False
                threading.Thread(
                    target=_train_worker,
                    # resume=True のとき data / epochs 等は last.pt 側の設定が使われる
                    args=(data_yaml_path, str(_rs_target["last"]), 0, 0,
                          mlflow_project, _rs_target["run"], {"resume": True}),
                    daemon=True,
                ).start()
                st.session_state.training_notified = False
                st.rerun()

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

        # ── 停止（エポック末で安全に打ち切る）──
        with _train_log_lock:
            _stop_pending = _train_state.get("stop_requested", False)
        _stop_c1, _stop_c2 = st.columns([1, 3])
        with _stop_c1:
            if st.button("⏹ 学習を停止", type="secondary", use_container_width=True,
                         disabled=_stop_pending, key="train_stop_btn"):
                with _train_log_lock:
                    _train_state["stop_requested"] = True
                st.rerun()
        with _stop_c2:
            if _stop_pending:
                st.warning("⏳ 停止要求を受け付けました。現在のエポックが終わり次第停止します。")
            else:
                st.caption("停止してもその時点までの `best.pt` / `last.pt` は保存されます。"
                           "`last.pt` があれば下の「中断した学習を再開」から続きから再開できます。")

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
        show_error(st.session_state.training_error, prefix="学習エラー: ")

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
                   if _mdl_count3 > 0 else "← 前のステップ: ⚠ Step3でモデルを先に学習してください")
    st.markdown(f"""
<div class="step-banner">
  <div class="sb-title">🔭 STEP 4: 推論・評価</div>
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

    # --- モデル評価（mAP を同一データセットで測る）---
    with st.expander("📊 モデル評価・比較（mAP を同一条件で測る）", expanded=False):
        st.caption(
            "学習時の results.csv は「そのモデルが自分の val で出した値」なので、"
            "別環境で学習したモデルとは比較できません。"
            "ここで同じデータセット・同じ条件で val を回すと、同じ土俵で比べられます。"
        )

        _ev_state, _ev_lock = _get_eval_shared()
        with _ev_lock:
            _ev_running  = _ev_state["running"]
            _ev_log      = list(_ev_state["log"])
            _ev_total    = _ev_state["total"]
            _ev_done     = _ev_state["done"]
            _ev_current  = _ev_state["current"]
            _ev_finished = _ev_state["finished"]
            _ev_error    = _ev_state["error"]

        _ev_yamls = sorted(DATA_DIR.rglob("data.yaml"), key=lambda p: p.stat().st_mtime,
                           reverse=True)
        if not _ev_yamls:
            st.info("評価に使える data.yaml がありません。Step2 でデータセットを作成してください。")
        elif not _model_map:
            st.info("models/ に .pt がありません。")
        else:
            _ev_c1, _ev_c2, _ev_c3 = st.columns([3, 1, 1])
            with _ev_c1:
                _ev_yaml_sel = st.selectbox(
                    "評価に使うデータセット (data.yaml)",
                    [str(p.relative_to(DATA_DIR)) for p in _ev_yamls],
                    key="ev_yaml_sel",
                )
                _ev_yaml_path = str(DATA_DIR / _ev_yaml_sel)
            with _ev_c2:
                _ev_split = st.selectbox("スプリット", ["val", "train"], key="ev_split")
            with _ev_c3:
                _ev_imgsz = st.selectbox("imgsz", [640, 960, 1280], key="ev_imgsz")

            _ev_models_sel = st.multiselect(
                "評価するモデル（複数選択で比較できます）",
                list(_model_map.keys()),
                default=([str(Path(current_model).relative_to(MODELS_DIR))]
                         if current_model and Path(current_model).exists() else []),
                key="ev_models_sel",
            )

            _ev_a1, _ev_a2, _ev_a3 = st.columns(3)
            with _ev_a1:
                _ev_batch = st.number_input("バッチサイズ", 1, 64, 8, key="ev_batch")
            with _ev_a2:
                _ev_conf = st.number_input("conf しきい値", 0.0001, 0.9, 0.001,
                                           format="%.4f", key="ev_conf",
                                           help="mAP は全信頼度域の PR 曲線から計算するため、"
                                                "低い値（既定 0.001）を使うのが正しい計測です")
            with _ev_a3:
                _ev_iou = st.number_input("NMS IoU", 0.1, 0.95, 0.6, key="ev_iou")

            _ev_key = f"{Path(_ev_yaml_path).parent.name}:{_ev_split}"

            if st.button(f"📊 {len(_ev_models_sel)} 件のモデルを評価",
                         type="primary", use_container_width=True,
                         disabled=_ev_running or not _ev_models_sel, key="ev_run"):
                start_evaluation(
                    [_model_map[m] for m in _ev_models_sel],
                    _ev_yaml_path, _ev_split, int(_ev_imgsz),
                    int(_ev_batch), float(_ev_conf), float(_ev_iou),
                )
                st.rerun()

            # --- 実行中 / 完了ログ ---
            if _ev_running or _ev_finished:
                if _ev_running:
                    st.progress(_ev_done / max(_ev_total, 1),
                                text=f"評価中 {_ev_done}/{_ev_total}　{_ev_current}")
                elif _ev_error:
                    show_error(_ev_error, prefix="❌ 評価に失敗しました: ")
                else:
                    st.success("✅ 評価が完了しました")
                st.code("\n".join(_ev_log[-20:]) or "(実行待ち)", language="text")

            # --- 比較表（保存済みの評価結果を横断で集める）---
            _ev_rows = collect_model_evals(_ev_key)
            if _ev_rows:
                st.markdown(f"**📋 比較表 — `{_ev_key}` で評価済みの {len(_ev_rows)} モデル**")
                import pandas as _pd_ev

                _ev_tbl = []
                _has_mask_metric = any(_r.get("mask_map50") is not None for _r in _ev_rows)
                _is_cls_eval = all(_r.get("task") == "classify" for _r in _ev_rows)
                for _r in _ev_rows:
                    _mp = _r["model_path"]
                    _spd = (_r.get("speed_ms") or {}).get("inference")
                    if _is_cls_eval:
                        # 画像分類は mAP ではなく accuracy で比較する
                        _ev_tbl.append({
                            "モデル": str(_mp.relative_to(MODELS_DIR)),
                            "top1 accuracy": round(_r.get("top1") or 0.0, 4),
                            "top5 accuracy": round(_r.get("top5") or 0.0, 4),
                            "推論(ms)": _spd,
                            "サイズ(MB)": round(_mp.stat().st_size / 1024 / 1024, 1),
                            "評価日時": _r.get("evaluated_at", ""),
                        })
                        continue
                    _row = {
                        "モデル": str(_mp.relative_to(MODELS_DIR)),
                        "mAP50": round(_r["map50"], 4),
                        "mAP50-95": round(_r["map50_95"], 4),
                    }
                    # セグメンテーションモデルはマスク基準の mAP も並べる
                    if _has_mask_metric:
                        _row["mask mAP50"] = (round(_r["mask_map50"], 4)
                                              if _r.get("mask_map50") is not None else None)
                        _row["mask mAP50-95"] = (round(_r["mask_map50_95"], 4)
                                                 if _r.get("mask_map50_95") is not None else None)
                    _row.update({
                        "Precision": round(_r["precision"], 3),
                        "Recall": round(_r["recall"], 3),
                        "推論(ms)": _spd,
                        "サイズ(MB)": round(_mp.stat().st_size / 1024 / 1024, 1),
                        "評価日時": _r.get("evaluated_at", ""),
                    })
                    _ev_tbl.append(_row)
                _sort_key = "top1 accuracy" if _is_cls_eval else "mAP50-95"
                _df_ev = _pd_ev.DataFrame(_ev_tbl).sort_values(_sort_key, ascending=False)
                st.dataframe(_df_ev, use_container_width=True, hide_index=True)

                _best = _df_ev.iloc[0]
                if _is_cls_eval:
                    st.success(
                        f"🏆 このデータセットで最も精度が高いのは **{_best['モデル']}** "
                        f"（top1 = {_best['top1 accuracy']:.4f} / "
                        f"top5 = {_best['top5 accuracy']:.4f}）"
                    )
                else:
                    st.success(
                        f"🏆 このデータセットで最も精度が高いのは **{_best['モデル']}** "
                        f"（mAP50-95 = {_best['mAP50-95']:.4f} / mAP50 = {_best['mAP50']:.4f}）"
                    )

                # クラス別 AP と成果物プロット
                _ev_detail_sel = st.selectbox(
                    "詳細を見るモデル", _df_ev["モデル"].tolist(), key="ev_detail_sel")
                _ev_detail = next(
                    (r for r in _ev_rows
                     if str(r["model_path"].relative_to(MODELS_DIR)) == _ev_detail_sel), None)
                if _ev_detail:
                    if _ev_detail.get("per_class"):
                        st.markdown("**クラス別**")
                        st.dataframe(
                            _pd_ev.DataFrame([{
                                "クラス": c["class"],
                                "AP50": round(c["ap50"], 4),
                                "AP50-95": round(c["ap50_95"], 4),
                                "Precision": round(c["precision"], 3),
                                "Recall": round(c["recall"], 3),
                            } for c in _ev_detail["per_class"]]),
                            use_container_width=True, hide_index=True,
                        )
                    _pd_dir = _ev_detail.get("plots_dir")
                    if _pd_dir and Path(_pd_dir).exists():
                        _cm = Path(_pd_dir) / "confusion_matrix_normalized.png"
                        _pr = Path(_pd_dir) / "BoxPR_curve.png"
                        _pcols = st.columns(2)
                        if _cm.exists():
                            _pcols[0].image(str(_cm), caption="混同行列（正規化）",
                                            use_column_width=True)
                        if _pr.exists():
                            _pcols[1].image(str(_pr), caption="Precision-Recall 曲線",
                                            use_column_width=True)
            else:
                st.caption(f"`{_ev_key}` での評価結果はまだありません。")

        if _ev_running:
            time.sleep(2)
            st.rerun()

    # --- GT との差分分析（ラベル漏れ・誤ラベルの発見）---
    # --- 実運用の conf を決める ---
    with st.expander("🎚 最適な信頼度しきい値 (conf) を探す", expanded=False):
        st.caption(
            "mAP は「モデルの実力」を測る指標ですが、実際に使うときは "
            "**どの conf で運用するか**を決める必要があります。"
            "しきい値を振って Precision / Recall / F1 を測り、判断材料を出します。"
        )

        _sw_yamls = sorted(DATA_DIR.rglob("data.yaml"), key=lambda p: p.stat().st_mtime,
                           reverse=True)
        if not _sw_yamls or not _model_map:
            st.info("data.yaml と学習済みモデルの両方が必要です。")
        else:
            _swc1, _swc2 = st.columns([3, 2])
            with _swc1:
                _sw_yaml_sel = st.selectbox(
                    "データセット (data.yaml)",
                    [str(p.relative_to(DATA_DIR)) for p in _sw_yamls], key="sw_yaml")
            with _swc2:
                _sw_model_sel = st.selectbox(
                    "モデル", list(_model_map.keys()),
                    index=(list(_model_map.values()).index(current_model)
                           if current_model in _model_map.values() else 0),
                    key="sw_model")

            _swp1, _swp2, _swp3 = st.columns(3)
            with _swp1:
                _sw_split = st.selectbox("スプリット", ["val", "train"], key="sw_split")
            with _swp2:
                _sw_iou = st.slider("一致とみなす IoU", 0.1, 0.9, 0.5, 0.05, key="sw_iou")
            with _swp3:
                _sw_max = st.number_input("最大画像数", 0, 100000, 300, 100, key="sw_max",
                                          help="0 で全画像")

            if st.button("🎚 しきい値を振って測る", type="primary",
                         use_container_width=True, key="sw_run"):
                with st.spinner("推論して各しきい値で評価しています…"):
                    st.session_state["sw_result"] = sweep_confidence(
                        Path(_model_map[_sw_model_sel]),
                        str(DATA_DIR / _sw_yaml_sel), split=_sw_split,
                        iou_match=float(_sw_iou), max_images=int(_sw_max),
                    )

            _sw = st.session_state.get("sw_result")
            if _sw and not _sw["ok"]:
                show_error(_sw["error"], prefix="❌ 測定に失敗しました: ")
            elif _sw:
                import pandas as _pd_sw

                _df_sw = _pd_sw.DataFrame([{
                    "conf": r["conf"], "Precision": round(r["precision"], 3),
                    "Recall": round(r["recall"], 3), "F1": round(r["f1"], 3),
                    "TP": r["tp"], "FP": r["fp"], "FN": r["fn"],
                } for r in _sw["rows"]])

                st.markdown(f"**{_sw['n_images']} 枚で測定**（IoU {_sw['iou_match']} で一致判定）")
                st.line_chart(_df_sw.set_index("conf")[["Precision", "Recall", "F1"]])

                _b = _sw["best_f1"]
                _hp, _hr = _sw["high_precision"], _sw["high_recall"]
                _rc1, _rc2, _rc3 = st.columns(3)
                with _rc1:
                    st.metric("バランス重視 (F1最大)", f"{_b['conf']:.2f}" if _b else "—")
                    if _b:
                        st.caption(f"P {_b['precision']:.3f} / R {_b['recall']:.3f} / "
                                   f"F1 {_b['f1']:.3f}")
                with _rc2:
                    st.metric("誤検出を避ける", f"{_hp['conf']:.2f}" if _hp else "—")
                    st.caption(f"P {_hp['precision']:.3f} / R {_hp['recall']:.3f}"
                               if _hp else "Precision 0.95 以上に届く点がありません")
                with _rc3:
                    st.metric("見逃しを避ける", f"{_hr['conf']:.2f}" if _hr else "—")
                    st.caption(f"P {_hr['precision']:.3f} / R {_hr['recall']:.3f}"
                               if _hr else "Recall 0.95 以上を保てる点がありません")

                st.caption(
                    "用途に合わせて選んでください。"
                    "検査や安全用途で見逃したくないなら Recall 寄り（低め）、"
                    "自動処理で誤検出を出したくないなら Precision 寄り（高め）、"
                    "自動アノテーションの下書きなら少し低めが便利です"
                    "（消す方が描くより速いため）。"
                )
                st.dataframe(_df_sw, use_container_width=True, hide_index=True, height=260)

    with st.expander("🔬 正解ラベルとの差分分析（アノテーション漏れを探す）", expanded=False):
        st.caption(
            "モデルの予測を正解ラベル(GT)と突き合わせ、画像ごとに "
            "**FN（取りこぼし）** と **FP（余計な検出）** を数えます。"
            "精度の高いモデルが FN を出す画像は、モデルの誤りではなく "
            "**GT 側のアノテーションが漏れている**ことがよくあります。"
        )

        _gd_yamls = sorted(DATA_DIR.rglob("data.yaml"), key=lambda p: p.stat().st_mtime,
                           reverse=True)
        if not _gd_yamls or not _model_map:
            st.info("data.yaml と学習済みモデルの両方が必要です。")
        else:
            _gd_c1, _gd_c2 = st.columns([3, 2])
            with _gd_c1:
                _gd_yaml_sel = st.selectbox(
                    "データセット (data.yaml)",
                    [str(p.relative_to(DATA_DIR)) for p in _gd_yamls], key="gd_yaml")
                _gd_yaml = str(DATA_DIR / _gd_yaml_sel)
            with _gd_c2:
                _gd_model_sel = st.selectbox(
                    "使用するモデル", list(_model_map.keys()),
                    index=(list(_model_map.values()).index(current_model)
                           if current_model in _model_map.values() else 0),
                    key="gd_model")

            _gd_p1, _gd_p2, _gd_p3, _gd_p4 = st.columns(4)
            with _gd_p1:
                _gd_split = st.selectbox("スプリット", ["val", "train"], key="gd_split")
            with _gd_p2:
                _gd_conf = st.slider("推論 conf", 0.05, 0.9, 0.25, 0.05, key="gd_conf")
            with _gd_p3:
                _gd_iou = st.slider("一致とみなす IoU", 0.1, 0.9, 0.5, 0.05, key="gd_iou")
            with _gd_p4:
                _gd_max = st.number_input("最大画像数", 0, 100000, 500, 100, key="gd_max",
                                          help="0 で全画像。多いほど時間がかかります")

            if st.button("🔬 差分を分析", type="primary", use_container_width=True,
                         key="gd_run"):
                with st.spinner("推論して GT と突き合わせています…"):
                    st.session_state["gd_result"] = compare_with_ground_truth(
                        Path(_model_map[_gd_model_sel]), _gd_yaml, split=_gd_split,
                        conf=float(_gd_conf), iou_match=float(_gd_iou),
                        max_images=int(_gd_max),
                    )

            _gd = st.session_state.get("gd_result")
            if _gd and not _gd["ok"]:
                show_error(_gd["error"], prefix="❌ 分析に失敗しました: ")
            elif _gd:
                import pandas as _pd_gd

                _gd_imgs = _gd["per_image"]
                _n_clean = sum(1 for p in _gd_imgs if p["fp"] == 0 and p["fn"] == 0)

                _gm = st.columns(5)
                _gm[0].metric("画像数", _gd["n_images"])
                _gm[1].metric("TP（一致）", _gd["tp"])
                _gm[2].metric("FN（取りこぼし）", _gd["fn"])
                _gm[3].metric("FP（余計な検出）", _gd["fp"])
                _gm[4].metric("完全一致", f"{_n_clean}/{_gd['n_images']}")
                if _gd["precision"] is not None:
                    st.caption(f"Precision {_gd['precision']:.3f} / "
                               f"Recall {_gd['recall']:.3f}"
                               f"（conf={_gd['conf']}, IoU={_gd['iou_match']} での実測）")

                if _gd["by_class"]:
                    st.markdown("**クラス別**")
                    st.dataframe(_pd_gd.DataFrame([
                        {"クラス": k, "TP": v["tp"], "FP": v["fp"], "FN": v["fn"]}
                        for k, v in _gd["by_class"].items()
                    ]), use_container_width=True, hide_index=True)

                # 要確認画像の抽出条件
                st.markdown("**要確認画像の抽出**")
                _gf1, _gf2 = st.columns(2)
                with _gf1:
                    _gd_min_fn = st.number_input("FN が この件数以上", 0, 50, 1, key="gd_min_fn")
                with _gf2:
                    _gd_min_fp = st.number_input("または FP が この件数以上", 0, 50, 2,
                                                 key="gd_min_fp")

                _gd_hits = [p for p in _gd_imgs
                            if (_gd_min_fn and p["fn"] >= _gd_min_fn)
                            or (_gd_min_fp and p["fp"] >= _gd_min_fp)]
                _gd_hits.sort(key=lambda d: -(d["fn"] * 2 + d["fp"]))

                st.markdown(f"該当: **{len(_gd_hits)}** 件"
                            f"（差分の大きい順。FN を重く重み付けしています）")
                if _gd_hits:
                    st.dataframe(_pd_gd.DataFrame([{
                        "ファイル": p["name"], "GT": p["n_gt"], "予測": p["n_pred"],
                        "TP": p["tp"], "FP": p["fp"], "FN": p["fn"],
                    } for p in _gd_hits]), use_container_width=True, hide_index=True,
                        height=260)

                    _ga1, _ga2 = st.columns(2)
                    with _ga1:
                        _gd_task = st.text_input(
                            "CVAT タスク名",
                            value=f"labelfix_{datetime.now():%Y%m%d_%H%M}",
                            key="gd_task_name")
                        if st.button(f"📤 {len(_gd_hits)} 件を CVAT に送る",
                                     type="primary", use_container_width=True,
                                     disabled=not _gd_task.strip(), key="gd_push"):
                            _gd_items = [{
                                "path": Path(p["image"]), "width": p["width"],
                                "height": p["height"], "boxes": p["pred_boxes"],
                            } for p in _gd_hits]
                            _gd_labels = sorted({b["label"] for it in _gd_items
                                                 for b in it["boxes"]})
                            if not _gd_labels:
                                _gd_labels = list(_gd["by_class"].keys())
                            with st.spinner("CVAT にタスクを作成中…"):
                                st.session_state["gd_push_result"] = push_items_to_cvat(
                                    _gd_items, _gd_labels, _gd_task.strip(),
                                    with_annotations=True)
                        _gdp = st.session_state.get("gd_push_result")
                        if _gdp:
                            if _gdp["ok"]:
                                st.success(f"✅ タスク作成（ID: {_gdp['task_id']} / "
                                           f"{_gdp['n_images']} 枚）")
                                st.markdown(f"👉 [CVAT で開く]({_gdp['url']})")
                            else:
                                st.error(f"❌ {_gdp['error']}")
                    with _ga2:
                        _gd_fo_name = st.text_input(
                            "FiftyOne データセット名", value="gt_vs_pred",
                            key="gd_fo_name")
                        if st.button("🔭 FiftyOne で GT と予測を見比べる",
                                     use_container_width=True, key="gd_fo"):
                            with st.spinner("FiftyOne App を起動中…"):
                                _gd_port = launch_fiftyone_comparison(
                                    _gd_fo_name.strip() or "gt_vs_pred", _gd_hits)
                            if _gd_port:
                                st.success(f"起動しました → http://localhost:{_gd_port}")
                                st.caption("`ground_truth`（正解）と `predictions`（予測）を"
                                           "重ねて表示できます。`n_fn` / `n_fp` でソートも可能です。")
                else:
                    st.success("✅ 条件に該当する画像はありませんでした。")

    # --- 推論対象ソース ---
    _infer_src = st.radio(
        "推論対象ソース",
        ["📂 data/ のディレクトリ", "📤 画像をアップロード", "🎬 動画をアップロード"],
        horizontal=True,
        key="infer_src_mode",
    )

    test_image_dir = ""
    test_video_path = ""

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

    elif _infer_src == "🎬 動画をアップロード":
        _infer_video = st.file_uploader(
            "推論したい動画ファイル（MP4 / AVI / MOV / MKV）",
            type=["mp4", "avi", "mov", "webm", "mkv"],
            accept_multiple_files=False,
            key="infer_upload_video",
        )
        if _infer_video:
            st.caption(f"✅ {_infer_video.name} 選択中")
            PREDICTIONS_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
            _saved_video = PREDICTIONS_VIDEOS_DIR / _infer_video.name
            _saved_video.write_bytes(_infer_video.getbuffer())
            test_video_path = str(_saved_video)

        # トラッキング設定
        _track_enabled = st.checkbox("🔄 オブジェクトトラッキングを有効にする", value=False, key="video_track_enabled")
        if _track_enabled:
            _tracker_choice = st.radio(
                "トラッカー",
                ["ByteTrack", "BoT-SORT"],
                horizontal=True,
                key="video_tracker_choice",
                help="ByteTrack: 高速・位置ベース。BoT-SORT: 高精度・外観特徴も使用（遮蔽に強い）",
            )
            _tracker_yaml = "/app/bytetrack.yaml" if _tracker_choice == "ByteTrack" else "/app/botsort.yaml"
        else:
            _tracker_yaml = "/app/bytetrack.yaml"

        # テンポラル平滑化設定
        _smooth_enabled = st.checkbox(
            "🕐 テンポラル平滑化（ちらつき抑制）",
            value=False,
            key="video_smooth_enabled",
            help="直近Nフレームの検出を記憶し、一時的に消えた検出をグレーで補完描画します",
        )
        if _smooth_enabled:
            _smooth_frames = st.slider(
                "補完フレーム数",
                min_value=1, max_value=30, value=5, step=1,
                key="video_smooth_frames",
                help="検出が消えてから何フレームまで補完するか。大きいほどちらつきが減るが残像が増える",
            )
        else:
            _smooth_frames = 0

        if compare_mode:
            st.info("ℹ 動画モードでは複数モデル比較は使用できません。メインモデルで推論します。")

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
            or (not test_image_dir and not test_video_path)
        )
        if st.button("▶ 推論実行", type="primary", use_container_width=True,
                    disabled=_infer_disabled):

            # ── 動画推論 ──────────────────────────────────────────────
            if _infer_src == "🎬 動画をアップロード":
                if not test_video_path:
                    st.error("動画ファイルを選択してください")
                elif not current_model:
                    st.error("モデルが未設定です。Step3 で学習するか、データ管理タブで選択してください")
                else:
                    _prog_bar = st.progress(0.0, text="動画推論中…")
                    def _video_prog(fi, tot):
                        if tot > 0:
                            _prog_bar.progress(min(fi / tot, 1.0),
                                               text=f"フレーム {fi}/{tot} 処理中…")
                    video_result = run_video_inference(
                        current_model,
                        Path(test_video_path),
                        PREDICTIONS_VIDEOS_DIR,
                        conf_threshold=inf_conf,
                        enable_tracking=_track_enabled,
                        tracker=_tracker_yaml,
                        temporal_smoothing=_smooth_enabled,
                        smooth_frames=_smooth_frames,
                        progress_cb=_video_prog,
                    )
                    _prog_bar.empty()
                    if video_result:
                        st.success(
                            f"✅ 動画推論完了: {video_result['total_frames']} フレーム処理"
                        )
                        st.session_state.last_video_result = video_result

            # ── 画像推論 ──────────────────────────────────────────────
            else:
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

    # --- 動画推論結果 ---
    _vr = st.session_state.get("last_video_result")
    if _vr:
        st.markdown("#### 🎬 動画推論結果")
        _out_video = _vr.get("video_path")
        _frame_stats = _vr.get("frame_stats", [])
        _total_frames = _vr.get("total_frames", 0)

        _vc1, _vc2 = st.columns([3, 2])
        with _vc1:
            if _out_video and Path(_out_video).exists():
                with open(_out_video, "rb") as _vf:
                    _video_bytes = _vf.read()
                st.video(_video_bytes)
                st.download_button(
                    "⬇ アノテーション済み動画をダウンロード",
                    _video_bytes,
                    file_name=Path(_out_video).name,
                    mime="video/mp4",
                    use_container_width=True,
                )
            else:
                st.warning("出力動画ファイルが見つかりません")

        with _vc2:
            if _frame_stats:
                import pandas as pd
                _df_frames = pd.DataFrame([
                    {"フレーム": s["frame"], "検出数": s["detections"]}
                    for s in _frame_stats
                ])
                _total_det = sum(s["detections"] for s in _frame_stats)
                _det_frames = sum(1 for s in _frame_stats if s["detections"] > 0)

                # トラッキング時: ユニーク ID 数を集計
                _all_track_ids = {
                    b["track_id"]
                    for s in _frame_stats
                    for b in s["boxes"]
                    if "track_id" in b
                }
                _is_tracked = len(_all_track_ids) > 0

                st.metric("総フレーム数", _total_frames)
                st.metric("検出フレーム数", _det_frames)
                st.metric("総検出数", _total_det)
                if _is_tracked:
                    st.metric("ユニークトラック数", len(_all_track_ids))

                st.markdown("**フレームごとの検出数**")
                st.line_chart(_df_frames.set_index("フレーム"))

        if st.button("🗑 この動画結果をクリア", key="clear_video_result"):
            st.session_state.last_video_result = None
            st.rerun()

    # --- 推論結果 画像プレビュー ---
    _pred_jsons = sorted(PREDICTIONS_DIR.glob("*.json"))
    if _pred_jsons:
        _reanno_count = len(st.session_state.reanno_set)
        _prev_header_c1, _prev_header_c2 = st.columns([4, 2])
        with _prev_header_c1:
            st.markdown("#### 🖼 推論結果プレビュー")
        with _prev_header_c2:
            if _reanno_count > 0:
                st.markdown(
                    f'<div style="padding-top:10px; color:#f4a84e; font-size:.85rem;">'
                    f'🚩 再アノテーション: <b>{_reanno_count}</b> 件</div>',
                    unsafe_allow_html=True,
                )
        _preview_jsons = _pred_jsons[:9]
        for _row_start in range(0, len(_preview_jsons), 3):
            _row_files = _preview_jsons[_row_start:_row_start + 3]
            _row_cols = st.columns(3)
            for _col, _jf in zip(_row_cols, _row_files):
                _res = _draw_predictions(_jf)
                with _col:
                    if _res:
                        _img, _n_boxes, _stem = _res
                        st.image(_img, caption=f"{_stem} ({_n_boxes}件検出)",
                                 use_column_width=True)
                    else:
                        st.caption(prediction_display_name(_jf))
                    _is_flagged = _jf.name in st.session_state.reanno_set
                    _flag_label = "🚩 フラグ解除" if _is_flagged else "🚩 再アノテーション要"
                    if st.button(_flag_label, key=f"prev_flag_{_jf.name}",
                                 use_container_width=True,
                                 type="secondary"):
                        if _is_flagged:
                            st.session_state.reanno_set.discard(_jf.name)
                        else:
                            st.session_state.reanno_set.add(_jf.name)
                        st.rerun()
        if len(_pred_jsons) > 9:
            st.caption(f"（他 {len(_pred_jsons) - 9} 件は省略。全件エクスポートの選択グリッドからフラグ付け可能）")

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

            # ── 画像グリッド + チェックボックス + フラグ ──
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
                            st.caption(prediction_display_name(_jf))
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
                        _is_flagged_sel = _jf.name in st.session_state.reanno_set
                        _flag_lbl_sel = "🚩 解除" if _is_flagged_sel else "🚩 要再アノテ"
                        if st.button(_flag_lbl_sel, key=f"sel_flag_{_cur_page}_{_jf.name}",
                                     use_container_width=True, type="secondary"):
                            if _is_flagged_sel:
                                st.session_state.reanno_set.discard(_jf.name)
                            else:
                                st.session_state.reanno_set.add(_jf.name)
                            st.rerun()

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

    # --- 要確認画像の自動抽出 ---
    if _pred_jsons:
        st.markdown("#### 🔍 要確認画像の自動抽出")
        st.caption(
            "推論結果を分析して「モデルが自信を持てていない画像」を機械的に拾います。"
            "全画像を目視して 🚩 を立てる代わりに、ここで一括フラグできます。"
        )

        _aa_c1, _aa_c2 = st.columns([2, 3])
        with _aa_c1:
            _aa_conf = st.slider("低信頼度のしきい値", 0.05, 0.95, 0.50, 0.05,
                                 key="aa_conf_low",
                                 help="この値未満の検出を含む画像を要確認とみなします")
        with _aa_c2:
            st.markdown("**抽出する条件**")
            _ac1, _ac2 = st.columns(2)
            with _ac1:
                _aa_zero     = st.checkbox("検出ゼロ（見逃しの疑い）", value=True, key="aa_zero")
                _aa_low      = st.checkbox("低信頼度を含む", value=True, key="aa_low")
            with _ac2:
                _aa_conflict = st.checkbox("クラス競合（迷っている）", value=True, key="aa_conflict")
                _aa_tiny     = st.checkbox("極小ボックス（ノイズの疑い）", value=False, key="aa_tiny")

        if st.button("🔍 要確認画像を抽出", use_container_width=True, key="aa_run"):
            with st.spinner(f"{len(_pred_jsons)} 件を分析中…"):
                st.session_state["aa_rows"] = analyze_predictions(
                    _pred_jsons, conf_low=_aa_conf)

        _aa_rows = st.session_state.get("aa_rows") or []
        if _aa_rows:
            # チェックした条件だけを採用する
            _aa_want = set()
            if _aa_zero:     _aa_want.add("検出ゼロ")
            if _aa_low:      _aa_want.add("低信頼度")
            if _aa_conflict: _aa_want.add("クラス競合")
            if _aa_tiny:     _aa_want.add("極小ボックス")

            _aa_hits = []
            for _r in _aa_rows:
                _kinds = {x.split("(")[0] for x in _r["reasons"]}
                if _kinds & _aa_want:
                    _aa_hits.append({**_r, "matched": sorted(_kinds & _aa_want)})

            _am1, _am2, _am3 = st.columns(3)
            _am1.metric("分析した画像", len(_aa_rows))
            _am2.metric("要確認", len(_aa_hits))
            _am3.metric("要確認の割合",
                        f"{len(_aa_hits) / len(_aa_rows) * 100:.1f}%" if _aa_rows else "—")

            # 理由別の内訳
            _aa_agg: dict[str, int] = {}
            for _h in _aa_hits:
                for _k in _h["matched"]:
                    _aa_agg[_k] = _aa_agg.get(_k, 0) + 1
            if _aa_agg:
                st.markdown("　".join(f"`{k}` {v}件" for k, v in sorted(_aa_agg.items())))

            if _aa_hits:
                import pandas as _pd_aa
                _df_aa = _pd_aa.DataFrame([{
                    "ファイル": _h.get("display_name") or _h["name"],
                    "検出数": _h["n_boxes"],
                    "最低conf": (f"{_h['min_conf']:.2f}" if _h["min_conf"] is not None else "—"),
                    "理由": ", ".join(_h["reasons"]),
                } for _h in _aa_hits])
                st.dataframe(_df_aa, use_container_width=True, hide_index=True, height=260)

                _ab1, _ab2 = st.columns(2)
                with _ab1:
                    if st.button(f"🚩 {len(_aa_hits)} 件にまとめてフラグを立てる",
                                 type="primary", use_container_width=True, key="aa_flag_all"):
                        for _h in _aa_hits:
                            st.session_state.reanno_set.add(_h["name"])
                        st.success(f"{len(_aa_hits)} 件にフラグを立てました")
                        st.rerun()
                with _ab2:
                    if st.button("抽出結果をクリア", use_container_width=True, key="aa_clear"):
                        st.session_state["aa_rows"] = []
                        st.rerun()
            else:
                st.success("✅ 選択した条件に該当する画像はありませんでした。")

    # --- 再アノテーション用エクスポート ---
    if _pred_jsons:
        st.markdown("#### 🚩 再アノテーション用エクスポート")
        _ra_set  = st.session_state.reanno_set
        _ra_jsons = [PREDICTIONS_DIR / n for n in sorted(_ra_set)
                     if (PREDICTIONS_DIR / n).exists()]

        if not _ra_jsons:
            st.info("上の自動抽出、またはプレビューの 🚩 ボタンで画像にフラグを立てると、ここに表示されます。")
        else:
            st.markdown(
                f'<div style="color:#f4a84e; font-size:.9rem; margin-bottom:8px;">'
                f'🚩 フラグ済み: <b>{len(_ra_jsons)}</b> 件</div>',
                unsafe_allow_html=True,
            )

            # フラグ済み画像のサムネイル一覧
            with st.expander(f"フラグ済み画像を確認する（{len(_ra_jsons)} 件）", expanded=False):
                for _ra_row in range(0, len(_ra_jsons), 3):
                    _ra_cols = st.columns(3)
                    for _rc, _rj in zip(_ra_cols, _ra_jsons[_ra_row:_ra_row + 3]):
                        with _rc:
                            _rr = _draw_predictions(_rj)
                            if _rr:
                                _ri, _rn, _rs = _rr
                                st.image(_ri, caption=f"{_rs} ({_rn}件)", use_column_width=True)
                            else:
                                st.caption(_rj.stem)

            # ── CVAT へ直接送る（ZIP ダウンロード → 手動アップロードを不要にする）──
            with st.expander("📤 CVAT に新規タスクとして送る（推奨）", expanded=True):
                st.caption(
                    "フラグ済み画像を CVAT のタスクとして直接作成します。"
                    "予測ボックスを事前アノテーションとして入れておけば、"
                    "作業者はゼロから引くのではなく「直す」だけで済みます。"
                )
                _pu_name = st.text_input(
                    "CVAT タスク名",
                    value=f"recheck_{datetime.now():%Y%m%d_%H%M}",
                    key="push_task_name",
                )
                _pu_c1, _pu_c2 = st.columns(2)
                with _pu_c1:
                    _pu_with_ann = st.checkbox(
                        "予測ボックスを事前アノテーションとして入れる",
                        value=True, key="push_with_ann",
                    )
                with _pu_c2:
                    _pu_extra = st.text_input(
                        "追加ラベル（カンマ区切り・任意）",
                        value="", key="push_extra_labels",
                        help="検出ゼロの画像だけを送る場合や、"
                             "予測に出てこないクラスを後から付けたい場合に指定します",
                    )

                if len(_ra_jsons) > 200:
                    st.warning(f"⚠ {len(_ra_jsons)} 件を送信します。画像のアップロードに時間がかかります。")

                if st.button(f"📤 CVAT に {len(_ra_jsons)} 件を送る",
                             type="primary", use_container_width=True,
                             disabled=not _pu_name.strip(), key="push_run"):
                    _pu_labels = [s.strip() for s in _pu_extra.split(",") if s.strip()]
                    with st.spinner("CVAT にタスクを作成中…（画像アップロード中）"):
                        _pu_res = push_predictions_to_cvat(
                            _ra_jsons,
                            task_name=_pu_name.strip(),
                            extra_labels=_pu_labels,
                            with_annotations=_pu_with_ann,
                        )
                    st.session_state["push_result"] = _pu_res

                _pu_last = st.session_state.get("push_result")
                if _pu_last:
                    if _pu_last["ok"]:
                        st.success(
                            f"✅ タスクを作成しました（ID: {_pu_last['task_id']} / "
                            f"{_pu_last['n_images']} 枚 / ラベル: {', '.join(_pu_last['labels'])}）"
                        )
                        st.markdown(f"👉 [CVAT でこのタスクを開く]({_pu_last['url']})")
                        st.caption("作業が終わったら「📤 Step2: データ取込」でエクスポートして学習に回せます。")
                    else:
                        show_error(_pu_last["error"], prefix="❌ 送信に失敗しました: ")

            st.caption(
                "ZIP 出力形式: 元画像 (`images/`) + YOLO txt ラベル (`labels/`) "
                "+ `classes.txt` + CVAT for images 1.1 XML (`annotations.xml`)"
            )
            _ra_c1, _ra_c2 = st.columns(2)
            with _ra_c1:
                if st.button("⬇ 再アノテーション用 ZIP を生成",
                             type="primary", use_container_width=True):
                    with st.spinner("ZIP を生成中…"):
                        _zip_bytes, _ok, _ng = build_reannotation_zip(_ra_jsons)
                    if _ok > 0:
                        st.download_button(
                            f"⬇ ダウンロード（{_ok} 件）",
                            _zip_bytes,
                            file_name=f"reannotation_{datetime.now():%Y%m%d_%H%M}.zip",
                            mime="application/zip",
                            use_container_width=True,
                        )
                    if _ng > 0:
                        st.warning(f"⚠ {_ng} 件は元画像が見つからずスキップしました")
            with _ra_c2:
                if st.button("🗑 フラグをすべてクリア", use_container_width=True):
                    st.session_state.reanno_set = set()
                    st.rerun()

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
            _ds_prov = read_provenance(ds)
            if _ds_prov:
                _src_label = {
                    "cvat": "CVAT から生成", "upload_zip": "ZIP を取込",
                    "upload_images": "画像を直接アップロード", "merge": "データセット統合",
                }.get(_ds_prov.get("source", ""), _ds_prov.get("source", "不明"))
                _tasks_txt = ", ".join(
                    f"[{t.get('id')}] {t.get('name')}" for t in (_ds_prov.get("cvat_tasks") or [])
                )
                st.caption(
                    f"📚 {_src_label}"
                    + (f"（{_ds_prov.get('task_type')}）" if _ds_prov.get("task_type") else "")
                    + f"　作成: {_ds_prov.get('created_at', '不明')}"
                    + (f"　元タスク: {_tasks_txt}" if _tasks_txt else "")
                )
            else:
                st.caption("📚 来歴の記録なし（この機能を入れる前に作られたデータセットです）")

            with st.expander(f"✂️ {ds.name} の train/val を分け直す"):
                st.caption(
                    "生成時に決めた比率のままだと「val が偏っていて評価が信用できない」"
                    "ときに手が出せません。ここで混ぜ直せます。"
                    "画像とラベルは対のまま移動し、何度でもやり直せます。"
                )
                _rs_before = dataset_split_counts(ds)
                st.caption("現在: " + (" / ".join(f"{k} {v}枚" for k, v in _rs_before.items())
                                       or "（画像なし）"))
                _rsc1, _rsc2 = st.columns(2)
                with _rsc1:
                    _rs_ratio = st.slider("val の割合", 0.05, 0.50, 0.20, 0.05,
                                          key=f"rs_ratio_{ds.name}")
                with _rsc2:
                    _rs_seed = st.number_input(
                        "乱数シード", 0, 9999, 0, key=f"rs_seed_{ds.name}",
                        help="同じ値なら同じ分け方になります。変えると別の組み合わせを試せます")
                if st.button("✂️ 分け直す", key=f"rs_run_{ds.name}",
                             use_container_width=True):
                    with st.spinner("分割し直しています…"):
                        _rs = resplit_dataset(ds, val_ratio=float(_rs_ratio),
                                              seed=int(_rs_seed))
                    if _rs["error"]:
                        show_error(_rs["error"], prefix="❌ 分割に失敗しました: ")
                    else:
                        st.success(
                            "✅ 分割し直しました　"
                            + " / ".join(f"{k} {v}枚" for k, v in _rs["after"].items())
                            + f"（{_rs['moved']} 件を移動）"
                        )
                        st.rerun()

            with st.expander(f"🏷 {ds.name} のクラス名を編集・統合する"):
                _cls_names = dataset_class_names(ds)
                if not _cls_names:
                    st.info("data.yaml にクラス定義がありません。")
                else:
                    st.caption(
                        "クラス名の変更・統合・削除ができます。"
                        "複数のクラスに同じ新しい名前を付けると統合されます。"
                        "空欄にするとそのクラスのアノテーションを削除します。"
                        "ラベルは `.txt.bak` にバックアップしてから書き換えます。"
                    )
                    _mapping: dict = {}
                    for _cn in _cls_names:
                        _mapping[_cn] = st.text_input(
                            f"`{_cn}` →", value=_cn, key=f"cls_map_{ds.name}_{_cn}",
                        ).strip()

                    _new_list: list[str] = []
                    for _cn in _cls_names:
                        _nv = _mapping[_cn]
                        if _nv and _nv not in _new_list:
                            _new_list.append(_nv)
                    _removed = [c for c in _cls_names if not _mapping[c]]

                    if _new_list != _cls_names or _removed:
                        st.markdown(f"変更後のクラス: **{', '.join(_new_list) or '（なし）'}**")
                        if _removed:
                            st.warning(f"⚠ 削除されるクラス: {', '.join(_removed)}"
                                       "（該当するアノテーションが消えます）")
                        if not _new_list:
                            st.error("すべてのクラスが削除対象です。1つ以上残してください。")
                        elif st.button("🏷 クラスを更新する", key=f"cls_run_{ds.name}",
                                       type="primary", use_container_width=True):
                            with st.spinner("ラベルを書き換えています…"):
                                _rm = remap_dataset_classes(
                                    ds, {k: (v or None) for k, v in _mapping.items()})
                            if _rm["error"]:
                                show_error(_rm["error"], prefix="❌ 更新に失敗しました: ")
                            else:
                                st.success(
                                    f"✅ {', '.join(_rm['old_classes'])} → "
                                    f"{', '.join(_rm['new_classes'])}"
                                    + (f"（{_rm['files_changed']} ファイルを書換、"
                                       f"{_rm['lines_removed']} 行を除去）"
                                       if _rm["files_changed"] else "")
                                    + (f"（{_rm['dirs_merged']} ディレクトリを整理）"
                                       if _rm["dirs_merged"] else "")
                                )
                                st.rerun()
                    else:
                        st.caption("変更はありません。")

            with st.expander(f"⬇ {ds.name} を持ち出す（ZIP エクスポート）"):
                st.caption(
                    "他の PC で学習させる場合などに、データセットを ZIP で書き出します。"
                    "展開すればそのまま YOLO の学習に使える構造のままです。"
                )
                _ex_labels_only = st.checkbox(
                    "ラベルと data.yaml のみ（画像を含めない）", value=False,
                    key=f"ex_lbl_{ds.name}",
                    help="画像は既に相手側にある場合や、アノテーションだけ共有したい場合に使います",
                )
                _ex_bytes = dataset_size_bytes(ds, labels_only=_ex_labels_only)
                _ex_mb = _ex_bytes / 1024 / 1024
                st.caption(f"対象サイズ: 約 {_ex_mb:,.1f} MB（圧縮前）")
                if _ex_mb > 500:
                    st.warning(
                        f"⚠ {_ex_mb:,.0f} MB あります。ZIP の生成とダウンロードに時間がかかり、"
                        "ブラウザ側のメモリも消費します。"
                        "画像が不要なら「ラベルと data.yaml のみ」を使ってください。"
                    )

                if st.button("📦 ZIP を生成", key=f"ex_build_{ds.name}",
                             use_container_width=True):
                    _ex_out = (PREDICTIONS_DIR / "_exports" /
                               f"{ds.name}{'_labels' if _ex_labels_only else ''}.zip")
                    with st.spinner("ZIP を生成中…（サイズによっては数分かかります）"):
                        _ok_ex, _msg_ex, _n_ex = build_dataset_zip(
                            ds, _ex_out, labels_only=_ex_labels_only)
                    st.session_state[f"ex_zip_{ds.name}"] = (
                        {"path": _msg_ex, "n": _n_ex} if _ok_ex else None)
                    if not _ok_ex:
                        st.error(f"❌ 生成に失敗しました: {_msg_ex}")

                _ex_info = st.session_state.get(f"ex_zip_{ds.name}")
                if _ex_info and Path(_ex_info["path"]).exists():
                    _ex_p = Path(_ex_info["path"])
                    st.success(f"✅ {_ex_info['n']} ファイル / "
                               f"{_ex_p.stat().st_size / 1024 / 1024:,.1f} MB")
                    with open(_ex_p, "rb") as _fz:
                        st.download_button(
                            "⬇ ダウンロード", _fz, file_name=_ex_p.name,
                            mime="application/zip", use_container_width=True,
                            key=f"ex_dl_{ds.name}",
                        )
                    st.caption(f"生成先: `{_ex_p}`（不要になったら削除して構いません）")

            with st.expander(f"🔍 {ds.name} の品質チェック"):
                st.caption(
                    "画像とラベルの対応漏れ・座標の破損・クラス分布の偏りを検査します。"
                    "外部から持ち込んだデータや複数人で分担したデータほど確認する価値があります。"
                )
                if st.button("🔍 チェックを実行", key=f"qc_run_{ds.name}",
                             use_container_width=True):
                    with st.spinner(f"{ds.name} を検査中…"):
                        st.session_state[f"qc_{ds.name}"] = check_dataset_quality(ds)

                _qc = st.session_state.get(f"qc_{ds.name}")
                if _qc:
                    if _qc["error"]:
                        st.error(f"❌ {_qc['error']}")
                    else:
                        _n_err = _qc.get("n_errors", 0)
                        if _qc["n_issues"] == 0:
                            st.success("✅ 問題は見つかりませんでした。")
                        elif _n_err > 0:
                            st.error(f"❌ 要対応 {_n_err} 件 / 指摘 {_qc['n_issues']} 件")
                        else:
                            st.warning(f"⚠ 指摘 {_qc['n_issues']} 件（いずれも警告レベル）")

                        # スプリット別の内訳
                        import pandas as _pd_qc
                        _rows_qc = [{
                            "スプリット": sp,
                            "画像": v["images"], "ラベル": v["labels"],
                            "ラベル無し画像": v["missing_label"],
                            "画像無しラベル": v["orphan_label"],
                            "空ラベル": v["empty_label"],
                            "ボックス数": v["boxes"],
                        } for sp, v in _qc["splits"].items()]
                        if _rows_qc:
                            st.dataframe(_pd_qc.DataFrame(_rows_qc),
                                         use_container_width=True, hide_index=True)

                        # クラス分布
                        if _qc["class_counts"]:
                            st.markdown("**クラス分布**")
                            _df_cls = _pd_qc.DataFrame(
                                sorted(_qc["class_counts"].items(),
                                       key=lambda kv: -kv[1]),
                                columns=["クラス", "件数"],
                            )
                            st.dataframe(_df_cls, use_container_width=True, hide_index=True)

                        # 指摘の内訳と詳細
                        if _qc["issue_counts"]:
                            st.markdown("**指摘の内訳**")
                            for _k, _v in sorted(_qc["issue_counts"].items(),
                                                 key=lambda kv: -kv[1]["count"]):
                                _icon = "❌" if _v["severity"] == "error" else "⚠"
                                st.markdown(f"- {_icon} **{_k}**: {_v['count']} 件")
                            # expander の入れ子は不可のためチェックボックスで開閉する
                            if st.checkbox("詳細を表示（種別ごとに最大20件）",
                                           key=f"qc_detail_{ds.name}"):
                                with st.container(border=True):
                                    for _is in _qc["issues"]:
                                        _icon = "❌" if _is["severity"] == "error" else "⚠"
                                        st.caption(f"{_icon} `{_is['path']}` — "
                                                   f"{_is['kind']}: {_is['detail']}")

                            # --- 自動修正 ---
                            _fixable = {"サイズ不正", "座標範囲外", "行フォーマット",
                                        "数値変換", "極小ボックス", "画像無しラベル"}
                            if _fixable & set(_qc["issue_counts"].keys()):
                                st.markdown("**🔧 壊れたラベルの自動修正**")
                                st.caption(
                                    "該当する行だけを取り除きます。書き換える前に "
                                    "`<ファイル名>.txt.bak` としてバックアップを作るので元に戻せます。"
                                )
                                _fx1, _fx2 = st.columns(2)
                                with _fx1:
                                    _fx_size = st.checkbox(
                                        "幅・高さが0以下の行を除去", value=True,
                                        key=f"fx_size_{ds.name}")
                                    _fx_range = st.checkbox(
                                        "座標が0〜1の範囲外の行を除去", value=True,
                                        key=f"fx_range_{ds.name}")
                                with _fx2:
                                    _fx_tiny = st.checkbox(
                                        "極小ボックスも除去", value=False,
                                        key=f"fx_tiny_{ds.name}",
                                        help="小さな物体を意図的にアノテーションしている場合は"
                                             "OFF のままにしてください")
                                    _fx_orphan = st.checkbox(
                                        "画像が無いラベルを退避", value=False,
                                        key=f"fx_orphan_{ds.name}")

                                if st.button("🔧 修正を実行", key=f"fx_run_{ds.name}",
                                             type="primary", use_container_width=True):
                                    with st.spinner("修正中…"):
                                        _fx = fix_dataset_labels(
                                            ds,
                                            drop_invalid_size=_fx_size,
                                            drop_out_of_range=_fx_range,
                                            drop_tiny=_fx_tiny,
                                            delete_orphan_labels=_fx_orphan,
                                        )
                                    if _fx["error"]:
                                        st.error(f"❌ {_fx['error']}")
                                    else:
                                        st.success(
                                            f"✅ {_fx['files_changed']} ファイルを修正し "
                                            f"{_fx['lines_removed']} 行を除去しました"
                                            + (f"／ {_fx['orphans_deleted']} 件の迷子ラベルを退避"
                                               if _fx["orphans_deleted"] else "")
                                        )
                                        if _fx["files_emptied"]:
                                            st.warning(
                                                f"⚠ {_fx['files_emptied']} ファイルが空になりました"
                                                "（その画像は背景画像として扱われます）")
                                        for _d in _fx["details"][:20]:
                                            st.caption(f"・{_d}")
                                        # 修正後の状態で再チェック
                                        st.session_state[f"qc_{ds.name}"] = \
                                            check_dataset_quality(ds)
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

    # --- 外部モデルの取り込み ---
    with st.expander("📤 学習済みモデル (.pt) をアップロード（他PCで学習したモデルの取り込み）"):
        import io as _io_mu

        st.caption(
            "他の環境で学習した YOLO の重みを `models/` に取り込みます。"
            "取り込み時にこの環境の ultralytics で読み込めるか検証し、クラス名を記録します。"
        )
        _mu_run = st.text_input(
            "モデル名（`models/` 以下に作成するディレクトリ名）",
            value=f"imported_{datetime.now():%Y%m%d_%H%M}",
            key="mu_run_name",
        ).strip()
        _mu_mode = st.radio(
            "アップロード形式",
            ["📦 .pt ファイル", "🗜 学習run ディレクトリ ZIP"],
            horizontal=True,
            key="mu_mode",
        )
        _mu_set_current = st.checkbox(
            "取り込み後、このモデルを「使用中」にする", value=True, key="mu_set_current"
        )

        _mu_dir = MODELS_DIR / _mu_run if _mu_run else None
        if _mu_dir and _mu_dir.exists():
            st.warning(f"⚠ `models/{_mu_run}/` は既に存在します。同名ファイルは上書きされます。")

        if _mu_mode == "📦 .pt ファイル":
            _mu_pts = st.file_uploader(
                "重みファイル（.pt、複数選択可）",
                type=["pt"],
                accept_multiple_files=True,
                key="mu_pt_files",
            )
            _mu_extras = st.file_uploader(
                "付随ファイル（任意・複数可）",
                type=["csv", "yaml", "yml", "json", "txt", "png", "jpg", "jpeg"],
                accept_multiple_files=True,
                key="mu_extra_files",
                help="results.csv を一緒に入れると下のカードに mAP50 が表示されます。"
                     "args.yaml / confusion_matrix.png なども保存できます。",
            )
            if _mu_pts and _mu_run:
                st.caption(
                    f"選択中: {len(_mu_pts)} 個の重み "
                    f"({sum(f.size for f in _mu_pts) / 1024 / 1024:.1f} MB)"
                    + (f" + 付随 {len(_mu_extras)} ファイル" if _mu_extras else "")
                )
                st.caption(f"保存先: `models/{_mu_run}/weights/`")
                if st.button("📥 models/ に取り込む", key="mu_pt_btn",
                             type="primary", use_container_width=True):
                    _mu_w = _mu_dir / "weights"
                    _mu_w.mkdir(parents=True, exist_ok=True)
                    _mu_saved = []
                    for _f in _mu_pts:
                        _dst = _mu_w / _f.name
                        _dst.write_bytes(_f.getbuffer())
                        _mu_saved.append(_dst)
                    for _f in (_mu_extras or []):
                        (_mu_dir / _f.name).write_bytes(_f.getbuffer())
                    st.session_state["mu_saved_paths"] = [str(p) for p in _mu_saved]
                    st.session_state["mu_pending_current"] = _mu_set_current
        else:
            _mu_zip = st.file_uploader(
                "学習run ディレクトリの ZIP（`weights/best.pt` を含む想定）",
                type=["zip"],
                key="mu_zip_file",
            )
            if _mu_zip and _mu_run:
                st.caption(f"選択中: {_mu_zip.name}  ({_mu_zip.size / 1024 / 1024:.1f} MB)")
                st.caption(f"展開先: `models/{_mu_run}/`")
                if st.button("📥 展開して models/ に取り込む", key="mu_zip_btn",
                             type="primary", use_container_width=True):
                    with zipfile.ZipFile(_io_mu.BytesIO(_mu_zip.read()), "r") as _zf:
                        _bad = [n for n in _zf.namelist()
                                if n.startswith("/") or ".." in Path(n).parts]
                        if _bad:
                            st.error(f"⚠ ZIP に不正なパスが含まれています: {_bad[:3]}")
                        else:
                            _mu_dir.mkdir(parents=True, exist_ok=True)
                            _zf.extractall(_mu_dir)
                            _mu_found = sorted(_mu_dir.rglob("*.pt"))
                            if not _mu_found:
                                st.error("⚠ ZIP 内に .pt ファイルが見つかりませんでした。")
                            st.session_state["mu_saved_paths"] = [str(p) for p in _mu_found]
                            st.session_state["mu_pending_current"] = _mu_set_current

        # --- 取り込み結果の検証・表示 ---
        _mu_saved_paths = [Path(p) for p in st.session_state.get("mu_saved_paths", [])]
        if _mu_saved_paths:
            st.markdown("---")
            st.markdown("**取り込み結果**")
            _mu_ok_paths = []
            for _p in _mu_saved_paths:
                if not _p.exists():
                    continue
                with st.spinner(f"{_p.name} を検証中…"):
                    _info = inspect_model_file(_p)
                if _info["ok"]:
                    _mu_ok_paths.append(_p)
                    st.success(f"✅ `{_p.relative_to(MODELS_DIR)}` — 読み込み成功")
                    _mi1, _mi2, _mi3 = st.columns(3)
                    _mi1.metric("クラス数", len(_info["names"]))
                    _mi2.metric("タスク", _info["task"] or "—")
                    _mi3.metric("学習時 ultralytics", _info["ultralytics_version"] or "—")
                    st.caption("クラス: " + (", ".join(_info["names"]) or "—"))
                    _mu_detail = [
                        f"ベースモデル: {_info['base_model']}" if _info.get("base_model") else "",
                        f"imgsz: {_info['imgsz']}" if _info.get("imgsz") else "",
                        f"epochs: {_info['epochs']}" if _info.get("epochs") else "",
                        f"学習日: {_info['trained_at']}" if _info.get("trained_at") else "",
                    ]
                    _mu_detail = [d for d in _mu_detail if d]
                    if _mu_detail:
                        st.caption(" / ".join(_mu_detail))
                else:
                    show_error(
                        _info["error"],
                        prefix=f"❌ `{_p.relative_to(MODELS_DIR)}` は読み込めませんでした: ",
                    )
            if _mu_ok_paths and st.session_state.get("mu_pending_current"):
                _mu_best = next((p for p in _mu_ok_paths if p.name == "best.pt"), _mu_ok_paths[0])
                st.session_state.last_model_path = str(_mu_best)
                st.session_state["mu_pending_current"] = False
                st.info(f"⭐ 使用中モデルに設定しました: `{_mu_best.relative_to(MODELS_DIR)}`\n\n"
                        "→ 「🔭 Step4: 推論・評価」タブで推論を実行できます。")
            if st.button("表示をクリア", key="mu_clear_result"):
                st.session_state["mu_saved_paths"] = []
                st.rerun()

    model_files = sorted(
        MODELS_DIR.rglob("*.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    ) if MODELS_DIR.exists() else []
    if not model_files:
        st.info("models/ に .pt ファイルがありません。")
    else:
        import pandas as pd

        # どのモデルが CVAT の自動アノテーションに載っているかを引くための対応表
        # （models/<run> ←→ serverless/custom/<dir> ←→ Nuclio 関数）
        _fn_states = {f["name"]: f for f in cached_nuclio_functions()} if serverless_ready() else {}
        _def_by_run = {d["model_run"]: d for d in list_serverless_defs() if d["model_run"]}

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
                    _meta = read_model_meta(mp)
                    if _meta and _meta.get("ok"):
                        _cls = _meta.get("names") or []
                        _cls_txt = ", ".join(_cls[:6])
                        if len(_cls) > 6:
                            _cls_txt += f" 他{len(_cls) - 6}件"
                        st.caption(f"🏷 {len(_cls)} クラス: {_cls_txt}")
                    elif _meta:
                        st.caption("⚠ この環境の ultralytics で読み込めないモデルです")
                    else:
                        if st.button("🔍 モデル情報を読み込む", key=f"insp_model_{mp}"):
                            with st.spinner(f"{mp.name} を読み込み中…"):
                                inspect_model_file(mp)
                            st.rerun()

                    # 来歴（何で学習したモデルか）
                    _prov = read_provenance(mp.parent.parent) if mp.parent.name == "weights" else None
                    if _prov:
                        _pds = _prov.get("dataset", {}) or {}
                        _pc = _pds.get("counts_at_train", {}) or {}
                        _cnt_txt = " / ".join(f"{k} {v}枚" for k, v in _pc.items())
                        st.caption(
                            f"📚 学習データ: `{_pds.get('name') or '不明'}`"
                            + (f"（{_cnt_txt}）" if _cnt_txt else "")
                            + f"　ベース: `{Path(_prov.get('base_model', '')).name or '不明'}`"
                            + ("　※再開あり" if _prov.get("resumed") else "")
                        )

                    # 評価済みなら最新の mAP を出す（results.csv とは別に、
                    # 任意データセットで測り直した値）
                    _evs = read_model_evals(mp)
                    if _evs:
                        _latest_key = max(_evs, key=lambda k: _evs[k].get("evaluated_at", ""))
                        _lv = _evs[_latest_key]
                        if _lv.get("ok"):
                            st.caption(
                                f"📊 評価 `{_latest_key}` — mAP50 {_lv['map50']:.4f} / "
                                f"mAP50-95 {_lv['map50_95']:.4f}"
                                + (f"（他 {len(_evs) - 1} 件）" if len(_evs) > 1 else "")
                            )

                    # CVAT 自動アノテーションへのデプロイ状態
                    _run_name = (mp.parent.parent.name
                                 if mp.parent.name == "weights" else "")
                    _def = _def_by_run.get(_run_name)
                    _fn_st = _fn_states.get(_def["function_name"]) if _def else None
                    if _fn_st:
                        _bdg = {"ready": "🟢", "error": "🔴"}.get(_fn_st["state"], "🟡")
                        st.caption(f"{_bdg} CVAT自動アノテーションに使用中 "
                                   f"(`{_fn_st['name']}` / {'GPU' if _fn_st['gpu'] else 'CPU'})")
                    elif mp.name == "best.pt":
                        st.caption("○ 自動アノテーション未デプロイ — 「🏷 アノテーション」タブから追加できます")
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
                        model_meta_path(mp).unlink(missing_ok=True)
                        model_eval_path(mp).unlink(missing_ok=True)
                        if st.session_state.last_model_path == str(mp):
                            st.session_state.last_model_path = None
                        st.rerun()

                # --- 持ち出し（他PCへ渡す）---
                _dl1, _dl2 = st.columns(2)
                with _dl1:
                    with open(mp, "rb") as _fm:
                        st.download_button(
                            f"⬇ {mp.name} をダウンロード", _fm, file_name=mp.name,
                            mime="application/octet-stream", use_container_width=True,
                            key=f"dl_pt_{mp}",
                            help="重みファイル単体。相手側の UI でそのまま取り込めます",
                        )
                with _dl2:
                    _bundle_key = f"bundle_{mp}"
                    if st.button("📦 一式ZIPを生成", key=f"mkbundle_{mp}",
                                 use_container_width=True,
                                 help="重み + results.csv + 評価結果 + プロットをまとめます"):
                        _b_out = (PREDICTIONS_DIR / "_exports" /
                                  f"{mp.parent.parent.name}_bundle.zip")
                        with st.spinner("ZIP を生成中…"):
                            _ok_b, _msg_b, _n_b = build_model_bundle_zip(mp, _b_out)
                        st.session_state[_bundle_key] = (
                            {"path": _msg_b, "n": _n_b} if _ok_b else None)
                        if not _ok_b:
                            st.error(f"❌ {_msg_b}")
                        st.rerun()
                    _b_info = st.session_state.get(_bundle_key)
                    if _b_info and Path(_b_info["path"]).exists():
                        _b_p = Path(_b_info["path"])
                        with open(_b_p, "rb") as _fb:
                            st.download_button(
                                f"⬇ 一式ZIP ({_b_p.stat().st_size / 1024 / 1024:.0f}MB)",
                                _fb, file_name=_b_p.name, mime="application/zip",
                                use_container_width=True, key=f"dl_bundle_{mp}",
                            )

    st.markdown("---")

    # --- モデルの系譜（何から何が作られたか）---
    with st.expander("📚 モデルの系譜を追跡する"):
        st.caption(
            "モデルがどのデータセットから作られたかを一覧します。"
            "データを足しながら学習を重ねると対応が分からなくなるため、"
            "学習開始時点の情報を記録しています。"
        )
        _lin_rows = []
        for _run in sorted([p for p in MODELS_DIR.iterdir() if p.is_dir()],
                           key=lambda p: p.stat().st_mtime, reverse=True) \
                if MODELS_DIR.exists() else []:
            _pv = read_provenance(_run)
            if not _pv:
                _lin_rows.append({
                    "モデル": _run.name, "学習日時": "—", "学習データ": "（記録なし）",
                    "枚数": "—", "ベースモデル": "—", "クラス": "—",
                })
                continue
            _d = _pv.get("dataset", {}) or {}
            _cnts = _d.get("counts_at_train", {}) or {}
            _lin_rows.append({
                "モデル": _run.name,
                "学習日時": _pv.get("trained_at", "—"),
                "学習データ": _d.get("name") or "—",
                "枚数": " / ".join(f"{k}:{v}" for k, v in _cnts.items()) or "—",
                "ベースモデル": Path(_pv.get("base_model", "")).name or "—",
                "クラス": ", ".join(_d.get("classes") or []) or "—",
            })

        if _lin_rows:
            import pandas as _pd_lin
            st.dataframe(_pd_lin.DataFrame(_lin_rows),
                         use_container_width=True, hide_index=True)

            # 1件を選んで詳細（データセット側の来歴まで辿る）
            _lin_names = [r["モデル"] for r in _lin_rows]
            _lin_sel = st.selectbox("詳細を見るモデル", _lin_names, key="lineage_sel")
            _lin_pv = read_provenance(MODELS_DIR / _lin_sel)
            if not _lin_pv:
                st.info("このモデルには来歴の記録がありません。"
                        "この機能を入れる前に学習されたモデルです。")
            else:
                _ld = _lin_pv.get("dataset", {}) or {}
                _dsp = _ld.get("provenance") or {}
                st.markdown("**系譜**")
                _chain = []
                if _dsp.get("cvat_tasks"):
                    _chain.append("CVAT タスク " + ", ".join(
                        f"[{t.get('id')}] {t.get('name')}" for t in _dsp["cvat_tasks"]))
                elif _dsp.get("source"):
                    _chain.append({"upload_zip": "外部 ZIP の取込",
                                   "upload_images": "画像の直接アップロード",
                                   "merge": "データセット統合"}.get(_dsp["source"], _dsp["source"]))
                if _ld.get("name"):
                    _chain.append(f"データセット `{_ld['name']}`")
                _chain.append(f"モデル `{_lin_sel}`")
                st.markdown("　→　".join(_chain))

                _lc1, _lc2 = st.columns(2)
                with _lc1:
                    st.markdown("**学習時の情報**")
                    st.caption(f"学習日時: {_lin_pv.get('trained_at', '—')}")
                    st.caption(f"ベースモデル: `{_lin_pv.get('base_model', '—')}`")
                    st.caption(f"再開: {'あり' if _lin_pv.get('resumed') else 'なし'}")
                    _pp = _lin_pv.get("params", {}) or {}
                    st.caption("主なパラメータ: " + ", ".join(
                        f"{k}={_pp[k]}" for k in ("epochs", "batch", "imgsz", "optimizer")
                        if k in _pp) or "—")
                with _lc2:
                    st.markdown("**学習に使ったデータ**")
                    st.caption(f"データセット: `{_ld.get('name') or '—'}`")
                    st.caption(f"クラス: {', '.join(_ld.get('classes') or []) or '—'}")
                    _cnts2 = _ld.get("counts_at_train", {}) or {}
                    st.caption("学習時の枚数: " + (
                        " / ".join(f"{k} {v}枚" for k, v in _cnts2.items()) or "—"))
                    # 現在のデータセットと比べて増減があれば知らせる
                    _cur_ds = Path(_ld.get("data_yaml", "")).parent if _ld.get("data_yaml") else None
                    if _cur_ds and _cur_ds.exists():
                        _now = count_dataset_items(_cur_ds)
                        if _now != _cnts2 and _cnts2:
                            st.warning(
                                "⚠ 学習後にデータセットが変わっています（現在: "
                                + " / ".join(f"{k} {v}枚" for k, v in _now.items())
                                + "）。再学習すると結果が変わります。"
                            )
                    elif _ld.get("data_yaml"):
                        st.warning("⚠ 学習に使ったデータセットは現在見つかりません。")

                with st.expander("生の来歴データ (JSON)"):
                    st.json(_lin_pv)
        else:
            st.info("models/ に学習 run がありません。")

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

    _tp1, _tp2, _tp3, _tp4, _tp5 = st.tabs([
        "🧭 タスクの選び方", "📐 指標の読み方", "🩺 うまくいかないとき",
        "🗂 データセットの作り方", "🛣 今後の方針",
    ])

    # ── タスク種別の選び方 ────────────────────────────────────────────
    with _tp1:
        st.markdown("#### どのタスク種別を選ぶか")
        st.markdown(
            "| やりたいこと | タスク種別 | CVAT で付けるもの | 出力 |\n"
            "|---|---|---|---|\n"
            "| 物体の位置を四角で囲む | `detect` | 矩形 (box) | 位置とクラス |\n"
            "| 物体の形を正確に取る | `segment` | ポリゴン | 輪郭マスク |\n"
            "| 傾いた物体を囲む | `obb` | 回転付き矩形 / 4点ポリゴン | 回転した四角 |\n"
            "| 画像全体を仕分ける | `classify` | タグ | 画像ごとのクラス |\n"
            "| 関節・特徴点を取る | `pose` | ポイント | キーポイント座標 |\n"
        )
        st.info(
            "**迷ったら `detect` から。** アノテーションが最も速く、"
            "必要な情報が足りないと分かってから `segment` に移っても、"
            "矩形は自動でポリゴンに変換できます（逆はできません）。"
        )

        st.markdown("#### モデルサイズの選び方")
        st.markdown(
            "`n` → `s` → `m` → `l` → `x` の順に大きく、精度が上がり、遅く重くなります。"
        )
        st.markdown(
            "- **まず `n` か `s` で一周する** … データやラベルの問題は小さいモデルでも分かります\n"
            "- **精度が頭打ちなら大きくする** … ただし伸びは小さいことが多く、"
            "データを増やす方が効くケースが大半です\n"
            "- **実機に載せるなら速度も測る** … 「🔭 Step4」の評価で推論時間(ms)が出ます\n"
        )
        st.caption(
            "参考: 同一データセットでの実測では、大きいモデルが必ず勝つとは限りません。"
            "mAP50 がほぼ同じでも推論時間が2倍以上違うことがあるので、"
            "評価タブで両方を比べてから決めてください。"
        )

        st.markdown("#### 画像サイズ (imgsz)")
        st.markdown(
            "- 推論時間はおおむね imgsz の**2乗に比例**します（640→1280 で約4倍）\n"
            "- 小さく写る対象が多いなら上げる価値があります\n"
            "- 学習と推論で同じ値を使うのが基本です\n"
        )

    # ── 指標の読み方 ──────────────────────────────────────────────
    with _tp2:
        st.markdown("#### 検出・セグメンテーションの指標")
        st.markdown(
            "- **Precision（適合率）** … 検出したもののうち正しかった割合。"
            "低い = **誤検出が多い**\n"
            "- **Recall（再現率）** … 実際にあるもののうち見つけられた割合。"
            "低い = **見逃しが多い**\n"
            "- **IoU** … 予測と正解の重なり具合。1.0 で完全一致\n"
            "- **mAP50** … IoU 0.5 以上を正解とみなした精度。"
            "「だいたい合っている」かを見る\n"
            "- **mAP50-95** … IoU 0.5〜0.95 で平均した精度。"
            "**位置の正確さまで含めた実力**。実用ではこちらが効く\n"
            "- **top1 / top5 accuracy** … 画像分類の正答率\n"
        )
        st.info(
            "**mAP50 が高いのに mAP50-95 が低い**場合、「物体は見つけられているが"
            "枠の位置が甘い」状態です。アノテーションの枠が雑になっていないか、"
            "imgsz が小さすぎないかを疑ってください。"
        )

        st.markdown("#### Precision と Recall はトレードオフ")
        st.markdown(
            "推論時の `conf`（信頼度しきい値）を上げると Precision が上がり Recall が下がります。"
            "下げるとその逆です。用途で決めてください。"
        )
        st.markdown(
            "- **見逃したくない**（検査・安全）… conf を下げて Recall を優先\n"
            "- **誤検出を出したくない**（自動処理）… conf を上げて Precision を優先\n"
            "- **自動アノテーションの下書き** … 少し低めが便利（消す方が描くより速い）\n"
        )
        st.caption(
            "なお mAP を測るときの conf は 0.001 が正しい値です（全信頼度域の"
            "PR 曲線から計算するため）。実運用のしきい値とは別物です。"
        )

    # ── トラブルシューティング ────────────────────────────────────
    with _tp3:
        st.markdown("#### mAP が上がらない")
        st.markdown(
            "**まずデータを疑ってください。** モデルやパラメータより効きます。\n\n"
            "1. 「📁 データ管理」の **品質チェック**を実行する"
            "（幅0の枠、画像とラベルの対応漏れ、クラス分布の偏りが出ます）\n"
            "2. 「🔭 Step4」の **正解ラベルとの差分分析**で FN が多い画像を見る"
            "— アノテーション漏れが見つかることが多いです\n"
            "3. 学習枚数が足りているか（目安: 1クラスあたり最低 100〜200 枚、"
            "実用なら 1000 枚以上）\n"
            "4. train と val で撮影条件が違いすぎないか\n"
        )

        st.markdown("#### 過学習している（train は良いのに val が悪い）")
        st.markdown(
            "- データを増やす / データ拡張を強める（mosaic, mixup, hsv 系）\n"
            "- モデルを小さくする\n"
            "- `patience` を設定して早期終了させる\n"
            "- エポックを減らす\n"
        )
        st.caption("学習曲線で val の loss が下げ止まって上がり始めたら過学習のサインです。")

        st.markdown("#### 特定のクラスだけ精度が低い")
        st.markdown(
            "- **クラス別 AP** を評価タブで確認（どのクラスが悪いか特定する）\n"
            "- そのクラスの枚数が少なければ追加する（クラス分布の偏りは品質チェックで検出できます）\n"
            "- 似たクラスと混同しているなら、混同行列を確認してクラス定義自体を見直す\n"
        )

        st.markdown("#### 学習が途中で止まってしまった / 止めたい")
        st.markdown(
            "- 学習中の **⏹ 学習を停止** でエポック末に安全に止められます\n"
            "- 止めた学習は **⏯ 中断した学習を再開する** から `last.pt` の続きから再開できます\n"
            "- GPU メモリ不足で落ちる場合は `batch` か `imgsz` を下げてください\n"
        )

        st.markdown("#### 他の PC で学習した .pt が読み込めない")
        st.markdown(
            "学習元の ultralytics のバージョンがこの環境（8.4.48）と離れていると起きます。"
            "「📁 データ管理」からアップロードすると読み込み検証まで行うので、"
            "エラー内容を確認してください。"
        )

    # ── データセットの作り方 ──────────────────────────────────────
    with _tp4:
        st.markdown("#### 枚数の目安")
        st.markdown(
            "| 段階 | 枚数の目安 | 何が分かるか |\n"
            "|---|---|---|\n"
            "| お試し | 50〜100 枚 | パイプラインが通るか |\n"
            "| 最低限 | 1クラス 100〜200 枚 | 実用になるかの当たり |\n"
            "| 実用 | 1クラス 1000 枚以上 | 安定した精度 |\n"
        )

        st.markdown("#### アノテーションの質")
        st.markdown(
            "- **枠は対象にぴったり合わせる** … 甘い枠は mAP50-95 を直接下げます\n"
            "- **基準を統一する** … 隠れている部分を含めるか、どこまでを1つと数えるか。"
            "複数人で作業するなら特に重要です\n"
            "- **迷う対象のルールを決めておく** … 後から直すコストは大きいです\n"
        )
        st.info(
            "**自動アノテーションを活用してください。** 一度モデルを作れば、"
            "「🏷 Step1」から CVAT にデプロイして下書きを自動生成できます。"
            "ゼロから描くより、間違いを直す方が圧倒的に速いです。"
        )

        st.markdown("#### 学習を回す順序")
        st.markdown(
            "1. 少ないデータ・小さいモデル・少ないエポックで**一周させる**\n"
            "2. 品質チェックと差分分析で**データの問題を潰す**\n"
            "3. データを追加する（自動アノテーションで効率化）\n"
            "4. モデルサイズ・エポック・パラメータを調整する\n"
        )
        st.caption("1〜3 を回すのが最も効きます。4 は最後で構いません。")

        st.markdown("#### 途中からデータを足したいとき")
        st.markdown(
            "- 「📁 データ管理」の各データセットから**画像を追加**できます\n"
            "- 複数のデータセットを**統合**することもできます\n"
            "- 既存モデルを初期重みにして**追加学習**できます"
            "（Step3 のモデル名に `models/<run>/weights/best.pt` を指定）\n"
        )

    # ── 今後の方針 ────────────────────────────────────────────────
    with _tp5:
        st.markdown("#### このリポジトリの目的")
        st.markdown(
            "画像系の学習モデルを作るために必要な作業を、"
            "**1つの環境で完結**させることを目指しています。"
            "アノテーション・データ整備・学習・評価・モデル管理を"
            "同じ UI から扱えるようにしています。"
        )
        st.markdown("#### 設計の方針")
        st.markdown(
            "- **どの段階からでもデータを入れられる** … CVAT 経由でも、ZIP でも、"
            "画像単体でも、他 PC で作った `.pt` でも受け入れる\n"
            "- **持ち出せる** … データセットもモデルも ZIP で書き出せる\n"
            "- **壊れたデータを検出して直せる** … 品質チェックと自動修正\n"
            "- **判断材料を UI 内に出す** … 同一条件での mAP 比較、推論速度、"
            "正解ラベルとの差分\n"
        )
        st.markdown("#### 実装予定・検討中")
        st.markdown(
            "- ハイパーパラメータ探索 / k-fold 交差検証\n"
            "- train/val の再分割、クラス名の編集・統合\n"
            "- 学習に使ったデータの来歴を記録する仕組み\n"
            "- MLflow の実験比較を UI 内に埋め込む\n"
            "- `app/main.py` の分割（機能追加を続けやすくするため）\n"
        )
        st.markdown(
            "**[→ docs/overview.md をGitHubで開く]"
            "(https://github.com/ryotaema/detection_dev_ui/blob/main/docs/overview.md)**"
        )
        st.caption(
            "実装済みの機能・コード構成・設計方針・実装上の落とし穴をまとめています。"
            "新しく参加する人はまずこれを読んでください。"
        )
        st.caption(
            "※ 今後の実装予定と既知の不具合は、この環境の `docs/roadmap.md` にあります"
            "（開発方針のため Git 管理外。`SPEC.md` / `CLAUDE.md` と同じ扱い）。"
        )

    st.markdown("---")

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