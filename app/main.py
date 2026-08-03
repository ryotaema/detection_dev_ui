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

# 配色の定義は ui/theme.py に集約している
from ui.theme import (DEFAULT_THEME_NAME, PRESET_THEMES, THEME_EDIT_FIELDS,
                      active_theme, build_theme_vars, load_user_themes,
                      save_user_themes)

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

/* セクション見出しの帯。
   以前は .pipeline-card で「中身を囲む枠」にするつもりだったが、
   st.markdown は呼び出しごとに独立した DOM ブロックになるため
   開きタグと閉じタグを別々に出しても囲めない（空の枠だけが描画される）。
   囲むのは諦めて、セクションの始まりを示す帯として使う。 */
.section-head {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-left: 4px solid var(--accent);
    border-radius: 6px;
    padding: 12px 18px;
    margin: 14px 0 16px;
}
.section-head h3 {
    color: var(--accent);
    margin: 0;
    font-size: 1.05rem;
}

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

/* ── Streamlit 標準ウィジェットの色合わせ ──────────────────────────────────
   ここから下は Streamlit が生成する DOM 構造に依存する。
   構造が変わってもセレクタが当たらなくなるだけで壊れはしないが、
   バージョンを上げたときは見た目を確認すること（現在 1.35.0）。 */

/* タブ。Step4 の内部タブが増えたので、選択中がはっきり分かるようにする */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    border-bottom: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    color: var(--text-secondary);
    border-radius: 6px 6px 0 0;
    padding: 6px 14px;
}
.stTabs [data-baseweb="tab"]:hover { color: var(--accent); }
.stTabs [aria-selected="true"] {
    color: var(--accent);
    background: var(--bg-card-inner);
    font-weight: 600;
}
.stTabs [data-baseweb="tab-highlight"] { background: var(--accent); }

/* expander の見出し */
.streamlit-expanderHeader,
[data-testid="stExpander"] summary {
    color: var(--text-primary);
    font-size: .88rem;
}
[data-testid="stExpander"] summary:hover { color: var(--accent); }
[data-testid="stExpander"] details {
    border: 1px solid var(--border);
    border-radius: 6px;
    background: var(--bg-card);
}

/* 表 */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: 6px;
}

/* ── 狭い画面での折り返し ────────────────────────────────────────────────
   layout="wide" 固定で、st.columns() は既定では折り返さない。
   幅が足りないときだけ縦に積ませる。
   ここで扱うのは入力を含むカラムで、表示だけの指標は metric_row() を使う。 */
@media (max-width: 900px) {
    [data-testid="stHorizontalBlock"] {
        flex-wrap: wrap;
    }
    [data-testid="stHorizontalBlock"] > [data-testid="column"] {
        flex: 1 1 240px;
        min-width: 240px;
    }
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

/* 指標の並び（幅に応じて折り返す）
   st.columns(5) は画面が狭くても 5 等分のままで、
   「FN（取りこぼし）」のような長いラベルが潰れる。
   flex-wrap なら幅が足りないぶんだけ段が増える。 */
.metric-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin: 8px 0 12px;
}
.metric-grid .mg-item {
    background: var(--bg-card-inner);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 6px 12px;
    flex: 1 1 116px;
    min-width: 116px;
}
.metric-grid .mg-label {
    color: var(--text-muted);
    font-size: .7rem;
    letter-spacing: .03em;
    margin-bottom: 2px;
}
.metric-grid .mg-value {
    color: var(--text-primary);
    font-size: .95rem;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
    word-break: break-all;
}

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
    "theme_name": DEFAULT_THEME_NAME,
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

# テーマ変数を注入（デフォルトCSS変数を上書き）
st.markdown(build_theme_vars(active_theme()), unsafe_allow_html=True)


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
    # よく使うものほど上に置く。設定・参照情報は下の折りたたみへ回す
    st.markdown("---")
    st.markdown("#### 🔗 クイックリンク")
    st.markdown(f"[📝 CVAT UI]({CVAT_WEB})", unsafe_allow_html=False)
    st.markdown(f"[📊 MLflow UI]({MLFLOW_WEB})", unsafe_allow_html=False)
    if st.session_state.fiftyone_port:
        fo_url = f"http://localhost:{st.session_state.fiftyone_port}"
        st.markdown(f"[🔭 FiftyOne App]({fo_url})", unsafe_allow_html=False)

    st.markdown("---")
    st.markdown("#### 🖥 サービス状態")

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
    # 毎回見るものではないので畳んでおく
    with st.expander("📁 ディレクトリ", expanded=False):
        st.code(
            f"data/        {DATA_DIR}\n"
            f"models/      {MODELS_DIR}\n"
            f"predictions/ {PREDICTIONS_DIR}",
            language="text",
        )

    # ── テーマ設定（サイドバー最下部） ──
    with st.expander("🎨 テーマ設定", expanded=False):
        _user_themes_db = load_user_themes()
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
                    save_user_themes(_user_themes_db)
                    st.session_state.theme_name = _new_name.strip()
                    st.success(f"保存: {_new_name.strip()}")
                    st.rerun()
                else:
                    st.warning("テーマ名を入力してください")
        with _sc2:
            if _cur_theme in _user_themes_db:
                if st.button("🗑 削除", key="del_theme_btn"):
                    del _user_themes_db[_cur_theme]
                    save_user_themes(_user_themes_db)
                    st.session_state.theme_name = DEFAULT_THEME_NAME
                    st.rerun()

        _t = active_theme()
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
#
#   表示/非表示でトップレベルの要素数が変わると、直後にある st.tabs の
#   識別がずれて画面が真っ白になることがある。
#   常にコンテナを1つ置き、その中身だけを出し入れして要素数を一定に保つ。
# ---------------------------------------------------------------------------
_onboarding_slot = st.container()
with _onboarding_slot:
    if st.session_state.get("show_onboarding"):
        with st.container(border=True):
            st.markdown("### 📖 はじめかた")

            # メインのタブ群の直前に別のタブを置くと表示が壊れることがあるため、
            # ここはラジオで切り替える
            _ob_page = st.radio(
                "表示する内容",
                ["① まず何をするか", "② 別のPCで環境を作る", "③ 用語とタスクの選び方"],
                horizontal=True, key="ob_page", label_visibility="collapsed",
            )

            # ── ① 現在の状態と次にやること ──────────────────────────────────
            if _ob_page.startswith("①"):
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
            if _ob_page.startswith("②"):
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
            if _ob_page.startswith("③"):
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
# ---------------------------------------------------------------------------
from ui.presets import (_apply_preset, _collect_current_params,
                        _load_user_presets, _save_user_presets)
from ui.widgets import _ckw, _nw, _ph, _selw, _sw, show_error
from ui.tab_annotate import render_annotate
from ui.tab_ingest import render_ingest
from ui.tab_train import render_train
from ui.tab_evaluate import render_evaluate
from ui.tab_manage import render_manage
from ui.tab_topics import render_topics

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
    render_annotate()

with tab1:
    render_ingest()

with tab2:
    render_train()

with tab3:
    render_evaluate()

with tab4:
    render_manage()

with tab5:
    render_topics()


# ---------------------------------------------------------------------------
# 進捗ポーリング
#
#   学習・評価・デプロイの実行中は定期的に再実行して進捗を更新する。
#   タブの描画途中で st.rerun() を呼ぶと、それ以降のタブが描画されず
#   画面が欠けるため、すべて描き終えたここで一度だけ行う。
# ---------------------------------------------------------------------------
_poll_interval = consume_rerun_poll()
if _poll_interval:
    time.sleep(_poll_interval)
    st.rerun()
