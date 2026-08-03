# =============================================================================
# テーマ定義と CSS 変数の組み立て
#
#   画面の配色はすべてここに集約する。各タブは色を直書きせず
#   CSS 変数（var(--accent) など）を使うこと。
#
#   例外は components.html() で描画する iframe の中で、
#   iframe は親ドキュメントの CSS 変数を継承しない。
#   そこだけは active_theme() から実際の色を取り出して埋め込む。
# =============================================================================
from __future__ import annotations

import json

import streamlit as st

from core import MODELS_DIR

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

DEFAULT_THEME_NAME = "ライト シンプル"

# OS の配色設定に合わせるモード。
#   Streamlit（サーバ側）からはブラウザの prefers-color-scheme を読めないので、
#   両方の配色を CSS に書き出しておき、どちらを使うかはブラウザに決めさせる。
AUTO_THEME_NAME = "🌗 OS に合わせる"
AUTO_LIGHT_BASE = "ライト シンプル"
AUTO_DARK_BASE  = "ダーク（デフォルト）"

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

# CSS 変数名 ⇔ テーマ辞書のキー
_VAR_MAP: list[tuple[str, str]] = [
    ("--bg-app",         "bg_app"),
    ("--bg-sidebar",     "bg_sidebar"),
    ("--bg-card",        "bg_card"),
    ("--bg-card-inner",  "bg_card_inner"),
    ("--bg-log",         "bg_log"),
    ("--border",         "border"),
    ("--border-accent",  "border_accent"),
    ("--text-primary",   "text_primary"),
    ("--text-secondary", "text_secondary"),
    ("--text-muted",     "text_muted"),
    ("--accent",         "accent"),
    ("--accent-dark",    "accent_dark"),
    ("--success",        "success"),
    ("--success-bg",     "success_bg"),
    ("--success-border", "success_border"),
    ("--warning",        "warning"),
    ("--warning-bg",     "warning_bg"),
    ("--warning-border", "warning_border"),
    ("--error",          "error"),
    ("--error-bg",       "error_bg"),
    ("--error-border",   "error_border"),
    ("--btn-bg",         "btn_bg"),
    ("--btn-hover",      "btn_hover"),
    ("--chip-bg",        "chip_bg"),
    ("--chip-border",    "chip_border"),
    ("--chip-text",      "chip_text"),
]


def load_user_themes() -> dict:
    if USER_THEMES_PATH.exists():
        try:
            return json.loads(USER_THEMES_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_user_themes(themes: dict) -> None:
    USER_THEMES_PATH.write_text(
        json.dumps(themes, ensure_ascii=False, indent=2), encoding="utf-8")


def is_auto_theme() -> bool:
    return st.session_state.get("theme_name", DEFAULT_THEME_NAME) == AUTO_THEME_NAME


def active_theme() -> dict:
    """現在選択中のテーマの色辞書を返す。

    「OS に合わせる」のときは実際にどちらが使われるか**サーバ側では分からない**。
    この関数の戻り値は iframe（学習ログ）に色を焼き込む用途にしか使っていないので、
    その場合はライト側を返す。iframe の中だけは OS 設定に追従しない。
    """
    name = st.session_state.get("theme_name", DEFAULT_THEME_NAME)
    if name == AUTO_THEME_NAME:
        return PRESET_THEMES[AUTO_LIGHT_BASE]
    if name in PRESET_THEMES:
        return PRESET_THEMES[name]
    return load_user_themes().get(name, PRESET_THEMES[DEFAULT_THEME_NAME])


def _var_block(t: dict, indent: str = "  ") -> str:
    return "\n".join(f"{indent}{var}: {t[key]};" for var, key in _VAR_MAP)


def build_theme_vars(t: dict) -> str:
    """テーマ辞書から :root の CSS 変数定義を組み立てる。"""
    return f"""<style>
:root {{
{_var_block(t)}
}}
.stApp {{ background: {t['bg_app']}; }}
[data-testid="stSidebar"] {{
  background: {t['bg_sidebar']};
  border-right: 1px solid {t['border']};
}}
</style>"""


def build_auto_theme_vars() -> str:
    """OS の配色設定に追従する CSS を組み立てる。

    ライトを既定として書き、`prefers-color-scheme: dark` のときだけ
    ダークで上書きする。切り替えはブラウザが行うので再実行は要らない。
    """
    light = PRESET_THEMES[AUTO_LIGHT_BASE]
    dark  = PRESET_THEMES[AUTO_DARK_BASE]
    return f"""<style>
:root {{
{_var_block(light)}
}}
.stApp {{ background: {light['bg_app']}; }}
[data-testid="stSidebar"] {{
  background: {light['bg_sidebar']};
  border-right: 1px solid {light['border']};
}}

@media (prefers-color-scheme: dark) {{
  :root {{
{_var_block(dark, indent="    ")}
  }}
  .stApp {{ background: {dark['bg_app']}; }}
  [data-testid="stSidebar"] {{
    background: {dark['bg_sidebar']};
    border-right: 1px solid {dark['border']};
  }}
}}
</style>"""
