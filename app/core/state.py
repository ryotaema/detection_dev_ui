# =============================================================================
# バックグラウンド処理の共有状態（rerun をまたいで保持する）
# =============================================================================
from __future__ import annotations

import threading

import streamlit as st


@st.cache_resource
def _get_eval_shared() -> tuple[dict, threading.Lock]:
    """モデル評価のバックグラウンド実行状態"""
    return (
        {"log": [], "running": False, "error": None, "finished": False,
         "total": 0, "done": 0, "current": "", "results": []},
        threading.Lock(),
    )


@st.cache_resource
def _get_deploy_shared() -> tuple[dict, threading.Lock]:
    """Nuclio デプロイのバックグラウンド実行状態（学習と同じく rerun をまたいで保持する）"""
    return (
        {"log": [], "running": False, "error": None, "target": None, "finished": False},
        threading.Lock(),
    )


@st.cache_resource
def _get_train_shared() -> tuple[dict, threading.Lock]:
    """st.rerun() をまたいで同一オブジェクトを保持する共有状態。
    Streamlit はスクリプトを再実行するたびにモジュール変数を再初期化するため、
    st.cache_resource でキャッシュして常に同一インスタンスを返す。
    """
    return (
        {"log": [], "progress": 0, "running": False, "error": None, "model_path": None,
         "metrics_history": [], "stop_requested": False},
        threading.Lock(),
    )
