# =============================================================================
# バックグラウンド処理の共有状態（rerun をまたいで保持する）
#
#   **`st.cache_resource` は使わない。**
#   キャッシュの取り出しは ScriptRunContext を必要とし、無い場合は
#   `raise CacheKeyNotFoundError()` で必ずミスする（Streamlit 1.35 の
#   `cache_resource_api.py`）。つまり `threading.Thread` で回すワーカーから
#   呼ぶと**毎回まっさらな dict が返り**、画面はその更新を一生見られない
#   （進捗が 0% のまま動かない）。
#
#   このモジュールは import されるだけで、スクリプトのように再実行されない。
#   モジュール変数はそのまま rerun をまたいで生き残るので、
#   素のシングルトンで足りるし、そのほうがスレッドから確実に同じ実体になる。
# =============================================================================
from __future__ import annotations

import threading

import streamlit as st   # 進捗ポーリング（下の方）で使う


_EVAL_SHARED = (
    {"log": [], "running": False, "error": None, "finished": False,
    "total": 0, "done": 0, "current": "", "results": []},
    threading.Lock(),
)


def _get_eval_shared() -> tuple[dict, threading.Lock]:
    """モデル評価のバックグラウンド実行状態"""
    return _EVAL_SHARED


_DEPLOY_SHARED = (
    {"log": [], "running": False, "error": None, "target": None, "finished": False},
    threading.Lock(),
)


def _get_deploy_shared() -> tuple[dict, threading.Lock]:
    """Nuclio デプロイのバックグラウンド実行状態（学習と同じく rerun をまたいで保持する）"""
    return _DEPLOY_SHARED


_TRAIN_SHARED = (
    {"log": [], "progress": 0, "running": False, "error": None, "model_path": None,
    "metrics_history": [], "stop_requested": False},
    threading.Lock(),
)


def _get_train_shared() -> tuple[dict, threading.Lock]:
    """st.rerun() をまたいで同一オブジェクトを保持する共有状態。
    Streamlit はスクリプトを再実行するたびにモジュール変数を再初期化するため、
    st.cache_resource でキャッシュして常に同一インスタンスを返す。
    """
    return _TRAIN_SHARED


_TUNE_SHARED = (
    {"log": [], "running": False, "error": None,
    "iteration": 0, "total": 0, "best_fitness": None,
    "best_params": None, "tune_dir": None, "history": [],
    "stop_requested": False, "started_at": None,
    # いま回しているイテレーションの中身。
    # 1 回が数分〜数十分かかるので、これが無いと「止まって見える」
    "current_params": None, "current_epoch": 0, "current_total_epochs": 0,
    "current_metrics": None, "iter_started_at": None},
    threading.Lock(),
)


def _get_tune_shared() -> tuple[dict, threading.Lock]:
    """ハイパーパラメータ探索の共有状態。学習と同じ作法。

    探索は 1 回のイテレーションが「学習まるごと 1 回」なので、
    数時間かかることもある。タブを離れても続くようスレッドで回す。
    """
    return _TUNE_SHARED


# ---------------------------------------------------------------------------
# 進捗ポーリングの予約
#
#   バックグラウンド処理（学習・評価・デプロイ）の進捗を追うには
#   定期的な再実行が要る。ただしタブの描画途中で st.rerun() を呼ぶと
#   スクリプトがそこで打ち切られ、**それ以降のタブが描画されない**。
#   （データ管理やトピックスが真っ白になる、他の入力が操作できない）
#
#   そこで「予約」だけしておき、全タブを描画し終えた main.py の末尾で
#   まとめて再実行する。
# ---------------------------------------------------------------------------
POLL_KEY = "_poll_rerun_after"


def request_rerun_poll(interval: float = 2.0) -> None:
    """描画が終わったあとに再実行するよう予約する（タブの中から呼ぶ）"""
    prev = st.session_state.get(POLL_KEY)
    # 複数の処理が同時に動いていたら短い方に合わせる
    st.session_state[POLL_KEY] = interval if prev is None else min(prev, interval)


def consume_rerun_poll() -> float | None:
    """予約されていた再実行間隔を取り出す（main.py の末尾から呼ぶ）"""
    return st.session_state.pop(POLL_KEY, None)
