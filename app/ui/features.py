# =============================================================================
# オプション機能の表示切り替え
#
#   全員が必要とするわけではない機能を、使う人だけ出せるようにする。
#   画面に出しっぱなしにすると、操作の選択肢が増え続けて
#   「自分に関係のあるものがどれか」が読み取れなくなるため。
#
#   既定はすべてオフ。clone した直後は素の状態から始まり、
#   必要になった人がサイドバーで足していく。
#
#   kind:
#     inline … 既存の画面の中に操作が増える（例: データ管理のデータセット操作）
#     tab   … 専用のタブが増える
# =============================================================================
from __future__ import annotations

import json

import streamlit as st

from core import MODELS_DIR

FEATURES_PATH = MODELS_DIR / ".user_features.json"

OPTIONAL_FEATURES: dict[str, dict] = {
    "mosaic": {
        "label": "🟦 モザイク",
        "kind": "inline",
        "desc": "写り込みを隠す（背景データに入った対象物・顔・プライバシー）",
        "where": "「📁 データ管理」のデータセット操作に追加されます",
    },
    "crop": {
        "label": "✂️ クロップ生成",
        "kind": "tab",
        "desc": "BBOX の検出結果で切り出し、2 段階目（セグメンテーション）用の"
                "アノテーション素材を作ります",
        "where": "専用のタブが増えます",
    },
}

# タブとして増える機能の並び順（この順にタブが後ろへ足される）
TAB_FEATURE_ORDER = [k for k, v in OPTIONAL_FEATURES.items() if v["kind"] == "tab"]


def load_enabled() -> set[str]:
    """有効にしている機能の名前を返す。壊れていれば空として扱う。"""
    if not FEATURES_PATH.exists():
        return set()
    try:
        data = json.loads(FEATURES_PATH.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return {str(x) for x in data if str(x) in OPTIONAL_FEATURES}
    except Exception:
        pass
    return set()


def save_enabled(names) -> bool:
    """有効にする機能を保存する。知らない名前は捨てる。"""
    keep = sorted({str(n) for n in names if str(n) in OPTIONAL_FEATURES})
    try:
        FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
        FEATURES_PATH.write_text(
            json.dumps(keep, ensure_ascii=False, indent=2), encoding="utf-8")
        return True
    except Exception:
        return False


def feature_enabled(name: str) -> bool:
    """その機能を出すかどうか。

    毎回ファイルを読まずに済むよう、セッションの間は覚えておく。
    設定を変えたときは `refresh_features()` で読み直す。
    """
    if "_enabled_features" not in st.session_state:
        st.session_state["_enabled_features"] = load_enabled()
    return name in st.session_state["_enabled_features"]


def refresh_features() -> None:
    st.session_state["_enabled_features"] = load_enabled()


def enabled_tab_features() -> list[str]:
    """タブとして出す機能を、決めた順で返す"""
    return [k for k in TAB_FEATURE_ORDER if feature_enabled(k)]


def render_feature_settings() -> None:
    """サイドバーに置く設定。チェックを変えたらその場で保存する。

    ここは**サイドバー＝タブより前**に描かれるので、
    切り替えた結果はこの回の描画からそのまま反映される。`st.rerun()` は呼ばない
    （描画の途中で呼ぶと、そこで打ち切られて以降の要素が描かれず、
      ウィジェットの対応がずれる）。
    """
    st.caption(
        "使う人だけ出せるようにしています。"
        "必要になったらここで足してください（設定は保存されます）。"
    )
    current = load_enabled()

    # チェックボックスの初期値は session_state に入れておき、
    # ウィジェットには value= を渡さない。
    # key と value を両方渡すと、どちらが正なのかが状況で変わって取り違えが起きる。
    for name in OPTIONAL_FEATURES:
        k = f"feat_{name}"
        if k not in st.session_state:
            st.session_state[k] = name in current

    picked: set[str] = set()
    for name, info in OPTIONAL_FEATURES.items():
        if st.checkbox(info["label"], key=f"feat_{name}",
                       help=f"{info['desc']}\n\n{info['where']}"):
            picked.add(name)
        st.caption(f"　{info['where']}")

    if picked != current and not save_enabled(picked):
        st.error(f"設定を保存できませんでした（{FEATURES_PATH}）")
    # この回の描画からすぐ効かせる（再実行しないため）
    st.session_state["_enabled_features"] = picked
