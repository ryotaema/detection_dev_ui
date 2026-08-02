# =============================================================================
# 画面ウィジェットの共通ヘルパー
#
#   説明つきのスライダー・数値入力など、複数のタブで使う部品を置く。
#   ここは見た目に関わる処理だけ（ロジックは core/ 側）。
# =============================================================================
from __future__ import annotations

import streamlit as st

from core import explain_error
from core import _DOC_AUG, _DOC_TRAIN  # ウィジェットの既定リンク先


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


def show_error(message: str, prefix: str = "") -> None:
    """エラーを表示し、よくある原因に当てはまれば対処も添える"""
    st.error(f"{prefix}{message}" if prefix else str(message))
    hint = explain_error(str(message))
    if hint:
        st.warning(f"**{hint['title']}**\n\n{hint['hint']}")


def empty_state(what: str, next_step: str, hint: str = "") -> None:
    """まだ何も無いときの表示。

    「無い」で終わらせず、次にどのタブで何をすればよいかまで書く。
    初めて触る人がここで手が止まらないようにするのが目的。

    what      … 何が無いのか
    next_step … 次にどこで何をするか（👉 付きで出る）
    hint      … 代替手段など、あれば
    """
    st.info(f"**{what}**\n\n👉 {next_step}" + (f"\n\n{hint}" if hint else ""))


# ---------------------------------------------------------------------------
# パイプライン状態ヘルパー
# ---------------------------------------------------------------------------

