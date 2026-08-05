# =============================================================================
# 画面ウィジェットの共通ヘルパー
#
#   説明つきのスライダー・数値入力など、複数のタブで使う部品を置く。
#   ここは見た目に関わる処理だけ（ロジックは core/ 側）。
# =============================================================================
from __future__ import annotations

import html

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


def metric_row(items: list[tuple[str, object]]) -> None:
    """指標を「幅に応じて折り返す」形で並べる。

    `st.columns(5)` は画面が狭くても 5 等分のままなので、
    ラベルの長い指標（「FN（取りこぼし）」など）が潰れて読めなくなる。
    ここでは flex-wrap の HTML にして、入りきらない分は次の段に送る。

    値を編集させたい場合はウィジェットが必要なので使えない。
    表示するだけの指標に使うこと。
    """
    cells = "".join(
        f'<div class="mg-item">'
        f'<div class="mg-label">{html.escape(str(label))}</div>'
        f'<div class="mg-value">{html.escape(str(value))}</div>'
        f'</div>'
        for label, value in items
    )
    st.markdown(f'<div class="metric-grid">{cells}</div>', unsafe_allow_html=True)


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



def open_folder(container_path, key: str, label: str = "📂 フォルダを開く",
                inline: bool = False) -> None:
    """そのフォルダを OS のファイルアプリで開く導線を出す。

    この UI はコンテナの中で動いていて画面を持たないので、
    自分でファイルアプリを起動することはできない。そこで:
      - ホスト側のパスを必ず表示する（コピーすればどこでも使える）
      - `tools/open_folder_watcher.sh` が動いていればボタンで実際に開く

    inline=True にすると、ボタンだけを出して説明は押したときに見せる
    （カードの中など、場所が限られるところ向け）。
    """
    from core.hostpath import (
        host_path_available, open_command, request_open, to_host_path,
        watcher_running,
    )

    host = to_host_path(container_path)
    if host is None:
        if not inline:
            st.caption(f"📁 `{container_path}`（コンテナ内のパス）")
        return

    if st.button(label, key=f"openfd_{key}", use_container_width=inline,
                 help=f"{host}\n\nホスト側のパスです"):
        res = request_open(container_path)
        if res["ok"]:
            st.success("✅ ファイルアプリで開きました")
        else:
            # ウォッチャーが動いていなくても詰まらないよう、必ず手段を示す
            st.info(
                f"ℹ {res['error']}。下のパスをコピーして開いてください。\n\n"
                "常時ワンクリックで開きたい場合は、ホスト側で "
                "`./tools/open_folder_watcher.sh` を動かしておいてください。"
            )
        st.session_state[f"openfd_show_{key}"] = True

    if st.session_state.get(f"openfd_show_{key}") or not inline:
        st.code(host, language="text")
        if st.session_state.get(f"openfd_show_{key}"):
            st.caption("端末から開く場合:")
            st.code(open_command(container_path), language="bash")


def folder_watcher_status() -> None:
    """「📂 開く」がワンクリックで効く状態か、サイドバー等に出す。

    動いていないときに何をすればよいかまで示す
    （動かなくてもパス表示は使えるので、必須ではないことも伝える）。
    """
    from core.hostpath import host_path_available, watcher_running

    if not host_path_available():
        st.caption("📂 ホスト側のパスを特定できません（フォルダを開く機能は使えません）")
        return

    if watcher_running():
        st.caption("📂 フォルダを開く: ✅ 使えます")
        return

    st.caption("📂 フォルダを開く: パス表示のみ")
    with st.popover("ワンクリックで開くには"):
        st.markdown(
            "この UI はコンテナの中で動いていて画面を持たないため、"
            "自分でファイルアプリを起動できません。\n\n"
            "ホスト側で次を一度だけ実行すると、"
            "以降はログインのたびに自動で動きます。"
        )
        st.code("./tools/install_folder_watcher.sh", language="bash")
        st.caption("その場だけ動かす場合:")
        st.code("./tools/open_folder_watcher.sh", language="bash")
        st.caption("解除: `./tools/install_folder_watcher.sh --uninstall`")
