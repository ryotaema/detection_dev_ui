# =============================================================================
# 拡張機能のタブ
#
#   extensions/ に clone された別リポジトリのツールを 1 つ 1 つタブにする。
#   拡張が壊れていても本体を巻き込まないよう、描画は必ず握りつぶす。
# =============================================================================
from __future__ import annotations

from pathlib import Path

import streamlit as st

from core import *  # noqa: F401,F403
from core.extensions import (
    PLACEHOLDERS, load_streamlit_action, resolve_command, run_extension_command,
)
from .widgets import empty_state, show_error


def render_extension_placeholder() -> None:
    """拡張が 1 つも入っていないときの案内タブ"""
    st.markdown('<div class="section-head"><h3>🧩 拡張機能</h3></div>',
                unsafe_allow_html=True)
    st.caption(
        "別のリポジトリで作った道具を `extensions/` に clone すると、"
        "ここに専用のタブが増えます。本体を大きくせずに機能を足していくための仕組みです。"
    )
    empty_state(
        "まだ拡張が入っていません",
        "リポジトリ直下の `extensions/` に git clone して、ブラウザを再読み込みしてください。",
    )
    st.markdown("**入れ方**")
    st.code(
        "cd extensions\n"
        "git clone https://github.com/ryotaema/anno_dataset_tools\n"
        "git clone https://github.com/ryotaema/mosaic_tool\n"
        "# ブラウザを再読み込みするとタブが増えます",
        language="bash",
    )
    st.caption("詳しい書き方は `extensions/README.md` にあります。")


def _render_desktop_action(ext: dict, action: dict) -> None:
    """デスクトップ GUI。コンテナには画面が無いのでホストで動かしてもらう。"""
    st.info(
        "🖥 これはデスクトップ画面（Tkinter など）の道具です。"
        "このアプリはコンテナの中で動いていて画面を持たないため、"
        "**ホスト側のターミナルで実行**してください。"
    )
    host_dir = f"extensions/{ext['dir_name']}"
    st.code(f"cd {host_dir}\n" + " ".join(action["command"]), language="bash")
    st.caption(
        "`data/` `models/` `predictions/` はホストのディレクトリをそのまま"
        "マウントしているので、ホストで開いても同じファイルを触れます。"
    )
    if action.get("note"):
        st.caption(f"ℹ {action['note']}")


def _render_command_action(ext: dict, action: dict, key: str) -> None:
    """CLI。実行する内容を先に見せてから、押されたときだけ動かす。"""
    values: dict = {}
    if action.get("inputs"):
        cols = st.columns(min(len(action["inputs"]), 3))
        for i, name in enumerate(action["inputs"]):
            with cols[i % len(cols)]:
                values[name] = st.text_input(name, key=f"{key}_in_{name}")

    resolved = resolve_command(action["command"], Path(ext["dir"]), values)
    st.caption("実行される内容:")
    st.code(" ".join(resolved), language="bash")
    if action.get("note"):
        st.caption(f"ℹ {action['note']}")

    if st.button("▶ 実行", key=f"{key}_run", type="primary"):
        with st.spinner("実行中…"):
            res = run_extension_command(action["command"], Path(ext["dir"]), values)
        st.session_state[f"{key}_result"] = res

    res = st.session_state.get(f"{key}_result")
    if res:
        if res["ok"]:
            st.success("✅ 実行できました")
        else:
            show_error(res["error"], prefix="❌ 実行に失敗しました: ")
        if res["stdout"]:
            st.caption("標準出力")
            st.code(res["stdout"][-8000:], language="text")
        if res["stderr"]:
            st.caption("標準エラー出力")
            st.code(res["stderr"][-4000:], language="text")


def _render_streamlit_action(ext: dict, action: dict) -> None:
    """拡張が用意した render() をこの場で描く。ここで初めて相手のコードが動く。"""
    fn, err = load_streamlit_action(
        Path(ext["dir"]), action["module"], action["function"])
    if fn is None:
        show_error(err, prefix="❌ 読み込めませんでした: ")
        st.caption(
            "この拡張を Streamlit に組み込むには、"
            f"`{action['module']}.py` に `def {action['function']}():` を用意して、"
            "その中で `st.*` を呼んでください。"
        )
        return
    try:
        fn()
    except Exception as e:
        # 拡張の中で落ちても本体は巻き込まない
        show_error(f"{type(e).__name__}: {e}", prefix="❌ 拡張の描画中にエラー: ")
        with st.expander("詳細"):
            import traceback
            st.code(traceback.format_exc(), language="text")


def render_extension(ext: dict) -> None:
    """拡張 1 つぶんのタブの中身"""
    st.markdown(
        f'<div class="section-head"><h3>{ext["icon"]} {ext["name"]}</h3></div>',
        unsafe_allow_html=True,
    )
    if ext.get("description"):
        st.caption(ext["description"])

    # --- 素性 ---
    _meta = [f"📁 `extensions/{ext['dir_name']}`"]
    if ext.get("revision"):
        _meta.append(f"版 `{ext['revision']}`")
    _meta.append(f"設定: {ext['manifest_source']}")
    st.caption("　".join(_meta))
    if ext.get("url"):
        st.caption(f"🔗 {ext['url']}")

    if ext.get("inferred"):
        st.info(
            "ℹ この拡張にはマニフェストが無いため、ファイル構成から推測して表示しています。"
            f"`extensions/{ext['dir_name']}/extension.json` を置くと、"
            "操作名や引数を思いどおりに出せます（書き方は `extensions/README.md`）。"
        )
    for w in ext.get("warnings", []):
        st.warning(f"⚠ マニフェスト: {w}")

    if ext.get("missing"):
        st.warning(
            "⚠ このコンテナに入っていない依存があります: "
            + ", ".join(f"`{m}`" for m in ext["missing"])
            + "\n\n`app/requirements.txt` に足して "
            "`docker compose build streamlit_app` すると使えるようになります。"
            "ホストで動かすぶんには不要です。"
        )

    if not ext.get("actions"):
        st.info("この拡張には実行できる操作が登録されていません。")
        return

    st.markdown("---")
    labels = [a["label"] for a in ext["actions"]]
    if len(labels) > 1:
        chosen = st.radio(
            "操作", labels, key=f"ext_act_{ext['dir_name']}",
            label_visibility="collapsed",
        )
        action = ext["actions"][labels.index(chosen)]
    else:
        action = ext["actions"][0]
        st.markdown(f"**{action['label']}**")

    key = f"ext_{ext['dir_name']}_{labels.index(action['label'])}"
    if action["kind"] == "desktop":
        _render_desktop_action(ext, action)
    elif action["kind"] == "command":
        _render_command_action(ext, action, key)
    else:
        _render_streamlit_action(ext, action)

    with st.expander("使えるプレースホルダ"):
        st.caption("マニフェストの command に書くと、実行時に実際のパスへ置き換わります。")
        for ph, desc in PLACEHOLDERS.items():
            st.markdown(f"- `{ph}` … {desc}")
