"""UI の通し検査。

「例外は出ないのに画面が壊れる」不具合を見つけるためのもの。
過去に踏んだ例:
  - タブの描画途中で st.rerun() → それ以降のタブが描画されない
  - 条件でトップレベルの要素が増減 → st.tabs の識別がずれて真っ白
  - st.expander の入れ子 → 例外

**重い処理を起動するボタンは押さない。** 押すと学習・推論・デプロイが実際に走る。
除外は key で行うので、実行系のウィジェットには必ず key を付けること。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

APP = str(Path(__file__).resolve().parents[1])
if APP not in sys.path:
    sys.path.insert(0, APP)

MAIN = f"{APP}/main.py"

# 各タブが描画されたことを示す文字列（1つでも欠けたら描画が途切れている）
TAB_MARKERS = {
    "Step1(アノテーション)": "自動アノテーションモデル",
    "Step2(データ取込)":     "CVATタスクエクスポート",
    "Step3(モデル学習)":     "学習設定サマリー",
    "Step4(推論・評価)":     "推論して結果を見る",
    "データ管理":            "学習済みモデル",
    "トピックス":            "どのタスク種別を選ぶか",
}

# 拡張タブは extensions/ に clone されているかで有無が変わるので、
# 「あれば描けていること」だけを見る（無くても失敗にしない）


# 押すと重い処理が走るもの（学習・推論・評価・デプロイ・ZIP生成・破壊的操作）
HEAVY_KEYS = {
    "train_start", "infer_run", "ev_run", "gd_run", "sw_run", "ap_run",
    "nt_create", "push_run", "gd_push", "gd_fo", "dep_run_btn", "aa_run",
    "tune_start", "tune_stop",   # 探索は学習を何度も回す
    "mu_pt_btn", "mu_zip_btn", "cvat_export_run", "dataset_generate_run",
    "xml_parse_run", "cvat_fetch_tasks", "anno_fetch_tasks", "fo_launch",
    "reanno_zip", "exp_images_run", "merge_datasets_run", "pred_clear_all",
    "resume_btn", "lc_pick",
}
HEAVY_PREFIXES = (
    # mz_ はモザイク。検出（推論が走る）・適用・復元はいずれも押させない
    "mz_",
    # 再アノテーションのフラグ。以前は session_state だけだったが、
    # データセットの .review_state.json に残るようになったので、
    # 検査で押すと実データの精査記録を汚してしまう
    "prev_flag_", "sel_flag_", "pv_zoom_flag",
    "rs_run_", "fx_run_", "cls_run_", "ex_build_", "qc_run_", "redeploy_",
    "delfn_", "del_model_", "del_ds_", "mkbundle_", "insp_model_", "use_model_",
    "dl_", "add_btn_", "ul_zip_btn", "ul_imgs_btn",
)


def _is_heavy(key: str | None) -> bool:
    if not key:
        return True                      # key が無いものは安全側に倒して押さない
    return key in HEAVY_KEYS or key.startswith(HEAVY_PREFIXES)


def _text(at) -> str:
    return " ".join([m.value for m in at.markdown]
                    + [c.value for c in at.caption]
                    + [i.value for i in at.info]
                    + [s.value for s in at.success])


def _assert_all_tabs_rendered(at, context: str) -> None:
    assert not at.exception, f"{context}: 例外 {at.exception[0].value[:300]}"
    text = _text(at)
    missing = [name for name, marker in TAB_MARKERS.items() if marker not in text]
    assert not missing, f"{context}: {missing} が描画されていない"


@pytest.fixture(scope="module")
def app():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(MAIN, default_timeout=300)
    at.run()
    return at


def test_all_tabs_render(app):
    _assert_all_tabs_rendered(app, "初期表示")


def test_no_exception_on_start(app):
    assert len(app.exception) == 0


def test_every_widget_has_key():
    """実行系ウィジェットに key があること。

    key が無いと、この検査で「押してはいけないボタン」を識別できず、
    学習や推論が誤って走ってしまう。
    """
    import ast

    offenders = []
    for f in sorted((Path(APP) / "ui").glob("*.py")):
        if f.name == "widgets.py":       # 呼び出し側の key を透過するので対象外
            continue
        for node in ast.walk(ast.parse(f.read_text())):
            if (isinstance(node, ast.Call)
                    and getattr(node.func, "attr", "") in {"button", "download_button"}
                    and not any(kw.arg == "key" for kw in node.keywords)):
                offenders.append(f"{f.name}:{node.lineno}")
    assert not offenders, f"key の無いボタン: {offenders}"


def test_no_rerun_inside_tabs():
    """タブの中でポーリング目的の rerun をしていないこと。

    タブの描画途中で st.rerun() を呼ぶと、それ以降のタブが描画されない。
    進捗の追従は request_rerun_poll() で予約し、main.py の末尾で実行する。
    """
    offenders = []
    for f in sorted((Path(APP) / "ui").glob("*.py")):
        for i, line in enumerate(f.read_text().splitlines(), 1):
            if "time.sleep" in line and not line.strip().startswith("#"):
                offenders.append(f"{f.name}:{i}")
    assert not offenders, f"タブ内に sleep がある（ポーリングの疑い）: {offenders}"


def test_no_nested_expander():
    """expander の入れ子は例外になる"""
    import ast

    offenders = []
    for f in sorted((Path(APP) / "ui").glob("*.py")) + [Path(MAIN)]:
        class Visitor(ast.NodeVisitor):
            def __init__(self):
                self.depth = 0

            def visit_With(self, node):
                is_expander = any(
                    isinstance(it.context_expr, ast.Call)
                    and getattr(it.context_expr.func, "attr", "") == "expander"
                    for it in node.items)
                if is_expander:
                    self.depth += 1
                    if self.depth > 1:
                        offenders.append(f"{f.name}:{node.lineno}")
                self.generic_visit(node)
                if is_expander:
                    self.depth -= 1

        Visitor().visit(ast.parse(f.read_text()))
    assert not offenders, f"expander の入れ子: {offenders}"


def test_toggling_checkboxes_keeps_all_tabs(app):
    """隠れている経路（詳細表示など）を開いても描画が壊れないこと"""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(MAIN, default_timeout=300)
    at.run()
    toggled = set()
    for _ in range(40):
        target = next((cb for cb in at.checkbox
                       if cb.key and cb.key not in toggled and not cb.value), None)
        if target is None:
            break
        toggled.add(target.key)
        target.check().run()
        _assert_all_tabs_rendered(at, f"チェックON: {target.key}")
    assert toggled, "チェックボックスが1つも見つからない"


def test_safe_buttons_keep_all_tabs(app):
    """重い処理を伴わないボタンを押しても描画が壊れないこと"""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(MAIN, default_timeout=300)
    at.run()
    pressed = []
    for _ in range(20):
        target = next((b for b in at.button
                       if b.key and b.key not in pressed
                       and not _is_heavy(b.key)
                       # 無効化されたボタンは実ブラウザでは押せない。
                       # AppTest は押せてしまうので、ここで除外する
                       and not getattr(b, "disabled", False)), None)
        if target is None:
            break
        pressed.append(target.key)
        target.click().run()
        _assert_all_tabs_rendered(at, f"ボタン: {target.key}")


def test_onboarding_toggle_does_not_break_tabs(app):
    """はじめかたガイドの表示/非表示でタブがずれないこと（過去に真っ白になった）"""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(MAIN, default_timeout=300)
    at.run()
    for expected in (True, False, True, False):
        cb = [c for c in at.checkbox if c.key == "show_onboarding"][0]
        (cb.check() if expected else cb.uncheck()).run()
        _assert_all_tabs_rendered(at, f"ガイド={'ON' if expected else 'OFF'}")


def test_preview_zoom_panel_opens_and_closes():
    """推論プレビューの拡大パネルが開き、前後に動け、閉じられること。

    フラグ付けの主要導線なので、描画が壊れていないかまで見る。
    """
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(MAIN, default_timeout=300)
    at.run()

    def _open(a):
        return any(b.key == "pv_zoom_close" for b in a.button)

    zoom_buttons = [b for b in at.button if b.key and b.key.startswith("prev_zoom_")]
    if not zoom_buttons:
        pytest.skip("推論結果が無いので拡大パネルを開けない")

    assert not _open(at), "最初からパネルが開いている"

    zoom_buttons[0].click().run()
    _assert_all_tabs_rendered(at, "拡大パネルを開いた")
    assert _open(at), "拡大パネルが開かない"

    nxt = [b for b in at.button if b.key == "pv_zoom_next"]
    if nxt and not getattr(nxt[0], "disabled", False):
        nxt[0].click().run()
        _assert_all_tabs_rendered(at, "次の画像へ移動")
        assert _open(at), "移動したらパネルが閉じてしまった"

    [b for b in at.button if b.key == "pv_zoom_close"][0].click().run()
    _assert_all_tabs_rendered(at, "拡大パネルを閉じた")
    assert not _open(at), "パネルが閉じない"


# ---------------------------------------------------------------------------
# プリセット適用（_MODEL_OPTS の import 漏れで全滅していた経路）
# ---------------------------------------------------------------------------
def test_プリセットを適用しても落ちず値が入る():
    """組み込みプリセットはすべて model を持つので、
    _MODEL_OPTS が引けないと全プリセットが NameError になる。"""
    from streamlit.testing.v1 import AppTest

    from core import _MODEL_OPTS

    at = AppTest.from_file(MAIN, default_timeout=300).run()
    sel = [s for s in at.selectbox if s.key == "preset_sel"]
    assert sel, "プリセット選択が見つからない"

    targets = [o for o in sel[0].options if "速度優先" in o or "精度優先" in o]
    assert targets, "組み込みプリセットが見つからない"

    for opt in targets:
        at = AppTest.from_file(MAIN, default_timeout=300).run()
        s = [x for x in at.selectbox if x.key == "preset_sel"][0]
        at = s.select(opt).run()
        btn = [b for b in at.button if b.key == "preset_apply"]
        assert btn, "適用ボタンが見つからない"
        at = btn[0].click().run()
        assert not at.exception, f"{opt}: {at.exception[0].value if at.exception else ''}"
        # 既定値のままではなく、プリセットの値が入っていること
        assert at.session_state["tp_model"] in _MODEL_OPTS
        assert at.session_state["tp_epochs"] > 0


# ---------------------------------------------------------------------------
# 探索中の表示
#
#   1 イテレーション = 学習 1 回なので、切れ目でしか更新しないと
#   数分〜数十分なにも動かず「固まった」ように見える。
#   いまどこで・何を試していて・どうだったかが出ていること。
# ---------------------------------------------------------------------------
def test_探索中は現在の設定と進捗が出る():
    import time

    from streamlit.testing.v1 import AppTest

    import ui.tab_train as tt
    from core.state import _get_tune_shared

    state, lock = _get_tune_shared()
    with lock:
        _backup = dict(state)
        state.update({
            "running": True, "total": 15, "iteration": 3, "best_fitness": 0.4821,
            "started_at": time.time() - 900, "iter_started_at": time.time() - 120,
            "current_params": {"lr0": 0.00432, "momentum": 0.8912},
            "current_epoch": 7, "current_total_epochs": 20,
            "current_metrics": {"mAP50(B)": 0.41},
            "history": [{"iteration": i, "fitness": 0.3 + i * 0.05,
                         "lr0": 0.01 - i * 0.001} for i in (1, 2, 3)],
            "log": ["[10:01:00] 1 / 15 回目を開始"],
            "tune_dir": "/workspace/models/.tuning/demo",
        })

    # 実行中は再描画を予約するので、検査では止めておく（無限に走るため）
    _orig_poll = tt.request_rerun_poll
    tt.request_rerun_poll = lambda *a, **k: None
    try:
        at = AppTest.from_file(MAIN, default_timeout=300).run()
        assert not at.exception, at.exception[0].value if at.exception else ""

        _prog = [str(getattr(p, "text", "") or "") for p in at.get("progress")]
        assert any("4 / 15 回目" in t for t in _prog), f"全体の進捗が出ていない: {_prog}"
        assert any("7 / 20 エポック" in t for t in _prog), \
            f"いま回している学習の進捗が出ていない: {_prog}"

        _exp = [str(getattr(e, "label", "") or "") for e in at.get("expander")]
        assert any("いま試している設定" in t for t in _exp), _exp

        # 既定からどれだけ振れているかが添えてあること
        _m = {m.label: m for m in at.metric}
        assert "lr0" in _m and _m["lr0"].delta, "既定との差が出ていない"

        assert any(b.key == "tune_stop" for b in at.button), "停止ボタンがない"
    finally:
        tt.request_rerun_poll = _orig_poll
        with lock:
            state.clear()
            state.update(_backup)


def test_探索の項目を自分で選べる():
    """既定のプリセットは 4 項目だが、26 項目から自由に選べること。
    1 回が学習まるごと 1 回なので、増やしすぎには警告を出す。"""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(MAIN, default_timeout=300).run()
    sel = [s for s in at.selectbox if s.key == "tune_preset"][0]
    at = sel.select("custom").run()
    assert not at.exception, at.exception[0].value if at.exception else ""

    ms = [m for m in at.multiselect if m.key == "tune_custom_keys"]
    assert ms, "項目を選ぶ欄がない"
    assert len(ms[0].options) >= 20, f"選べる項目が少なすぎる: {len(ms[0].options)}"

    at = ms[0].select("box").select("cls").select("degrees").run()
    assert not at.exception, at.exception[0].value if at.exception else ""
    assert any("項目は多めです" in str(w.value) for w in at.warning), \
        "項目を増やしすぎたときの注意が出ていない"


def test_探索は学習タブの条件を引き継ぐ():
    """探索だけ imgsz や optimizer が違うと、見つけた値が本番で再現しない。"""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(MAIN, default_timeout=300).run()
    _cap = " ".join(str(c.value) for c in at.caption)
    assert "③ 詳細設定と同じ条件" in _cap, "条件を引き継ぐ旨の表示がない"
