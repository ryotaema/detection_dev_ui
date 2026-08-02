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
    "Step4(推論・評価)":     "推論 & FiftyOne 可視化",
    "データ管理":            "学習済みモデル",
    "トピックス":            "どのタスク種別を選ぶか",
}

# 押すと重い処理が走るもの（学習・推論・評価・デプロイ・ZIP生成・破壊的操作）
HEAVY_KEYS = {
    "train_start", "infer_run", "ev_run", "gd_run", "sw_run", "ap_run",
    "nt_create", "push_run", "gd_push", "gd_fo", "dep_run_btn", "aa_run",
    "mu_pt_btn", "mu_zip_btn", "cvat_export_run", "dataset_generate_run",
    "xml_parse_run", "cvat_fetch_tasks", "anno_fetch_tasks", "fo_launch",
    "reanno_zip", "exp_images_run", "merge_datasets_run", "pred_clear_all",
    "resume_btn", "lc_pick",
}
HEAVY_PREFIXES = (
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
