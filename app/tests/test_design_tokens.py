# =============================================================================
# tools/apply_design_tokens.py のテスト
#
#   theme.py（ソース）を書き換える処理なので、
#   壊れた入力を弾けること・往復して値が変わらないことを確かめる。
# =============================================================================
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent


def _load(name: str):
    """tools/ のスクリプトをモジュールとして読み込む（パッケージではないため）"""
    path = ROOT / "tools" / f"{name}.py"
    if not path.exists():
        pytest.skip(f"{path} がありません")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def apply_mod():
    return _load("apply_design_tokens")


@pytest.fixture(scope="module")
def build_mod():
    return _load("build_design_system")


@pytest.fixture(scope="module")
def colors_html(build_mod):
    """実際の theme.py から colors.html 相当を組み立てる"""
    themes = build_mod.load_themes()
    app_css = build_mod.load_app_css()
    return build_mod.page(
        build_mod.card("Colors", "カラートークン", "26変数 × 4テーマ"),
        "カラートークン", "テスト用", themes, app_css, build_mod.body_tokens(themes),
    ), themes


# ---------------------------------------------------------------------------
# 読み取り
# ---------------------------------------------------------------------------
def test_生成した_html_から元のテーマを復元できる(apply_mod, colors_html):
    html, themes = colors_html
    parsed = apply_mod.parse_colors_html(html)

    assert set(parsed) == set(themes), "テーマの顔ぶれが一致しない"
    for name, colors in themes.items():
        for key, value in colors.items():
            assert parsed[name][key] == value.lower(), f"{name}.{key} が復元できていない"


def test_26個の変数がすべて取れる(apply_mod, colors_html):
    html, _ = colors_html
    for colors in apply_mod.parse_colors_html(html).values():
        assert len(colors) == 26


def test_目印が無ければエラーになる(apply_mod):
    with pytest.raises(apply_mod.ParseError, match="data-theme-name"):
        apply_mod.parse_colors_html("<html><body>関係のないページ</body></html>")


def test_変数が欠けていればエラーになる(apply_mod):
    html = ('<div data-theme-index="0" data-theme-name="ダーク"></div>'
            '<style>.t0 { --bg-app: #000000; --accent: #ffffff; }</style>')
    with pytest.raises(apply_mod.ParseError, match="足りない変数"):
        apply_mod.parse_colors_html(html)


def test_16進表記でない色はエラーになる(apply_mod, colors_html):
    html, _ = colors_html
    broken = html.replace("--accent: #2d6bb8;", "--accent: rgb(45,107,184);")
    with pytest.raises(apply_mod.ParseError, match="16 進表記"):
        apply_mod.parse_colors_html(broken)


def test_知らない変数は無視される(apply_mod, colors_html):
    """将来 CSS 変数が増えても、既知の 26 個さえ揃っていれば読める"""
    html, themes = colors_html
    extended = html.replace(".t0 { ", ".t0 { --brand-new-token: #123456; ")
    parsed = apply_mod.parse_colors_html(extended)
    assert len(parsed[list(themes)[0]]) == 26


# ---------------------------------------------------------------------------
# 書き出し
# ---------------------------------------------------------------------------
def test_書き出した定義を読み直すと同じ辞書になる(apply_mod, build_mod):
    themes = build_mod.load_themes()
    src = apply_mod.THEME_PY.read_text(encoding="utf-8")
    start, end, _ = apply_mod.find_preset_themes(src)

    lines = src.split("\n")
    new_src = "\n".join(
        lines[:start] + apply_mod.render_preset_themes(themes) + lines[end + 1:])

    _, _, roundtrip = apply_mod.find_preset_themes(new_src)
    assert roundtrip == themes


def test_色を変えたときだけ差し替わる(apply_mod, build_mod):
    themes = build_mod.load_themes()
    first = list(themes)[0]
    edited = {n: dict(c) for n, c in themes.items()}
    edited[first]["accent"] = "#ff00ff"

    src = apply_mod.THEME_PY.read_text(encoding="utf-8")
    start, end, _ = apply_mod.find_preset_themes(src)
    lines = src.split("\n")
    new_src = "\n".join(
        lines[:start] + apply_mod.render_preset_themes(edited) + lines[end + 1:])

    _, _, roundtrip = apply_mod.find_preset_themes(new_src)
    assert roundtrip[first]["accent"] == "#ff00ff"
    # 他のテーマは元のまま
    for other in list(themes)[1:]:
        assert roundtrip[other] == themes[other]
    # theme.py の他の部分は壊れていない
    assert "def build_theme_vars" in new_src
    assert "def active_theme" in new_src


def test_書き出しても他の定義を巻き込まない(apply_mod, build_mod):
    """PRESET_THEMES の直後にある DEFAULT_THEME_NAME を消さないこと"""
    themes = build_mod.load_themes()
    src = apply_mod.THEME_PY.read_text(encoding="utf-8")
    start, end, _ = apply_mod.find_preset_themes(src)
    lines = src.split("\n")
    new_src = "\n".join(
        lines[:start] + apply_mod.render_preset_themes(themes) + lines[end + 1:])

    assert "DEFAULT_THEME_NAME" in new_src
    assert "THEME_EDIT_FIELDS" in new_src
    compile(new_src, "theme.py", "exec")   # 構文として妥当か
