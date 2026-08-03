# =============================================================================
# 拡張機能のテスト
#
#   いちばん大事なのは「一覧を作るだけの段階で相手のコードを実行しない」こと。
#   マニフェストが壊れていても本体が落ちないことも合わせて確かめる。
# =============================================================================
from __future__ import annotations

import json

import pytest

from core import extensions as ex


@pytest.fixture
def ext_root(tmp_path, monkeypatch):
    """extensions/ を差し替えて隔離する"""
    root = tmp_path / "extensions"
    root.mkdir()
    monkeypatch.setattr(ex, "EXTENSIONS_DIR", root)
    monkeypatch.setattr(ex, "PRESET_DIR", tmp_path / "presets")
    (tmp_path / "presets").mkdir()
    return root


def _make_ext(root, name, manifest=None, files=None):
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    if manifest is not None:
        (d / "extension.json").write_text(
            json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    for path, body in (files or {}).items():
        f = d / path
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(body, encoding="utf-8")
    return d


# ---------------------------------------------------------------------------
# 探索
# ---------------------------------------------------------------------------
def test_拡張が無ければ空(ext_root):
    assert ex.discover_extensions() == []


def test_マニフェストを読んで一覧にする(ext_root):
    _make_ext(ext_root, "tool_a", {
        "name": "道具A", "icon": "🔧", "description": "せつめい",
        "actions": [{"label": "実行", "kind": "command", "command": ["echo", "hi"]}],
    })
    got = ex.discover_extensions()
    assert len(got) == 1
    assert got[0]["name"] == "道具A"
    assert got[0]["icon"] == "🔧"
    assert got[0]["actions"][0]["command"] == ["echo", "hi"]
    assert got[0]["manifest_source"] == "リポジトリ内の extension.json"


def test_探索では拡張のコードを実行しない(ext_root):
    """import されたら分かるように、実行時にファイルを作るモジュールを置く"""
    marker = ext_root.parent / "実行された.txt"
    _make_ext(ext_root, "danger", {
        "name": "危険", "actions": [
            {"label": "描画", "kind": "streamlit", "module": "extension"}]},
        files={"extension.py":
               f"from pathlib import Path\n"
               f"Path({str(marker)!r}).write_text('x')\n"
               f"def render(): pass\n"})

    ex.discover_extensions()
    assert not marker.exists(), "一覧を作るだけで拡張のコードが実行されている"


def test_既定マニフェストが使われる(ext_root):
    (ex.PRESET_DIR / "known.json").write_text(
        json.dumps({"name": "既知の道具", "icon": "📦", "actions": []}),
        encoding="utf-8")
    _make_ext(ext_root, "known")
    got = ex.discover_extensions()[0]
    assert got["name"] == "既知の道具"
    assert got["manifest_source"] == "同梱の既定マニフェスト"


def test_リポジトリ内のマニフェストが既定より優先される(ext_root):
    (ex.PRESET_DIR / "known.json").write_text(
        json.dumps({"name": "既定", "actions": []}), encoding="utf-8")
    _make_ext(ext_root, "known", {"name": "同梱", "actions": []})
    assert ex.discover_extensions()[0]["name"] == "同梱"


def test_マニフェストが無ければ構成から推測する(ext_root):
    _make_ext(ext_root, "guess", files={
        "extension.py": "def render(): pass\n",
        "my_gui.py": "import tkinter\n",
        "scripts/conv.py": "print(1)\n",
    })
    got = ex.discover_extensions()[0]
    assert got["inferred"] is True
    kinds = {a["kind"] for a in got["actions"]}
    assert kinds == {"streamlit", "desktop", "command"}


def test_壊れたJSONでも落ちない(ext_root):
    d = _make_ext(ext_root, "broken")
    (d / "extension.json").write_text("{ これは JSON ではない", encoding="utf-8")
    got = ex.discover_extensions()
    assert len(got) == 1               # 推測にフォールバックする
    assert got[0]["inferred"] is True


def test_変な名前のディレクトリは無視する(ext_root):
    (ext_root / ".git").mkdir()
    (ext_root / "__pycache__").mkdir()
    _make_ext(ext_root, "ok_tool", {"name": "OK", "actions": []})
    assert [e["dir_name"] for e in ex.discover_extensions()] == ["ok_tool"]


# ---------------------------------------------------------------------------
# マニフェストの検証
# ---------------------------------------------------------------------------
def test_未対応のkindは落として警告する(ext_root):
    _make_ext(ext_root, "t", {"name": "t", "actions": [
        {"label": "変", "kind": "なにか", "command": ["echo"]},
        {"label": "正", "kind": "command", "command": ["echo"]},
    ]})
    got = ex.discover_extensions()[0]
    assert [a["label"] for a in got["actions"]] == ["正"]
    assert any("未対応の kind" in w for w in got["warnings"])


def test_commandが文字列リストでなければ落とす(ext_root):
    _make_ext(ext_root, "t", {"name": "t", "actions": [
        {"label": "文字列", "kind": "command", "command": "echo hi"},
        {"label": "空", "kind": "command", "command": []},
    ]})
    got = ex.discover_extensions()[0]
    assert got["actions"] == []
    assert len(got["warnings"]) == 2


def test_streamlitはmodule必須(ext_root):
    _make_ext(ext_root, "t", {"name": "t", "actions": [
        {"label": "module無し", "kind": "streamlit"}]})
    got = ex.discover_extensions()[0]
    assert got["actions"] == []
    assert any("module" in w for w in got["warnings"])


# ---------------------------------------------------------------------------
# プレースホルダと実行
# ---------------------------------------------------------------------------
def test_プレースホルダが置き換わる(tmp_path):
    from core.config import DATA_DIR
    out = ex.resolve_command(
        ["python3", "s.py", "--src", "{data_dir}/ds", "--name", "{who}"],
        tmp_path, {"who": "太郎"})
    assert out == ["python3", "s.py", "--src", f"{DATA_DIR}/ds", "--name", "太郎"]


def test_ext_dirが置き換わる(tmp_path):
    assert ex.resolve_command(["ls", "{ext_dir}"], tmp_path)[1] == str(tmp_path)


def test_コマンドを実行して出力を返す(tmp_path):
    r = ex.run_extension_command(["python3", "-c", "print('こんにちは')"], tmp_path)
    assert r["ok"] and "こんにちは" in r["stdout"]


def test_失敗した終了コードを拾う(tmp_path):
    r = ex.run_extension_command(["python3", "-c", "import sys; sys.exit(3)"], tmp_path)
    assert not r["ok"] and r["returncode"] == 3


def test_存在しないコマンドでも例外にしない(tmp_path):
    r = ex.run_extension_command(["この_コマンドは_無い"], tmp_path)
    assert not r["ok"] and "見つかりません" in r["error"]


def test_打ち切り時間を守る(tmp_path):
    r = ex.run_extension_command(
        ["python3", "-c", "import time; time.sleep(5)"], tmp_path, timeout=1)
    assert not r["ok"] and "打ち切り" in r["error"]


def test_拡張のディレクトリで実行される(tmp_path):
    (tmp_path / "しるし.txt").write_text("x")
    r = ex.run_extension_command(
        ["python3", "-c", "import os; print(os.listdir('.'))"], tmp_path)
    assert "しるし.txt" in r["stdout"]


# ---------------------------------------------------------------------------
# Streamlit 拡張の読み込み
# ---------------------------------------------------------------------------
def test_render関数を読み込める(tmp_path):
    (tmp_path / "extension.py").write_text(
        "def render():\n    return '描画した'\n", encoding="utf-8")
    fn, err = ex.load_streamlit_action(tmp_path, "extension", "render")
    assert err == "" and fn() == "描画した"


def test_ファイルが無ければ理由を返す(tmp_path):
    fn, err = ex.load_streamlit_action(tmp_path, "ない", "render")
    assert fn is None and "見つかりません" in err


def test_関数が無ければ理由を返す(tmp_path):
    (tmp_path / "extension.py").write_text("x = 1\n", encoding="utf-8")
    fn, err = ex.load_streamlit_action(tmp_path, "extension", "render")
    assert fn is None and "render()" in err


def test_読み込み時に例外が出ても落ちない(tmp_path):
    (tmp_path / "extension.py").write_text(
        "raise RuntimeError('壊れている')\n", encoding="utf-8")
    fn, err = ex.load_streamlit_action(tmp_path, "extension", "render")
    assert fn is None and "壊れている" in err


def test_足りない依存を検出する():
    assert ex.missing_requirements(["まず存在しないパッケージ名"]) == \
        ["まず存在しないパッケージ名"]
    assert ex.missing_requirements(["numpy"]) == []
    # パッケージ名と import 名が違うもの
    assert ex.missing_requirements(["opencv-python", "Pillow", "pyyaml"]) == []


# ---------------------------------------------------------------------------
# マニフェストをディレクトリにまとめる形
# ---------------------------------------------------------------------------
def test_extensionディレクトリのマニフェストを読む(ext_root):
    _make_ext(ext_root, "t", files={
        "extension/extension.json": json.dumps(
            {"name": "まとめた", "actions": []}, ensure_ascii=False)})
    got = ex.discover_extensions()[0]
    assert got["name"] == "まとめた"
    assert got["manifest_source"].endswith("extension/extension.json")
    assert got["base_dir"].endswith("/extension")
    assert got["has_own_manifest"] is True


def test_隠しディレクトリのマニフェストも読む(ext_root):
    _make_ext(ext_root, "t", files={
        ".dev_ui/extension.json": json.dumps({"name": "隠し", "actions": []},
                                             ensure_ascii=False)})
    assert ex.discover_extensions()[0]["name"] == "隠し"


def test_ディレクトリ形が直下の単体ファイルより優先される(ext_root):
    _make_ext(ext_root, "t", {"name": "直下", "actions": []}, files={
        "extension/extension.json": json.dumps({"name": "ディレクトリ", "actions": []},
                                               ensure_ascii=False)})
    assert ex.discover_extensions()[0]["name"] == "ディレクトリ"


def test_既定マニフェストは自前扱いにしない(ext_root):
    (ex.PRESET_DIR / "known.json").write_text(
        json.dumps({"name": "既定", "actions": []}), encoding="utf-8")
    _make_ext(ext_root, "known")
    assert ex.discover_extensions()[0]["has_own_manifest"] is False


# ---------------------------------------------------------------------------
# 雛形の書き出し
# ---------------------------------------------------------------------------
def test_雛形を書き出せる(ext_root):
    d = _make_ext(ext_root, "t", files={"my_gui.py": "import tkinter\n"})
    ext = ex.discover_extensions()[0]
    res = ex.scaffold_manifest(d, ext)
    assert res["ok"], res["error"]

    written = json.loads((d / "extension" / "extension.json").read_text(encoding="utf-8"))
    assert written["name"] == "t"
    assert written["actions"][0]["kind"] == "desktop"

    # 書き出したものが次回そのまま読まれる
    again = ex.discover_extensions()[0]
    assert again["has_own_manifest"] is True
    assert again["inferred"] is False


def test_雛形は既存を上書きしない(ext_root):
    d = _make_ext(ext_root, "t", files={
        "extension/extension.json": json.dumps({"name": "既存", "actions": []},
                                               ensure_ascii=False)})
    res = ex.scaffold_manifest(d, {"name": "新しい", "actions": []})
    assert not res["ok"] and "すでにあります" in res["error"]
    assert json.loads((d / "extension" / "extension.json").read_text())["name"] == "既存"
