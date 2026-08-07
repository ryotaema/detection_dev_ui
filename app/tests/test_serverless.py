"""Nuclio 関数定義の生成（CVAT 自動アノテーション）"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

import core.serverless as sl


@pytest.fixture
def fn_dir(tmp_path: Path, monkeypatch):
    """SERVERLESS_DIR を一時ディレクトリに差し替える"""
    monkeypatch.setattr(sl, "SERVERLESS_DIR", tmp_path)
    return tmp_path


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def test_generates_cpu_and_gpu_definitions(fn_dir: Path):
    out, name = sl.generate_function_files("my-model", "run1", ["a", "b"])
    assert name == "custom-my-model"
    assert sorted(p.name for p in out.iterdir()) == [
        "function-gpu.yaml", "function.yaml", "model.env"]


def test_labels_come_from_model_classes(fn_dir: Path):
    """ラベル名をモデルのクラス名から作ることで CVAT との不一致を防ぐ"""
    out, _ = sl.generate_function_files("m", "run1", ["bell_pepper", "peduncle"])
    spec = _load(out / "function.yaml")["metadata"]["annotations"]["spec"]
    assert [d["name"] for d in json.loads(spec)] == ["bell_pepper", "peduncle"]


def test_segment_model_declares_polygon(fn_dir: Path):
    out, _ = sl.generate_function_files("m", "run1", ["a"], task="segment")
    spec = json.loads(_load(out / "function.yaml")["metadata"]["annotations"]["spec"])
    assert spec[0]["type"] == "polygon"


def test_detect_model_declares_rectangle(fn_dir: Path):
    out, _ = sl.generate_function_files("m", "run1", ["a"], task="detect")
    spec = json.loads(_load(out / "function.yaml")["metadata"]["annotations"]["spec"])
    assert spec[0]["type"] == "rectangle"


def test_gpu_definition_requests_gpu_and_cu128(fn_dir: Path):
    """Blackwell では cu128 が必須（cu126 では動かない）"""
    out, _ = sl.generate_function_files("m", "run1", ["a"])
    gpu = _load(out / "function-gpu.yaml")
    assert gpu["spec"]["resources"]["limits"]["nvidia.com/gpu"] == 1
    assert "12.8" in gpu["spec"]["build"]["baseImage"]
    assert any("cu128" in d["value"]
               for d in gpu["spec"]["build"]["directives"]["preCopy"])


def test_cpu_definition_has_no_gpu_request(fn_dir: Path):
    out, _ = sl.generate_function_files("m", "run1", ["a"])
    assert "resources" not in _load(out / "function.yaml")["spec"]


def test_model_env_points_to_run(fn_dir: Path):
    """deploy.sh がこの値から best.pt を探す"""
    out, _ = sl.generate_function_files("m", "yolo11s_ep100", ["a"])
    assert "MODEL_RUN=yolo11s_ep100" in (out / "model.env").read_text()


def test_class_names_with_quotes_do_not_break_yaml(fn_dir: Path):
    """ラベル名に記号が入っても YAML が壊れないこと"""
    out, _ = sl.generate_function_files("m", "run1", ['say "hi"', "a:b"])
    spec = json.loads(_load(out / "function.yaml")["metadata"]["annotations"]["spec"])
    assert [d["name"] for d in spec] == ['say "hi"', "a:b"]


# ---------------------------------------------------------------------------
# 取り込んだモデル（best.pt 以外の名前）のデプロイ
# ---------------------------------------------------------------------------
def test_best_pt_以外の名前の重みも指せる():
    """取り込んだモデルはファイル名が best.pt とは限らない。
    run 名だけでは指し切れないので、models/ からの相対パスで持つ。"""
    import shutil

    from core.config import SERVERLESS_DIR
    from core.serverless import generate_function_files

    d, _ = generate_function_files(
        fn_dir="_test_imported_weights", model_run="imported_x",
        class_names=["a"], weights_rel="imported_x/weights/my_model.pt")
    try:
        env = (d / "model.env").read_text(encoding="utf-8")
        assert "MODEL_WEIGHTS=imported_x/weights/my_model.pt" in env
        assert "MODEL_RUN=imported_x" in env      # 表示・旧形式の互換
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_weights_rel_を渡さなければ従来どおり():
    import shutil

    from core.serverless import generate_function_files

    d, _ = generate_function_files(
        fn_dir="_test_legacy_weights", model_run="run_a", class_names=["a"])
    try:
        env = (d / "model.env").read_text(encoding="utf-8")
        assert "MODEL_WEIGHTS=run_a/weights/best.pt" in env
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_旧形式の_model_env_も読める():
    """MODEL_RUN しか無い既存の定義を壊さないこと。"""
    import shutil

    from core.config import SERVERLESS_DIR
    from core.serverless import list_serverless_defs

    d = SERVERLESS_DIR / "custom" / "_test_legacy_env"
    d.mkdir(parents=True, exist_ok=True)
    try:
        (d / "model.env").write_text("MODEL_RUN=legacy_run\n", encoding="utf-8")
        got = [x for x in list_serverless_defs() if x["dir"] == "_test_legacy_env"]
        assert got, "旧形式の定義が読めていない"
        assert got[0]["model_run"] == "legacy_run"
        # 相対パスは補完される
        assert got[0]["model_weights"] == "legacy_run/weights/best.pt"
    finally:
        shutil.rmtree(d, ignore_errors=True)
