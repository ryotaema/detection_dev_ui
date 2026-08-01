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
