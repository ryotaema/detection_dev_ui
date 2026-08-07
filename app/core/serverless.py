# =============================================================================
# CVAT 自動アノテーション（Nuclio）連携
# =============================================================================
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Optional

import streamlit as st

from .config import CVAT_NETWORK, MODELS_DIR, NUCTL_BIN, SERVERLESS_DIR
from .state import _get_deploy_shared
from .utils import slugify_function_name


def serverless_status() -> dict:
    """UI から Nuclio デプロイが実行可能かどうかを判定する"""
    import shutil as _sh

    return {
        "deploy_sh":   (SERVERLESS_DIR / "deploy.sh").exists(),
        "nuctl":       NUCTL_BIN.exists(),
        "docker_sock": Path("/var/run/docker.sock").exists(),
        "docker_cli":  _sh.which("docker") is not None,
        "network":     CVAT_NETWORK,
    }


def serverless_ready() -> bool:
    s = serverless_status()
    return all([s["deploy_sh"], s["nuctl"], s["docker_sock"], s["docker_cli"]])


def _nuctl(*args: str, timeout: int = 60):
    """nuctl をサブプロセス実行して CompletedProcess を返す"""
    import subprocess

    return subprocess.run(
        [str(NUCTL_BIN), *args, "--platform", "local"],
        capture_output=True, text=True, timeout=timeout,
    )


def list_nuclio_functions() -> list[dict]:
    """デプロイ済み Nuclio 関数の一覧（= CVAT の Automatic annotation に出るモデル）"""
    if not NUCTL_BIN.exists():
        return []
    try:
        proc = _nuctl("get", "function", "-o", "json")
        if proc.returncode != 0:
            return []
        raw = json.loads(proc.stdout or "[]")
    except Exception:
        return []

    out = []
    for fn in raw if isinstance(raw, list) else [raw]:
        meta   = fn.get("metadata", {}) or {}
        spec   = fn.get("spec", {}) or {}
        status = fn.get("status", {}) or {}
        ann    = meta.get("annotations", {}) or {}

        labels = []
        try:
            labels = [l.get("name", "") for l in json.loads(ann.get("spec") or "[]")]
        except Exception:
            pass

        res = spec.get("resources", {}) or {}
        out.append({
            "name":    meta.get("name", ""),
            "display": ann.get("name", "") or meta.get("name", ""),
            "type":    ann.get("type", ""),
            "labels":  labels,
            "image":   spec.get("image", ""),
            "gpu":     "nvidia.com/gpu" in json.dumps(res.get("limits", {}) or {}),
            "state":   status.get("state", ""),
            "port":    (status.get("httpPort") or ""),
        })
    return sorted(out, key=lambda d: d["name"])


@st.cache_data(ttl=10, show_spinner=False)
def cached_nuclio_functions() -> list[dict]:
    """関数一覧の短期キャッシュ。毎 rerun で nuctl を起動しないための薄いラッパ。
    デプロイ・削除の直後は呼び出し側で .clear() すること。
    """
    return list_nuclio_functions()


def list_serverless_defs() -> list[dict]:
    """serverless/custom/ 配下の関数定義（未デプロイのものも含む）"""
    defs = []
    cdir = SERVERLESS_DIR / "custom"
    if not cdir.exists():
        return defs

    for d in sorted(p for p in cdir.iterdir() if p.is_dir()):
        model_run, model_weights = "", ""
        env_f = d / "model.env"
        if env_f.exists():
            for line in env_f.read_text().splitlines():
                line = line.strip()
                for key, setter in (("MODEL_RUN=", "run"), ("MODEL_WEIGHTS=", "w")):
                    if line.startswith(key):
                        v = line.split("=", 1)[1].strip().strip('"').strip("'")
                        if setter == "run":
                            model_run = v
                        else:
                            model_weights = v
        if not model_weights and model_run:
            model_weights = f"{model_run}/weights/best.pt"

        fn_name, fn_display = "", ""
        y = d / "function.yaml"
        if y.exists():
            try:
                import yaml as _yml
                _meta = ((_yml.safe_load(y.read_text()) or {}).get("metadata") or {})
                fn_name    = _meta.get("name", "")
                fn_display = (_meta.get("annotations") or {}).get("name", "")
            except Exception:
                pass

        defs.append({
            "dir": d.name,
            "model_run": model_run,
            "model_weights": model_weights,
            "function_name": fn_name,
            "display": fn_display,
            "has_gpu_yaml": (d / "function-gpu.yaml").exists(),
            "model_exists": bool(model_weights)
                            and (MODELS_DIR / model_weights).exists(),
        })
    return defs


def generate_function_files(
    fn_dir: str,
    model_run: str,
    class_names: list[str],
    display_name: str = "",
    description: str = "",
    task: str = "detect",
    weights_rel: str = "",
) -> tuple[Path, str]:
    """serverless/custom/<fn_dir>/ に関数定義一式を生成し、(ディレクトリ, 関数名) を返す。

    CPU 版 (function.yaml) と GPU 版 (function-gpu.yaml) の両方を出力する。
    deploy.sh がどちらを使うかを選ぶ。
    """
    slug     = slugify_function_name(fn_dir)
    fn_name  = f"custom-{slug}"
    image    = f"cvat.custom.{slug}"
    _src_rel = weights_rel or f"{model_run}/weights/best.pt"
    disp     = display_name or f"{model_run} (custom)"
    desc     = description or f"自作 YOLO 検出器 ({model_run} / Ultralytics)"

    # ラベル定義はモデルのクラス名から生成（json.dumps でエスケープを担保）。
    # セグメンテーションモデルは polygon を返すため、ラベル種別も合わせる。
    shape_type = "polygon" if str(task) == "segment" else "rectangle"
    items = [{"id": i, "name": n, "type": shape_type} for i, n in enumerate(class_names)]
    spec_block = "\n".join(
        "      " + line for line in json.dumps(items, ensure_ascii=False, indent=2).splitlines()
    )

    def _yaml(gpu: bool) -> str:
        if gpu:
            base_image = "nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04"
            torch_url  = "https://download.pytorch.org/whl/cu128"
            tag        = ":latest-gpu"
            suffix     = "GPU cu128"
            resources  = (
                "\n  # GPU を割り当てる。有効化には Docker daemon の default-runtime=nvidia が必要\n"
                "  resources:\n"
                "    limits:\n"
                "      nvidia.com/gpu: 1\n"
            )
        else:
            base_image = "ubuntu:22.04"
            torch_url  = "https://download.pytorch.org/whl/cpu"
            tag        = ""
            suffix     = "CPU"
            resources  = ""

        return f"""# =============================================================================
# このファイルは Streamlit UI (データ管理タブ) が自動生成しました。
#   生成元モデル: models/{_src_rel}
#   annotations.spec のラベルはモデルのクラス名から生成されています。
#   CVAT タスク側のラベル名と一致していることを確認してください。
# =============================================================================
metadata:
  name: {fn_name}
  namespace: cvat
  annotations:
    name: {json.dumps(f"{disp} / {suffix}", ensure_ascii=False)}
    type: detector
    spec: |
{spec_block}

spec:
  description: {json.dumps(f"{desc} / {suffix}", ensure_ascii=False)}
  runtime: "python:3.10"
  handler: main:handler
  eventTimeout: 60s

  env:
    - name: YOLO_CONFIG_DIR
      value: /tmp/Ultralytics
    - name: MPLCONFIGDIR
      value: /tmp/matplotlib

  build:
    image: {image}{tag}
    baseImage: {base_image}
    directives:
      preCopy:
        - kind: RUN
          value: apt-get update && apt-get install --no-install-recommends -y python3-pip python-is-python3 libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
        - kind: RUN
          value: pip install --no-cache-dir torch torchvision --index-url {torch_url}
        - kind: RUN
          value: pip install --no-cache-dir ultralytics==8.4.48 opencv-python-headless==4.10.0.82 pillow pyyaml
        - kind: WORKDIR
          value: /opt/nuclio

  triggers:
    myHttpTrigger:
      numWorkers: 1
      kind: "http"
      workerAvailabilityTimeoutMilliseconds: 10000
      attributes:
        maxRequestBodySize: 33554432 # 32MB
{resources}
  platform:
    attributes:
      restartPolicy:
        name: always
        maximumRetryCount: 3
      mountMode: volume
"""

    out_dir = SERVERLESS_DIR / "custom" / fn_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "function.yaml").write_text(_yaml(gpu=False))
    (out_dir / "function-gpu.yaml").write_text(_yaml(gpu=True))
    # 取り込んだモデルはファイル名が best.pt とは限らないので、
    # models/ からの相対パスで持つ（MODEL_RUN は表示・旧形式の互換用）
    _rel = weights_rel or f"{model_run}/weights/best.pt"
    (out_dir / "model.env").write_text(
        "# この関数が使う学習済みモデル\n"
        "#   MODEL_WEIGHTS … models/ からの相対パス（best.pt 以外の名前でも可）\n"
        "#   MODEL_RUN     … 表示用。旧形式では models/<run>/weights/best.pt を指した\n"
        "# serverless/deploy.sh がこれを読み、重みをビルドコンテキストへコピーする\n"
        f"MODEL_RUN={model_run}\n"
        f"MODEL_WEIGHTS={_rel}\n"
    )
    return out_dir, fn_name


def _deploy_worker(fn_dir: str, use_gpu: bool) -> None:
    """deploy.sh をサブプロセス実行し、出力を共有ログへ流す（バックグラウンドスレッド）"""
    import subprocess

    state, lock = _get_deploy_shared()

    def _log(msg: str) -> None:
        with lock:
            state["log"].append(msg)

    try:
        env = os.environ.copy()
        if CVAT_NETWORK:
            env["CVAT_NETWORK"] = CVAT_NETWORK

        cmd = ["bash", str(SERVERLESS_DIR / "deploy.sh"),
               "--gpu" if use_gpu else "--cpu", fn_dir]
        _log(f"$ {' '.join(cmd)}")
        _log(f"(network={env.get('CVAT_NETWORK', '自動判定')})")

        proc = subprocess.Popen(
            cmd, cwd=str(SERVERLESS_DIR), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        for line in proc.stdout:            # type: ignore[union-attr]
            _log(line.rstrip())
        proc.wait()

        if proc.returncode != 0:
            with lock:
                state["error"] = f"deploy.sh が終了コード {proc.returncode} で失敗しました"
        else:
            _log("")
            _log("✅ デプロイ完了。CVAT の Actions → Automatic annotation に反映されます"
                 "（反映まで十数秒かかる場合があります）")
    except Exception as e:
        with lock:
            state["error"] = f"{type(e).__name__}: {e}"
    finally:
        with lock:
            state["running"] = False
            state["finished"] = True


def start_deploy(fn_dir: str, use_gpu: bool) -> None:
    """デプロイをバックグラウンドで開始する"""
    state, lock = _get_deploy_shared()
    with lock:
        if state["running"]:
            return
        state["log"] = []
        state["error"] = None
        state["running"] = True
        state["finished"] = False
        state["target"] = fn_dir

    threading.Thread(target=_deploy_worker, args=(fn_dir, use_gpu), daemon=True).start()


def delete_nuclio_function(fn_name: str) -> tuple[bool, str]:
    """デプロイ済み関数を削除する（CVAT の選択肢からも消える）"""
    if not NUCTL_BIN.exists():
        return False, "nuctl が見つかりません"
    try:
        proc = _nuctl("delete", "function", fn_name, timeout=120)
        if proc.returncode == 0:
            return True, proc.stdout.strip() or "削除しました"
        return False, (proc.stderr or proc.stdout).strip()
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
