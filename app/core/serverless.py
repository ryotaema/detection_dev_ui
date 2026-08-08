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

from .config import (
    CVAT_NETWORK,
    MODELS_DIR,
    NUCTL_BIN,
    SAM3_DIR,
    SAM3_MOUNT_PATH,
    SAM3_WEIGHTS_NAME,
    SERVERLESS_DIR,
)
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


def _read_deployed(fn_dir: Path) -> dict:
    """`.deployed.json`（どの重みで焼いたか）を読む。無ければ空。"""
    f = Path(fn_dir) / ".deployed.json"
    if not f.exists():
        return {}
    try:
        d = json.loads(f.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def _weights_sha1(path) -> str:
    """重みの照合用ハッシュ。先頭だけで足りる（差し替われば必ず変わる）。"""
    import hashlib

    p = Path(path)
    if not p.exists():
        return ""
    try:
        h = hashlib.sha1()
        with open(p, "rb") as f:
            h.update(f.read(8 * 1024 * 1024))
        return h.hexdigest()
    except Exception:
        return ""


def _read_model_env(path) -> dict:
    """`model.env`（KEY=VALUE の羅列）を読む。読めなければ空。"""
    out: dict = {}
    p = Path(path)
    if not p.exists():
        return out
    try:
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    except Exception:
        pass
    return out


def list_serverless_defs() -> list[dict]:
    """serverless/custom/ 配下の関数定義（未デプロイのものも含む）"""
    defs = []
    cdir = SERVERLESS_DIR / "custom"
    if not cdir.exists():
        return defs

    for d in sorted(p for p in cdir.iterdir() if p.is_dir()):
        env = _read_model_env(d / "model.env")
        model_run     = env.get("MODEL_RUN", "")
        model_weights = env.get("MODEL_WEIGHTS", "")
        kind          = env.get("MODEL_KIND", "")
        variant       = env.get("SAM3_VARIANT", "")

        if not model_weights and model_run:
            model_weights = f"{model_run}/weights/best.pt"
        if kind == "sam3" and not model_weights:
            model_weights = f"{SAM3_DIR.name}/{SAM3_WEIGHTS_NAME}"

        # 重みはビルド時にコンテナへ焼き込まれる。models/ 側を差し替えても
        # 再デプロイするまで古いままなので、デプロイ時の記録と突き合わせる
        dep = _read_deployed(d)
        wp = MODELS_DIR / model_weights if model_weights else None
        cur_sha = _weights_sha1(wp) if wp else ""
        cur_size = wp.stat().st_size if wp and wp.exists() else 0
        changed = bool(
            dep.get("sha1") and cur_sha
            and (dep["sha1"] != cur_sha
                 or (dep.get("size") and dep["size"] != cur_size)))

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
            "kind": kind or "yolo",
            "variant": variant,
            "model_run": model_run,
            "model_weights": model_weights,
            "deployed_sha1": dep.get("sha1", ""),
            "deployed_at": dep.get("deployed_at", ""),
            "weights_changed": changed,
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


# ===========================================================================
# SAM 3 (Segment Anything Model 3)
#
#   2 通りの使い方があり、CVAT 側の出方も違うので関数を分けている:
#     concept     … detector。テキスト（英語の短い名詞句）に当てはまるものを全部出す
#                   → Actions → Automatic annotation
#     interactive … interactor。点やボックスで指した 1 個だけをマスクにする
#                   → AI Tools → Interactors
#
#   自作モデルと違い、重みはイメージに焼き込まずホストからマウントする
#   （3.45GB あり、焼き込むとビルドのたびにコピーが走るため）。
#   マウント元のホストパスは deploy.sh がデプロイ時に確定させる。
# ===========================================================================
SAM3_VARIANTS = {
    "concept": {
        "label": "テキストで一括（detector）",
        "fn_dir": "sam3-concept",
        "display": "SAM 3 Concept (text)",
        "where": "Actions → Automatic annotation",
    },
    "interactive": {
        "label": "クリックで 1 個ずつ（interactor）",
        "fn_dir": "sam3-interactive",
        "display": "SAM 3 Interactive",
        "where": "AI Tools → Interactors",
    },
}


def sam3_weights_status(weights_name: str = SAM3_WEIGHTS_NAME) -> dict:
    """SAM 3 の重みが置かれているかを見る。

    重みは各自が HuggingFace から取ってくるものなので、
    「無い」ことを前提に案内できるよう置き場所も一緒に返す。
    """
    p = SAM3_DIR / weights_name
    exists = p.exists() and p.is_file()
    return {
        "path": p,
        "dir": SAM3_DIR,
        "name": weights_name,
        "exists": exists,
        "size": p.stat().st_size if exists else 0,
    }


def parse_sam3_prompt_lines(text: str) -> tuple[list[tuple[str, str]], list[str]]:
    """「CVAT ラベル名 = 英語プロンプト」の行を [(ラベル, プロンプト), ...] にする。

    書き方は緩く受ける（人が手で書くものなので）:
      - `fruit = red fruit`  … ラベル名とプロンプトを分ける
      - `fruit`              … 区切りが無ければラベル名をそのままプロンプトにする
      - 区切りは `=` `:` `＝` `：` のどれでもよい
      - 空行と `#` から始まる行は無視する

    **読めなかった行は黙って捨てず、第 2 要素で返して画面に出すこと。**
    ラベルが 1 つ抜けたまま気づかずにデプロイすると、
    CVAT 側では「そのラベルだけ検出されない」という分かりにくい形で出る。

    Returns:
        ([(ラベル, プロンプト), ...], 読めなかった行)
    """
    pairs: list[tuple[str, str]] = []
    bad: list[str] = []
    seen: set[str] = set()

    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        label, prompt = line, line
        for sep in ("=", ":", "＝", "："):
            if sep in line:
                left, right = line.split(sep, 1)
                label, prompt = left.strip(), right.strip()
                break

        if not label or not prompt:
            bad.append(raw)
            continue
        if label in seen:       # 同じラベルを 2 度書かれたら後がちにせず先を残す
            bad.append(raw)
            continue

        seen.add(label)
        pairs.append((label, prompt))

    return pairs, bad


def generate_sam3_function_files(
    variant: str,
    pairs: Optional[list[tuple[str, str]]] = None,
    display_name: str = "",
    weights_name: str = SAM3_WEIGHTS_NAME,
    half: bool = False,
) -> tuple[Path, str]:
    """serverless/custom/<fn_dir>/ に SAM 3 の関数定義一式を生成する。

    Args:
        variant: "concept"（テキスト検出）または "interactive"（クリック）
        pairs:   concept のみ。[(CVAT のラベル名, 英語のテキストプロンプト), ...]
                 並び順がそのまま SAM 3 のクラス index になるので順序を保つこと。
        half:    FP16 で読み込む（GPU メモリを減らせる）

    Returns:
        (生成先ディレクトリ, Nuclio の関数名)
    """
    if variant not in SAM3_VARIANTS:
        raise ValueError(f"未知の variant: {variant}")

    info    = SAM3_VARIANTS[variant]
    fn_dir  = info["fn_dir"]
    slug    = slugify_function_name(fn_dir)
    fn_name = slug
    disp    = display_name or info["display"]
    pairs   = list(pairs or [])

    if variant == "concept" and not pairs:
        raise ValueError("concept では少なくとも 1 つのラベルとプロンプトが要ります")

    def _annotations(suffix: str) -> str:
        name_line = f"    name: {json.dumps(f'{disp} / {suffix}', ensure_ascii=False)}"
        if variant == "interactive":
            # interactor の spec は空。点やボックスの要件はこの下のキーで宣言する
            return "\n".join([
                name_line,
                "    version: 2",
                "    type: interactor",
                "    spec:",
                "    min_pos_points: 0",
                "    min_neg_points: 0",
                "    startswith_box_optional: true",
                "    help_message: " + json.dumps(
                    "対象をボックスで囲むか、内側に正の点・外側に負の点を打つと"
                    "その 1 個のマスクを返します", ensure_ascii=False),
            ])

        # detector: ラベル名は CVAT タスク側と一致させる必要がある。
        # SAM 3 へ渡す英語プロンプトは SAM3_PROMPTS 側に持たせて分離している
        items = [{"id": i, "name": lb, "type": "polygon"} for i, (lb, _) in enumerate(pairs)]
        spec_block = "\n".join(
            "      " + line
            for line in json.dumps(items, ensure_ascii=False, indent=2).splitlines()
        )
        return "\n".join([name_line, "    type: detector", "    spec: |", spec_block])

    def _env(suffix: str) -> str:
        lines = [
            "  env:",
            "    - name: SAM3_WEIGHTS_PATH",
            f"      value: {SAM3_MOUNT_PATH}/{weights_name}",
        ]
        if variant == "concept":
            prompts = json.dumps(
                [{"label": lb, "prompt": pr} for lb, pr in pairs], ensure_ascii=False)
            lines += [
                "    # CVAT のラベル名 → SAM 3 に渡すテキストプロンプト。",
                "    # SAM 3 は英語の短い名詞句を前提にしているので、",
                "    # CVAT 側のラベル名（日本語でも可）とは分けて持つ",
                "    - name: SAM3_PROMPTS",
                f"      value: {json.dumps(prompts, ensure_ascii=False)}",
            ]
        if half:
            lines += ["    - name: SAM3_HALF", '      value: "1"']
        lines += [
            "    - name: YOLO_CONFIG_DIR",
            "      value: /tmp/Ultralytics",
            "    - name: MPLCONFIGDIR",
            "      value: /tmp/matplotlib",
        ]
        return "\n".join(lines)

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
# このファイルは Streamlit UI (Step1 アノテーションタブ) が自動生成しました。
#   SAM 3 ({variant}) — CVAT では「{info["where"]}」に出ます。
#
#   重みはイメージに焼き込まず、ホストの models/{SAM3_DIR.name}/ をマウントして読む。
#   マウント元のホストパスは serverless/deploy.sh がデプロイ時に埋める
#   （__SAM3_WEIGHTS_HOST_DIR__ のまま nuctl に渡してはいけない）。
# =============================================================================
metadata:
  name: {fn_name}
  namespace: cvat
  annotations:
{_annotations(suffix)}

spec:
  description: {json.dumps(f"SAM 3 ({variant}) / Meta / {suffix}", ensure_ascii=False)}
  runtime: "python:3.10"
  handler: main:handler
  eventTimeout: 120s
  # 3.45GB の重みを読み終えるまで起動完了にならない。既定 (120秒) では足りない
  readinessTimeoutSeconds: 900

{_env(suffix)}

  build:
    image: cvat.sam3.{variant}{tag}
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

  # 重みはビルド時にコピーせず、ホストのディレクトリをそのまま見せる。
  # 差し替えても再ビルドは不要だが、**関数の再起動は必要**
  # （プロセスが起動時に読んだモデルを持ち続けるため）
  volumes:
    - volume:
        name: sam3-weights
        hostPath:
          path: __SAM3_WEIGHTS_HOST_DIR__
      volumeMount:
        name: sam3-weights
        mountPath: {SAM3_MOUNT_PATH}
        readOnly: true

  triggers:
    myHttpTrigger:
      # SAM 3 は GPU メモリを大きく使うので、worker を増やさないこと
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
    (out_dir / "model.env").write_text(
        "# SAM 3 の関数定義\n"
        "#   MODEL_KIND    … sam3 なら deploy.sh が重みをマウント方式で扱う\n"
        "#   SAM3_VARIANT  … concept (detector) / interactive (interactor)\n"
        "#   MODEL_WEIGHTS … models/ からの相対パス\n"
        "MODEL_KIND=sam3\n"
        f"SAM3_VARIANT={variant}\n"
        f"MODEL_WEIGHTS={SAM3_DIR.name}/{weights_name}\n"
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
