# =============================================================================
# モデルファイルのメタ情報と持ち出し
# =============================================================================
from __future__ import annotations

import json
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# モデルファイル (.pt) のメタ情報
# ---------------------------------------------------------------------------

def model_weight_files(root=None) -> list:
    """モデル一覧に出してよい重みだけを返す。

    `.tuning/`（探索の作業場）のように、**中間生成物を混ぜないこと**。
    探索は 1 回ごとに使い捨ての学習を回すので、そのまま列挙すると
    「train-2」のような名前が本物のモデルに紛れる。
    ドット始まりのディレクトリは総じて作業用とみなす。
    """
    from .config import MODELS_DIR

    base = Path(root or MODELS_DIR)
    if not base.exists():
        return []
    return sorted(p for p in base.rglob("*.pt")
                  if not any(part.startswith(".") for part in
                             p.relative_to(base).parts[:-1]))


def model_run_dirs(root=None) -> list:
    """モデルの run ディレクトリ（作業用は除く）"""
    from .config import MODELS_DIR

    base = Path(root or MODELS_DIR)
    if not base.exists():
        return []
    return sorted(d for d in base.iterdir()
                  if d.is_dir() and not d.name.startswith("."))

def safe_run_name(name: str) -> str:
    """アップロード時のモデル名を、そのままディレクトリ名にできる形に直す。

    利用者が入力した文字列をパスに使うので、`/` や `..` を素通しにしない
    （models/ の外に書き込まれるのを防ぐ）。
    """
    import re

    safe = re.sub(r"[^\w\-.]", "_", str(name or "")).strip("._")
    return safe or f"imported_{datetime.now():%Y%m%d_%H%M}"


def import_model_weights(run_name: str, files, extras=None) -> tuple[list, str]:
    """アップロードされた重みを `models/<run_name>/weights/` に取り込む。

    UI（Step1 のデプロイ画面 / データ管理タブ）の 2 か所から呼ぶので core に置く。
    Streamlit に依存させないため、受け取るのは (ファイル名, バイト列) の並び。

    Args:
        run_name: models/ 以下に作るディレクトリ名（危険な文字は落とす）
        files:   [(名前, bytes), ...] 重みファイル
        extras:  [(名前, bytes), ...] results.csv などの付随ファイル（任意）

    Returns:
        (保存した重みのパス, エラーメッセージ)。エラーがあれば空リストを返す。
    """
    from .config import MODELS_DIR

    if not files:
        return [], "重みファイルが選ばれていません"

    run = safe_run_name(run_name)
    wdir = Path(MODELS_DIR) / run / "weights"
    try:
        wdir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        return [], f"保存先を作れませんでした: {type(e).__name__}: {e}"

    saved = []
    try:
        for name, data in files:
            # ファイル名も利用者由来なので、ディレクトリ部分は捨てる
            fname = Path(str(name)).name
            if not fname.endswith(".pt"):
                fname += ".pt"
            dst = wdir / fname
            dst.write_bytes(data)
            saved.append(dst)
        for name, data in (extras or []):
            (wdir.parent / Path(str(name)).name).write_bytes(data)
    except Exception as e:
        return saved, f"保存に失敗しました: {type(e).__name__}: {e}"

    return saved, ""


def model_meta_path(model_path: Path) -> Path:
    """`weights/best.pt` に対する `weights/.best.pt.meta.json` を返す（rglob('*.pt') に載らない名前）"""
    return model_path.parent / f".{model_path.name}.meta.json"


def read_model_meta(model_path: Path) -> Optional[dict]:
    """保存済みメタ情報を読む。存在しない/壊れている場合は None"""
    mp = model_meta_path(model_path)
    if not mp.exists():
        return None
    try:
        with open(mp) as f:
            return json.load(f)
    except Exception:
        return None


def inspect_model_file(model_path: Path, save: bool = True) -> dict:
    """.pt を実際に読み込んでクラス名などを取得する。
    この環境の ultralytics で読めるか（バージョン互換）の検証も兼ねる。
    """
    info: dict = {
        "ok": False, "error": None, "names": [], "task": None,
        "ultralytics_version": None, "trained_at": None,
        "imgsz": None, "epochs": None, "base_model": None,
        "inspected_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # ckpt 側のメタ（学習時の情報）。読めなくても致命的ではない
    try:
        import torch
        ckpt = torch.load(str(model_path), map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict):
            info["ultralytics_version"] = ckpt.get("version")
            info["trained_at"] = ckpt.get("date")
            targs = ckpt.get("train_args") or {}
            if isinstance(targs, dict):
                info["imgsz"]      = targs.get("imgsz")
                info["epochs"]     = targs.get("epochs")
                info["base_model"] = targs.get("model")
    except Exception:
        pass

    # YOLO として読めるか（ここが通れば推論可能）
    try:
        from ultralytics import YOLO

        model = YOLO(str(model_path))
        names = getattr(model, "names", None) or {}
        if isinstance(names, dict):
            info["names"] = [names[k] for k in sorted(names)]
        else:
            info["names"] = list(names)
        info["task"] = getattr(model, "task", None)
        info["ok"] = True
    except Exception as e:
        info["error"] = f"{type(e).__name__}: {e}"

    if save:
        try:
            with open(model_meta_path(model_path), "w") as f:
                json.dump(info, f, indent=2, ensure_ascii=False)
        except Exception:
            pass
    return info


def build_model_bundle_zip(model_path: Path, out_path: Path) -> tuple[bool, str, int]:
    """モデル一式（重み + 学習ログ + 評価結果 + プロット）を ZIP にまとめる。

    他の PC へ持ち出したとき、`models/<run>/weights/best.pt` の構造のまま
    展開できるようにする（このUIの取込・デプロイがその構造を前提にするため）。
    """
    try:
        run_dir = model_path.parent.parent if model_path.parent.name == "weights" \
            else model_path.parent
        n = 0
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            zf.write(model_path, arcname=f"weights/{model_path.name}")
            n += 1
            for p in sorted(run_dir.rglob("*")):
                if not p.is_file() or p == model_path:
                    continue
                # 他の重み（last.pt 等）は除き、記録類とメタ情報だけ入れる
                if p.suffix == ".pt":
                    continue
                zf.write(p, arcname=str(p.relative_to(run_dir)))
                n += 1
        return True, str(out_path), n
    except Exception as e:
        return False, f"{type(e).__name__}: {e}", 0
