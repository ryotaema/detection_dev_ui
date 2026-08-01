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
