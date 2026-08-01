# =============================================================================
# 来歴管理（どのデータで何を作ったかの記録）
# =============================================================================
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import IMG_EXTS, PROVENANCE_FILE


def read_provenance(target_dir: Path) -> Optional[dict]:
    """データセット / モデル run ディレクトリの来歴を読む"""
    p = Path(target_dir) / PROVENANCE_FILE
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def write_provenance(target_dir: Path, data: dict) -> None:
    try:
        Path(target_dir).mkdir(parents=True, exist_ok=True)
        (Path(target_dir) / PROVENANCE_FILE).write_text(
            json.dumps(data, ensure_ascii=False, indent=2))
    except Exception:
        pass


def count_dataset_items(dataset_dir: Path) -> dict:
    """データセットのスプリット別枚数を数える（学習時点のスナップショット用）"""
    counts: dict[str, int] = {}
    img_root = dataset_dir / "images"
    if img_root.exists():                      # detect / segment / pose / obb
        for sp in sorted(p for p in img_root.iterdir() if p.is_dir()):
            counts[sp.name] = len([p for p in sp.iterdir()
                                   if p.is_file() and p.suffix.lower() in IMG_EXTS])
    else:                                      # classify
        for sp in ("train", "val", "test"):
            sp_dir = dataset_dir / sp
            if sp_dir.exists():
                counts[sp] = sum(
                    len([p for p in c.iterdir()
                         if p.is_file() and p.suffix.lower() in IMG_EXTS])
                    for c in sp_dir.iterdir() if c.is_dir()
                )
    return counts


def record_dataset_provenance(
    dataset_dir: Path,
    source: str,
    task_type: str = "",
    labels: Optional[list[str]] = None,
    cvat_tasks: Optional[list[dict]] = None,
    extra: Optional[dict] = None,
) -> dict:
    """データセットの出所を記録する。

    source: "cvat" / "upload_zip" / "upload_images" / "merge" / "unknown"
    """
    prov = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": Path(dataset_dir).name,
        "source": source,
        "task_type": task_type,
        "labels": labels or [],
        "cvat_tasks": cvat_tasks or [],
        "counts": count_dataset_items(Path(dataset_dir)),
    }
    if extra:
        prov.update(extra)
    write_provenance(Path(dataset_dir), prov)
    return prov


def record_model_provenance(
    run_dir: Path,
    data_yaml: str,
    base_model: str,
    params: dict,
    resumed: bool = False,
) -> dict:
    """学習開始時点の情報を記録する。データセット側の来歴もコピーして保持する。"""
    ds_dir = Path(data_yaml).parent if data_yaml else None
    ds_prov = read_provenance(ds_dir) if ds_dir and ds_dir.exists() else None

    classes: list[str] = []
    if ds_dir and (ds_dir / "data.yaml").exists():
        try:
            import yaml as _y
            cfg = _y.safe_load((ds_dir / "data.yaml").read_text()) or {}
            names = cfg.get("names")
            if isinstance(names, dict):
                classes = [names[k] for k in sorted(names)]
            elif isinstance(names, list):
                classes = list(names)
        except Exception:
            pass

    prov = {
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "run": Path(run_dir).name,
        "resumed": resumed,
        "base_model": base_model,
        "dataset": {
            "name": ds_dir.name if ds_dir else "",
            "data_yaml": str(data_yaml),
            "classes": classes,
            # 学習した時点の枚数。あとでデータを足しても、この値は当時のまま残る
            "counts_at_train": count_dataset_items(ds_dir) if ds_dir and ds_dir.exists() else {},
            "provenance": ds_prov,
        },
        "params": params,
    }
    write_provenance(Path(run_dir), prov)
    return prov
