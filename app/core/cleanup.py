# =============================================================================
# 片付け（使われていないもの・未完成のものを見つける）
#
#   放っておくと「データセット 6 件・モデル 14 件」と表示されるのに
#   実際に使えるのは 2 件と 5 個、という状態になる。
#   数だけ増えて中身が伴わないと、次に何をすべきか読み取れなくなる。
#
#   **消す判断は人がする。** ここは候補を挙げて材料を添えるだけで、
#   自動では何も消さない。特に「使用実績のあるもの」は必ず知らせる。
# =============================================================================
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

from .config import DATA_DIR, IMG_EXTS, MODELS_DIR, PREDICTIONS_DIR
from .provenance import dataset_usage_summary, read_provenance, read_status

# predictions/ 配下の、消しても作り直せるもの
TEMP_DIRS = {
    "_tmp_uploads":  "推論のためにアップロードした画像の一時置き場",
    "_mosaic_scan":  "モザイクの対象を探すために走らせた推論結果",
    "_crop_scan":    "クロップの対象を探すために走らせた推論結果",
    "exports":       "書き出した結果画像",
    "_exports":      "書き出したデータセット ZIP",
}


def _dir_size(path: Path) -> int:
    try:
        return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())
    except Exception:
        return 0


def _count_images(path: Path) -> int:
    return sum(1 for p in Path(path).rglob("*")
               if p.is_file() and p.suffix.lower() in IMG_EXTS
               and "_backup_original" not in p.parts)


# ---------------------------------------------------------------------------
# 未完成のデータセット
# ---------------------------------------------------------------------------
def find_incomplete_datasets(data_root: Optional[Path] = None) -> list[dict]:
    """学習に使えないデータセットを挙げる。

    判定は「`data.yaml` があるか」で行う。画像の有無ではない。
    CVAT からエクスポートしただけ（raw/ のみ）の状態は、
    画像はあっても学習には使えないため。
    """
    root = Path(data_root or DATA_DIR)
    if not root.exists():
        return []

    out: list[dict] = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if (d / "data.yaml").exists():
            continue
        n_img = _count_images(d)
        has_raw = (d / "raw").exists()

        if n_img == 0:
            reason, hint = "中身がありません", "削除して構いません"
        elif has_raw:
            reason = "CVAT からエクスポートしただけで、まだ YOLO 形式に変換していません"
            hint = "「📤 Step2: データ取込」の「💡 既にRAWデータがある場合」から変換できます"
        else:
            reason = "画像はありますが data.yaml がありません"
            hint = "「📤 Step2: データ取込」で変換するか、削除してください"

        out.append({
            "name": d.name, "dir": str(d), "images": n_img,
            "has_raw": has_raw, "size": _dir_size(d),
            "reason": reason, "hint": hint,
            "status": read_status(d),
            "usage": dataset_usage_summary(d),
        })
    return out


# ---------------------------------------------------------------------------
# 重みの無い学習 run
# ---------------------------------------------------------------------------
def find_runs_without_weights(models_root: Optional[Path] = None) -> list[dict]:
    """`.pt` が残っていない run を挙げる。

    学習が途中で終わると、プロットや args.yaml だけが残る。
    モデル一覧には出ないので気づきにくく、放っておくと溜まる。
    """
    root = Path(models_root or MODELS_DIR)
    if not root.exists():
        return []

    out: list[dict] = []
    for d in sorted((p for p in root.iterdir() if p.is_dir()),
                    key=lambda p: p.stat().st_mtime, reverse=True):
        if list(d.rglob("*.pt")):
            continue
        prov = read_provenance(d) or {}
        # results.csv があれば何エポックまで進んだか分かる
        epochs = None
        rcsv = d / "results.csv"
        if rcsv.exists():
            try:
                epochs = max(0, len(rcsv.read_text().splitlines()) - 1)
            except Exception:
                pass
        out.append({
            "name": d.name, "dir": str(d), "size": _dir_size(d),
            "trained_at": prov.get("trained_at", ""),
            "dataset": (prov.get("dataset") or {}).get("name", ""),
            "epochs": epochs,
            "files": len([p for p in d.rglob("*") if p.is_file()]),
        })
    return out


# ---------------------------------------------------------------------------
# 一時ファイル
# ---------------------------------------------------------------------------
def find_temp_files(predictions_root: Optional[Path] = None) -> list[dict]:
    """消しても作り直せるものを挙げる"""
    root = Path(predictions_root or PREDICTIONS_DIR)
    if not root.exists():
        return []

    out: list[dict] = []
    for name, desc in TEMP_DIRS.items():
        d = root / name
        if not d.exists():
            continue
        size = _dir_size(d)
        if size == 0 and not any(d.iterdir()):
            continue
        out.append({
            "name": name, "dir": str(d), "desc": desc, "size": size,
            "files": len([p for p in d.rglob("*") if p.is_file()]),
        })
    return sorted(out, key=lambda x: -x["size"])


# ---------------------------------------------------------------------------
# まとめ
# ---------------------------------------------------------------------------
def cleanup_summary() -> dict:
    """片付けの候補をまとめて返す"""
    incomplete = find_incomplete_datasets()
    runs = find_runs_without_weights()
    temps = find_temp_files()
    return {
        "incomplete_datasets": incomplete,
        "runs_without_weights": runs,
        "temp_files": temps,
        "total_size": (sum(x["size"] for x in incomplete)
                       + sum(x["size"] for x in runs)
                       + sum(x["size"] for x in temps)),
        "n_items": len(incomplete) + len(runs) + len(temps),
    }


def delete_paths(paths, guard_root: Optional[Path] = None) -> dict:
    """まとめて削除する。

    guard_root を渡すと、その配下でないものは消さない。
    片付けは「まとめて消す」操作なので、範囲の取り違えを防ぐ。
    """
    deleted, skipped, errors = [], [], []
    guard = Path(guard_root).resolve() if guard_root else None

    for p in paths:
        path = Path(p)
        if not path.exists():
            skipped.append((str(path), "既にありません"))
            continue
        if guard is not None:
            try:
                path.resolve().relative_to(guard)
            except ValueError:
                skipped.append((str(path), f"{guard} の外にあります"))
                continue
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            deleted.append(str(path))
        except Exception as e:
            errors.append((str(path), str(e)))

    return {"ok": not errors, "deleted": deleted, "skipped": skipped,
            "errors": errors,
            "error": f"{len(errors)} 件を削除できませんでした" if errors else ""}
