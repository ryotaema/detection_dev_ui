# =============================================================================
# 来歴管理（どのデータで何を作ったかの記録）
# =============================================================================
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import IMG_EXTS, PROVENANCE_FILE


# ---------------------------------------------------------------------------
# 状態（status）とタグ
#
#   status は「いまどの段階にあるか」を表す**排他の 1 つ**。
#   tags   は「どういう性格のものか」を表す**自由入力・複数可**。
#
#   両方を tags にすると「精査済みかつ未精査」のような状態が作れてしまうので
#   分けている。段階は決まった語彙、性格づけは自由、という住み分け。
# ---------------------------------------------------------------------------
DATASET_STATUSES: dict[str, dict] = {
    "draft":          {"icon": "🟡", "label": "作成中",
                       "desc": "取り込んだだけ。中身は未確認"},
    "auto_annotated": {"icon": "🟠", "label": "自動アノテのみ",
                       "desc": "BBOX は付いているが人の目が入っていない"},
    "reviewed":       {"icon": "🟢", "label": "精査済み",
                       "desc": "人が確認・修正済み。学習に使ってよい"},
    "test_only":      {"icon": "🔵", "label": "テスト用",
                       "desc": "評価専用。学習には使わない"},
    "archived":       {"icon": "⚪", "label": "保管",
                       "desc": "もう使わないが消したくない"},
}
DEFAULT_DATASET_STATUS = "draft"

MODEL_STATUSES: dict[str, dict] = {
    "experimental": {"icon": "🧪", "label": "実験中",
                     "desc": "試している途中。結果は当てにしない"},
    "candidate":    {"icon": "🔬", "label": "候補",
                     "desc": "評価済みで見込みあり。比較対象に使う"},
    "production":   {"icon": "🚀", "label": "実用",
                     "desc": "実機・自動アノテーションで使ってよい"},
    "deprecated":   {"icon": "🗑", "label": "非推奨",
                     "desc": "より良いものに置き換わった"},
}
DEFAULT_MODEL_STATUS = "experimental"

MAX_TAGS = 12
MAX_TAG_LEN = 24


def status_table(kind: str = "dataset") -> dict[str, dict]:
    return MODEL_STATUSES if kind == "model" else DATASET_STATUSES


def default_status(kind: str = "dataset") -> str:
    return DEFAULT_MODEL_STATUS if kind == "model" else DEFAULT_DATASET_STATUS


def status_label(value: str, kind: str = "dataset") -> str:
    """"🟢 精査済み" のような表示用の文字列を返す。"""
    info = status_table(kind).get(value)
    if not info:
        return f"❔ {value}" if value else "❔ 未設定"
    return f"{info['icon']} {info['label']}"


def normalize_tags(raw) -> list[str]:
    """タグを整える。文字列ならカンマ区切りとして解釈する。

    重複を除き、前後の空白を落とし、長すぎるもの・多すぎるものは切り捨てる。
    順序は入力順を保つ（表示が安定するように）。
    """
    if raw is None:
        return []
    items = raw.split(",") if isinstance(raw, str) else list(raw)

    out: list[str] = []
    for item in items:
        t = str(item).strip()
        if not t:
            continue
        t = t[:MAX_TAG_LEN]
        if t not in out:
            out.append(t)
    return out[:MAX_TAGS]


def read_provenance(target_dir: Path) -> Optional[dict]:
    """データセット / モデル run ディレクトリの来歴を読む"""
    p = Path(target_dir) / PROVENANCE_FILE
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def read_status(target_dir: Path, kind: str = "dataset") -> str:
    """状態を読む。記録が無い / 知らない値なら既定を返す。

    来歴を入れる前に作られたものを勝手に「精査済み」にしないよう、
    既定はいちばん手前の段階にしている。
    """
    prov = read_provenance(target_dir) or {}
    value = prov.get("status")
    return value if value in status_table(kind) else default_status(kind)


def read_tags(target_dir: Path) -> list[str]:
    return normalize_tags((read_provenance(target_dir) or {}).get("tags"))


def read_note(target_dir: Path) -> str:
    return str((read_provenance(target_dir) or {}).get("note") or "")


def write_provenance(target_dir: Path, data: dict) -> bool:
    """来歴を書き出す。成功したかを返す（既存の呼び出し側は戻り値を見ていない）"""
    try:
        Path(target_dir).mkdir(parents=True, exist_ok=True)
        (Path(target_dir) / PROVENANCE_FILE).write_text(
            json.dumps(data, ensure_ascii=False, indent=2))
        return True
    except Exception:
        return False


def update_provenance(
    target_dir: Path,
    kind: str = "dataset",
    status: Optional[str] = None,
    tags=None,
    note: Optional[str] = None,
) -> dict:
    """既存の来歴を保ったまま status / tags / note だけを差し替える。

    来歴がまだ無いディレクトリにも付けられる（最小限の器を作る）。
    None を渡した項目は触らない。

    戻り値: {"ok": bool, "error": str, "provenance": dict}
    """
    target = Path(target_dir)
    if not target.exists():
        return {"ok": False, "error": f"ディレクトリがありません: {target}",
                "provenance": {}}

    if status is not None and status not in status_table(kind):
        return {"ok": False,
                "error": f"知らない状態です: {status}"
                         f"（{'/'.join(status_table(kind))} のいずれか）",
                "provenance": {}}

    prov = read_provenance(target)
    if prov is None:
        # 来歴が無いものにも付けられるようにする。
        # source は "unknown" のままにして、後から作った記録だと分かるようにする。
        prov = {
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset" if kind == "dataset" else "run": target.name,
            "source": "unknown",
            "provenance_added_later": True,
        }

    if status is not None:
        prov["status"] = status
    if tags is not None:
        prov["tags"] = normalize_tags(tags)
    if note is not None:
        prov["note"] = str(note).strip()
    prov["status_updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if not write_provenance(target, prov):
        return {"ok": False, "error": f"{PROVENANCE_FILE} を書き込めませんでした",
                "provenance": prov}
    return {"ok": True, "error": "", "provenance": prov}


def collect_tags(dirs) -> list[str]:
    """複数のディレクトリで使われているタグを集める（絞り込みの候補用）"""
    seen: list[str] = []
    for d in dirs:
        for t in read_tags(Path(d)):
            if t not in seen:
                seen.append(t)
    return sorted(seen)


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


def snapshot_dataset_source(dataset_dir: Path) -> dict:
    """統合の記録用に、親データセットの「その時点」の姿を写し取る。

    親があとで消えたり中身が変わっても、何を混ぜたのかが残るようにする。
    モデル側の来歴で枚数を焼き込んでいるのと同じ考え方。
    """
    d = Path(dataset_dir)
    prov = read_provenance(d) or {}
    return {
        "name": d.name,
        "counts": count_dataset_items(d),
        "status": read_status(d, "dataset"),
        "tags": read_tags(d),
        "task_type": prov.get("task_type", ""),
        "cvat_tasks": prov.get("cvat_tasks", []),
    }


def record_dataset_provenance(
    dataset_dir: Path,
    source: str,
    task_type: str = "",
    labels: Optional[list[str]] = None,
    cvat_tasks: Optional[list[dict]] = None,
    extra: Optional[dict] = None,
    status: Optional[str] = None,
    tags=None,
    sources: Optional[list[dict]] = None,
) -> dict:
    """データセットの出所を記録する。

    source:  "cvat" / "upload_zip" / "upload_images" / "merge" / "unknown"
    sources: source == "merge" のとき、何を混ぜたか（snapshot_dataset_source の並び）
    """
    prov = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": Path(dataset_dir).name,
        "source": source,
        "task_type": task_type,
        "labels": labels or [],
        "cvat_tasks": cvat_tasks or [],
        "counts": count_dataset_items(Path(dataset_dir)),
        "status": (status if status in status_table("dataset")
                   else DEFAULT_DATASET_STATUS),
        "tags": normalize_tags(tags),
    }
    if sources:
        prov["sources"] = sources
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
        "status": DEFAULT_MODEL_STATUS,
        "tags": [],
    }
    write_provenance(Path(run_dir), prov)
    return prov
