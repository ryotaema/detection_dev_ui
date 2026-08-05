# =============================================================================
# 学習パラメータのプリセット
# =============================================================================
from __future__ import annotations

import json

import streamlit as st

from core import MODELS_DIR, _MODEL_OPTS


_USER_PRESETS_FILE = MODELS_DIR / ".user_presets.json"

_BUILTIN_PRESETS: dict[str, dict] = {
    "🚫 ノーマル (augなし · yolo11s · 100ep · 640px)": {
        "model": "yolo11s", "epochs": 100, "batch": 8,
        "imgsz": 640, "patience": 50, "optimizer": "auto",
        "lr0": 0.01, "cos_lr": False,
        "warmup_epochs": 3, "dropout": 0.0, "weight_decay": 0.0005, "workers": 8,
        "degrees": 0.0, "scale": 0.0, "fliplr": 0.0, "flipud": 0.0,
        "translate": 0.0, "perspective": 0.0,
        "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0,
        "mosaic": 0.0, "mixup": 0.0, "erasing": 0.0, "close_mosaic": 0,
    },
    "⚡ 速度優先 (yolo11n · 50ep · 640px)": {
        "model": "yolo11n", "epochs": 50, "batch": 16,
        "imgsz": 640, "patience": 20, "optimizer": "SGD",
        "lr0": 0.01, "cos_lr": False,
        "mosaic": 0.5, "close_mosaic": 5, "scale": 0.5, "fliplr": 0.5,
    },
    "⚖️ バランス型 (yolo11s · 100ep · 640px)": {
        "model": "yolo11s", "epochs": 100, "batch": 16,
        "imgsz": 640, "patience": 30, "optimizer": "auto",
        "lr0": 0.01, "cos_lr": False,
        "mosaic": 1.0, "close_mosaic": 10, "scale": 0.5, "fliplr": 0.5,
    },
    "🎯 精度優先 (yolo11l · 200ep · 640px)": {
        "model": "yolo11l", "epochs": 200, "batch": 8,
        "imgsz": 640, "patience": 50, "optimizer": "AdamW",
        "lr0": 0.001, "cos_lr": True,
        "mosaic": 1.0, "close_mosaic": 15, "scale": 0.5, "fliplr": 0.5,
    },
    "🔍 小物体向け (yolo11m · 150ep · 640px)": {
        "model": "yolo11m", "epochs": 150, "batch": 8,
        "imgsz": 640, "patience": 30, "optimizer": "AdamW",
        "lr0": 0.001, "cos_lr": True,
        "mosaic": 1.0, "close_mosaic": 10, "scale": 0.3, "fliplr": 0.5,
    },
    "🤖 ロボット視点 (yolo11x · 2000ep · 640px)": {
        "model": "yolo11x", "epochs": 2000, "batch": 8,
        "imgsz": 640, "patience": 50, "optimizer": "auto",
        "lr0": 0.001, "cos_lr": True,
        "warmup_epochs": 10, "dropout": 0.1, "weight_decay": 0.0005, "workers": 8,
        "degrees": 60.0, "scale": 0.5, "fliplr": 0.5, "flipud": 0.1,
        "translate": 0.2, "perspective": 0.0005,
        "hsv_h": 0.02, "hsv_s": 0.7, "hsv_v": 0.7,
        "mosaic": 1.0, "mixup": 0.15, "erasing": 0.2, "close_mosaic": 30,
    },
}
def _load_user_presets() -> dict:
    try:
        if _USER_PRESETS_FILE.exists():
            return json.loads(_USER_PRESETS_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _save_user_presets(presets: dict) -> None:
    _USER_PRESETS_FILE.parent.mkdir(parents=True, exist_ok=True)
    _USER_PRESETS_FILE.write_text(
        json.dumps(presets, ensure_ascii=False, indent=2), encoding="utf-8"
    )


_PRESET_KEYS: dict[str, tuple] = {
    # (session_state_key, type, default)
    "model":        ("tp_model",        str,   "yolo11s"),
    "epochs":       ("tp_epochs",       int,   100),
    "batch":        ("tp_batch",        int,   8),
    "imgsz":        ("tp_imgsz",        int,   640),
    "patience":     ("tp_patience",     int,   50),
    "optimizer":    ("tp_optimizer",    str,   "auto"),
    "lr0":          ("tp_lr0",          float, 0.01),
    "cos_lr":       ("tp_cos_lr",       bool,  False),
    "warmup_epochs":("tp_warmup_epochs",int,   3),
    "dropout":      ("tp_dropout",      float, 0.0),
    "weight_decay": ("tp_weight_decay", float, 0.0005),
    "workers":      ("tp_workers",      int,   8),
    "degrees":      ("tp_degrees",      float, 0.0),
    "scale":        ("tp_scale",        float, 0.5),
    "fliplr":       ("tp_fliplr",       float, 0.5),
    "flipud":       ("tp_flipud",       float, 0.0),
    "translate":    ("tp_translate",    float, 0.1),
    "perspective":  ("tp_perspective",  float, 0.0),
    "hsv_h":        ("tp_hsv_h",        float, 0.015),
    "hsv_s":        ("tp_hsv_s",        float, 0.7),
    "hsv_v":        ("tp_hsv_v",        float, 0.4),
    "mosaic":       ("tp_mosaic",       float, 1.0),
    "mixup":        ("tp_mixup",        float, 0.0),
    "erasing":      ("tp_erasing",      float, 0.4),
    "close_mosaic": ("tp_close_mosaic", int,   10),
}

# _PRESET_KEYS のデフォルト値を session_state に事前登録（ウィジェット初回表示用）
for _pk, (_pk_ss, _pk_typ, _pk_def) in _PRESET_KEYS.items():
    if _pk_ss not in st.session_state:
        st.session_state[_pk_ss] = _pk_def


def _apply_preset(params: dict) -> None:
    """プリセット値をセッションステート（widget key）に書き込む。"""
    for k, v in params.items():
        if k not in _PRESET_KEYS:
            continue
        ss_key, typ, _ = _PRESET_KEYS[k]
        if k == "model":
            if v in _MODEL_OPTS:
                st.session_state[ss_key] = v
        else:
            st.session_state[ss_key] = typ(v)


def _collect_current_params() -> dict:
    """現在のウィジェット値からプリセット保存用 dict を生成する。"""
    result = {}
    for k, (ss_key, typ, default) in _PRESET_KEYS.items():
        result[k] = typ(st.session_state.get(ss_key, default))
    return result


# ===========================================================================
# UI レイアウト
# ===========================================================================

# ---------------------------------------------------------------------------
# エラー表示（原因の推定と対処をセットで出す）
# ---------------------------------------------------------------------------

