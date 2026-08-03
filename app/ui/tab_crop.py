# =============================================================================
# ✂️ クロップ生成（2 段階目のアノテーション素材を作る）
#
#   BBOX で果実を見つけ → 周辺を切り出して CVAT へ → セグメンテーションを付ける、
#   という流れの前段。切り出しの核（make_crop）は実機と共通なので、
#   ここで決めた倍率・基準は実機側とそろえること。
# =============================================================================
from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

import streamlit as st

from core import *  # noqa: F401,F403
from core import _find_image_dirs  # noqa: F401
from .widgets import empty_state, metric_row, show_error


def _model_meta(model_path: Path) -> dict:
    """メタに残すモデルの素性。あとでどの重みで作ったか辿れるようにする。"""
    info = {"model_id": model_path.stem, "weights_path": str(model_path)}
    prov = read_provenance(model_path.parent.parent
                           if model_path.parent.name == "weights"
                           else model_path.parent)
    if prov:
        info["imgsz"] = (prov.get("params") or {}).get("imgsz")
        info["trained_at"] = prov.get("trained_at")
    meta = read_model_meta(model_path)
    if meta and meta.get("ok"):
        info["classes"] = meta.get("names") or []
    info["inferred_at"] = datetime.now().astimezone().isoformat()
    return info


def render_crop() -> None:
    st.markdown(
        '<div class="section-head"><h3>✂️ クロップ生成</h3></div>',
        unsafe_allow_html=True)
    st.caption(
        "BBOX の検出結果をもとに果実ごとの切り出しを作ります。"
        "CVAT に入れてセグメンテーションを付けると、2 段階目のモデルを学習できます。"
    )
    st.info(
        "ℹ 切り出しの規則（倍率・基準・パディング）は**実機と共通の処理**を使います。"
        "ここで決めた値は実機側の設定とそろえてください。ずれると精度が落ちます。"
    )

    _imgs_dirs = _find_image_dirs(DATA_DIR)
    _models = sorted(MODELS_DIR.rglob("*.pt"))
    if not _imgs_dirs:
        empty_state("元画像が見つかりません",
                    "「📤 Step2: データ取込」でデータを入れてください。")
        return
    if not _models:
        empty_state("BBOX 検出モデルがありません",
                    "「🚀 Step3: モデル学習」で学習するか、"
                    "「📁 データ管理」から取り込んでください。")
        return

    # ── ① 入力 ──────────────────────────────────────────────────────────
    st.markdown("#### ① 元画像とモデルを選ぶ")
    _c1, _c2 = st.columns(2)
    with _c1:
        _dir_labels = [str(d.relative_to(DATA_DIR)) for d in _imgs_dirs]
        _dir_sel = st.selectbox("元画像のディレクトリ", _dir_labels, key="cr_dir")
        _src_dir = DATA_DIR / _dir_sel
    with _c2:
        _mmap = {str(p.relative_to(MODELS_DIR)): p for p in _models}
        _msel = st.selectbox("BBOX 検出モデル", list(_mmap), key="cr_model")
        _model = _mmap[_msel]

    _meta = read_model_meta(_model)
    _classes = (_meta or {}).get("names") or []
    _c3, _c4 = st.columns(2)
    with _c3:
        _targets = st.multiselect(
            "果実として扱うクラス（空なら全部）", _classes, key="cr_cls",
            help="複数クラスのモデルでは、果実だけを選んでください") if _classes else []
    with _c4:
        _conf = st.slider("conf しきい値", 0.05, 0.95, 0.25, 0.05, key="cr_conf",
                          help="低いほど拾いますが、誤検出のクロップも増えます")

    if st.button("🔍 果実を検出する", type="primary", use_container_width=True,
                 key="cr_detect"):
        _tmp = PREDICTIONS_DIR / "_crop_scan"
        if _tmp.exists():
            shutil.rmtree(_tmp)
        _tmp.mkdir(parents=True, exist_ok=True)
        with st.spinner("検出しています…"):
            _saved = run_inference(str(_model), _src_dir, _tmp,
                                   conf_threshold=float(_conf))
        _found: dict = {}
        _bg: list[str] = []
        for _jf in _saved:
            import json as _json
            try:
                _d = _json.loads(Path(_jf).read_text())
            except Exception:
                continue
            _ip = _d.get("image_path")
            if not _ip:
                continue
            _boxes = [
                {"bbox_xyxy": [float(v) for v in b["bbox_xyxy"]],
                 "confidence": float(b.get("confidence", 0.0)),
                 "label": b.get("label", ""),
                 "class_id": None}
                for b in (_d.get("boxes") or [])
                if len(b.get("bbox_xyxy") or []) == 4
                and (not _targets or b.get("label") in _targets)
            ]
            if _boxes:
                _found[_ip] = _boxes
            else:
                _bg.append(_ip)
        st.session_state["cr_found"] = _found
        st.session_state["cr_bg"] = _bg

    _found = st.session_state.get("cr_found")
    if _found is None:
        st.caption("「🔍 果実を検出する」を押すと対象を探します。")
        return

    _bg = st.session_state.get("cr_bg") or []
    _n_fruit = sum(len(v) for v in _found.values())
    metric_row([
        ("果実の写る画像", len(_found)),
        ("検出した果実", _n_fruit),
        ("果実の出ない画像", len(_bg)),
    ])
    if not _found:
        st.warning("果実が 1 つも検出されませんでした。conf を下げてみてください。")

    # ── ② 切り出しの決め方 ───────────────────────────────────────────────
    st.markdown("#### ② 切り出し方を決める")
    _s1, _s2, _s3 = st.columns(3)
    with _s1:
        _ann_scale = st.number_input(
            "annotation_scale", 1.0, 6.0, 2.0, 0.1, key="cr_ann",
            help="アノテーション用に大きめに切る倍率。あとで内側へ切り直せます")
    with _s2:
        _tgt_scale = st.number_input(
            "target_scale", 1.0, 6.0, 1.5, 0.1, key="cr_tgt",
            help="学習・実機で使う想定倍率。ここでは切らず、メタに記録します")
    with _s3:
        _basis = st.selectbox(
            "倍率の基準", list(SCALE_BASIS),
            format_func=lambda k: f"{k}（{SCALE_BASIS[k]}）", key="cr_basis")

    _o1, _o2, _o3 = st.columns(3)
    with _o1:
        _out_size = st.select_slider(
            "出力の一辺", options=[256, 384, 512, 640, 768, 1024, 1280],
            value=1024, key="cr_out")
    with _o2:
        _max_up = st.number_input(
            "max_upscale", 1.0, 4.0, 1.5, 0.1, key="cr_maxup",
            help="これを超える拡大はしません（小さい果実の水増しを防ぐ）")
    with _o3:
        _square = st.checkbox("正方形にする", value=True, key="cr_square")

    _p1, _p2, _p3 = st.columns(3)
    with _p1:
        _pad_mode = st.selectbox(
            "端のはみ出し", list(PAD_MODES),
            format_func=lambda k: PAD_MODES[k], key="cr_pad")
    with _p2:
        _fmt = st.selectbox("出力形式", ["png", "jpg"], key="cr_fmt",
                            help="png を推奨。可逆なので細部が保てます")
    with _p3:
        _dedup = st.slider(
            "重複除去", 0.0, 1.0, 0.0, 0.05, key="cr_dedup",
            help="0 なら全果実を出します。密集して重複が多いときだけ上げます")

    # ── ③ 下見 ──────────────────────────────────────────────────────────
    st.markdown("#### ③ 切り出しを確認する")
    if _found and st.button("👁 3 枚だけ試す", use_container_width=True,
                            key="cr_preview"):
        st.session_state["cr_prev"] = True

    if st.session_state.get("cr_prev") and _found:
        import cv2
        _shown = 0
        for _ip, _dets in sorted(_found.items()):
            if _shown >= 3:
                break
            _img = cv2.imread(_ip)
            if _img is None:
                continue
            _crop, _g = make_crop(
                _img, _dets[0]["bbox_xyxy"], scale=float(_ann_scale),
                scale_basis=_basis, square=_square, pad_mode=_pad_mode,
                out_size=int(_out_size), max_upscale=float(_max_up))
            _tr = target_rect_in_crop(_g, float(_tgt_scale))
            # target 範囲を枠で示す（学習時に切り直す位置）
            _vis = _crop.copy()
            cv2.rectangle(_vis, (int(_tr[0]), int(_tr[1])),
                          (int(_tr[0] + _tr[2]), int(_tr[1] + _tr[3])),
                          (78, 207, 244), max(2, int(_out_size / 250)))
            st.caption(
                f"`{Path(_ip).name}` — 出力 {_g['output_size'][0]}×{_g['output_size'][1]}px"
                + ("　⚠ 拡大の上限に当たっています" if _g["max_upscale_applied"] else "")
            )
            st.image(cv2.cvtColor(_vis, cv2.COLOR_BGR2RGB),
                     caption=f"青枠 = target_scale {_tgt_scale} の範囲",
                     use_column_width=True)
            _shown += 1

    # ── ④ 書き出し ──────────────────────────────────────────────────────
    st.markdown("#### ④ 書き出す")
    _out_name = st.text_input(
        "出力先（data/ 配下の名前）",
        value=f"crops_{datetime.now():%Y%m%d_%H%M}", key="cr_outname")
    _out_dir = DATA_DIR / _out_name.strip() if _out_name.strip() else None
    if _out_dir is None:
        st.warning("出力先の名前を入れてください。")
        return
    st.caption(f"書き出し先: `{_out_dir}`")

    _bg_on = st.checkbox(
        f"果実の出ない画像を背景タイルにする（{len(_bg)} 枚）",
        value=False, key="cr_bgon",
        help="空タイルの自動除去はしません。採否は人が決めてください",
        disabled=not _bg)
    if _bg_on:
        _bg1, _bg2 = st.columns(2)
        with _bg1:
            _tile = st.number_input("タイルの一辺", 128, 2048, int(_out_size),
                                    64, key="cr_tile")
        with _bg2:
            _overlap = st.slider("重なり", 0.0, 0.5, 0.0, 0.05, key="cr_ovl")

    if st.button(f"✂️ {_n_fruit} 件のクロップを作る", type="primary",
                 use_container_width=True, key="cr_run",
                 disabled=not _found):
        _bar = st.progress(0.0, text="切り出しています…")

        def _prog(done, total):
            if total:
                _bar.progress(min(done / total, 1.0), text=f"{done} / {total} 枚")

        _res = generate_crops(
            _found, _out_dir,
            annotation_scale=float(_ann_scale), target_scale=float(_tgt_scale),
            scale_basis=_basis, square=_square, pad_mode=_pad_mode,
            out_size=int(_out_size), max_upscale=float(_max_up),
            dedup_center_dist=float(_dedup), out_format=_fmt,
            model_info=_model_meta(_model), on_progress=_prog)
        _bar.empty()

        if _res["crops"]:
            st.success(
                f"✅ {_res['images']} 枚から {_res['crops']} 件のクロップを作りました"
                f" → `{_out_dir.name}`")
            if _res["upscale_limited"]:
                st.info(f"ℹ {_res['upscale_limited']} 件は拡大の上限に当たり、"
                        "指定より小さい出力になっています（水増しを避けるため）。")
            if _res["rejected"]:
                st.caption(f"重複除去で {_res['rejected']} 件を除きました。")
            st.caption(
                "`images/` `meta/` `manifest.jsonl` ができています。"
                "`images/` を CVAT のタスクにしてセグメンテーションを付けてください。"
            )
        else:
            show_error(_res["error"], prefix="❌ 作れませんでした: ")

        for _p, _why in _res["skipped"][:20]:
            st.caption(f"・{Path(_p).name} — {_why}")

        if _bg_on and _bg:
            _b = generate_background_tiles(
                _bg, _out_dir / "background", tile_size=int(_tile),
                tile_overlap=float(_overlap), out_format=_fmt)
            if _b["ok"]:
                st.success(f"✅ 背景タイルを {_b['tiles']} 枚作りました")
            else:
                show_error(_b["error"], prefix="⚠ 背景タイル: ")

        if _res["crops"]:
            record_dataset_provenance(
                _out_dir, source="crop", task_type="segment",
                labels=_targets or _classes,
                status="draft",
                tags=["クロップ", "2段階目"],
                extra={
                    "crop": {
                        "source_dir": str(_src_dir),
                        "model": str(_model),
                        "annotation_scale": float(_ann_scale),
                        "target_scale": float(_tgt_scale),
                        "scale_basis": _basis,
                        "out_size": int(_out_size),
                        "max_upscale": float(_max_up),
                        "conf_threshold": float(_conf),
                        "crops": _res["crops"],
                    }
                })
            st.session_state.pop("cr_prev", None)
