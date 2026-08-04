# =============================================================================
# ✂️ クロップ生成（2 段階目のアノテーション素材を作る）
#
#   BBOX で対象を見つけ → 周辺を切り出して CVAT へ → セグメンテーションを付ける、
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
        "BBOX の検出結果をもとに、対象ごとの切り出しを作ります。"
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
            "切り出す対象のクラス（空なら全部）", _classes, key="cr_cls",
            help="複数クラスのモデルでは、切り出したいクラスだけを選んでください") if _classes else []
    with _c4:
        st.session_state.setdefault("cr_conf", 0.25)
        _conf = st.slider("conf しきい値", 0.05, 0.95, step=0.05, key="cr_conf",
                          help="低いほど拾いますが、誤検出のクロップも増えます")

    if st.button("🔍 対象を検出する", type="primary", use_container_width=True,
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
        st.caption("「🔍 対象を検出する」を押すと切り出す対象を探します。")
        return

    _bg = st.session_state.get("cr_bg") or []
    _n_obj = sum(len(v) for v in _found.values())
    metric_row([
        ("対象の写る画像", len(_found)),
        ("検出した対象", _n_obj),
        ("対象の出ない画像", len(_bg)),
    ])
    if not _found:
        st.warning("対象が 1 つも検出されませんでした。conf を下げてみてください。")

    # ── ② 切り出しの決め方 ───────────────────────────────────────────────
    #     key= を付けたウィジェットには value= を渡さない。
    #     両方渡すとどちらが正か状況で変わり、値を変えても反映されなくなる
    #     （ui/widgets.py も同じ理由で使い分けている）。既定値は先に入れておく。
    for _k, _v in {
        "cr_ann": 2.0, "cr_tgt": 1.5, "cr_basis": "long_side",
        "cr_out": 512, "cr_prefer_out": True, "cr_maxup": 1.5,
        "cr_square": True, "cr_pad": "reflect", "cr_fmt": "png",
        "cr_dedup": 0.0, "cr_prev_n": 3, "cr_tile": 1024, "cr_ovl": 0.0,
        "cr_bg_prev_n": 4,
    }.items():
        st.session_state.setdefault(_k, _v)

    st.markdown("#### ② 切り出し方を決める")
    _s1, _s2, _s3 = st.columns(3)
    with _s1:
        _ann_scale = st.number_input(
            "annotation_scale", 1.0, 6.0, step=0.1, key="cr_ann",
            help="アノテーション用に大きめに切る倍率。あとで内側へ切り直せます")
    with _s2:
        _tgt_scale = st.number_input(
            "target_scale", 1.0, 6.0, step=0.1, key="cr_tgt",
            help="学習・実機で使う想定倍率。ここでは切らず、メタに記録します")
    with _s3:
        _basis = st.selectbox(
            "倍率の基準", list(SCALE_BASIS),
            format_func=lambda k: f"{k}（{SCALE_BASIS[k]}）", key="cr_basis")

    _o1, _o2, _o3 = st.columns(3)
    with _o1:
        _out_size = st.select_slider(
            "出力の一辺（学習で使うサイズ）",
            options=[256, 384, 512, 640, 768, 1024, 1280], key="cr_out")
    with _o2:
        _prefer_out = st.checkbox(
            "出力サイズを優先する", key="cr_prefer_out",
            help="拡大の上限を外し、必ず指定したサイズで書き出します。"
                 "学習時のサイズをそろえたいときはこちら")
    with _o3:
        _square = st.checkbox("正方形にする", key="cr_square")

    # 検出済みの対象から、いまの設定で何倍に引き伸ばされるかを実測して出す。
    # データセットによって対象の写る大きさが違うので、既定値を当てにしない
    _longs = sorted(max(b["bbox_xyxy"][2] - b["bbox_xyxy"][0],
                        b["bbox_xyxy"][3] - b["bbox_xyxy"][1])
                    for v in _found.values() for b in v)
    if _longs:
        _med_crop = _longs[len(_longs) // 2] * float(_ann_scale)
        _ratio = int(_out_size) / max(1e-6, _med_crop)
        _big = sum(1 for l in _longs
                   if int(_out_size) / (l * float(_ann_scale)) > 3.0)
        metric_row([
            ("bbox 長辺の中央値", f"{_longs[len(_longs) // 2]:.0f} px"),
            ("切り出しの中央値", f"{_med_crop:.0f} px"),
            ("出力までの倍率", f"×{_ratio:.2f}"),
            ("3倍超の拡大", f"{_big} / {len(_longs)} 件"),
        ])
        if _ratio > 3.0:
            st.warning(
                f"⚠ 中央値で **×{_ratio:.1f} の引き伸ばし**になります。"
                "拡大しても細かさは増えないので、ファイルが大きくなるだけです。"
                f"出力の一辺を {int(_med_crop // 64 * 64) or 256} 前後まで下げるか、"
                "「出力サイズを優先する」を外すことを検討してください。"
            )

    if _prefer_out:
        _max_up = 0.0          # 上限なし
        st.caption(
            "ℹ 指定したサイズで必ず書き出します（学習の入力サイズがそろいます）。"
            "小さな対象は引き伸ばされますが、実効解像度は上がりません。"
        )
    else:
        _max_up = st.number_input(
            "max_upscale（これを超える拡大はしない）", 1.0, 8.0, step=0.1,
            key="cr_maxup",
            help="小さい対象の水増しを防ぎます。上限に当たったクロップは"
                 "指定より小さい出力になります")

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
            "重複除去", 0.0, 1.0, step=0.05, key="cr_dedup",
            help="0 なら検出した対象をすべて出します。密集して重複が多いときだけ上げます")

    # ── ③ 下見 ──────────────────────────────────────────────────────────
    st.markdown("#### ③ 切り出しを確認する")
    _pv1, _pv2 = st.columns([1, 3])
    with _pv1:
        _prev_n = st.select_slider(
            "表示枚数", options=[1, 3, 6, 9, 12], key="cr_prev_n",
            help="増やすと描画に時間がかかります")
    with _pv2:
        st.markdown(
            '<div style="padding-top:28px; color:var(--text-muted); font-size:.82rem;">'
            '設定を変えるとその場で作り直されます</div>', unsafe_allow_html=True)

    if _found and st.button("👁 切り出しを試す", use_container_width=True,
                            key="cr_preview"):
        st.session_state["cr_prev"] = True

    if st.session_state.get("cr_prev") and _found:
        import cv2
        _shown = 0
        for _ip, _dets in sorted(_found.items()):
            if _shown >= int(_prev_n):
                break
            _img = cv2.imread(_ip)
            if _img is None:
                continue
            for _det in _dets:
                if _shown >= int(_prev_n):
                    break
                _crop, _g = make_crop(
                    _img, _det["bbox_xyxy"], scale=float(_ann_scale),
                    scale_basis=_basis, square=_square, pad_mode=_pad_mode,
                    out_size=int(_out_size), max_upscale=float(_max_up))
                _tr = target_rect_in_crop(_g, float(_tgt_scale))
                # target 範囲を枠で示す（学習時に切り直す位置）
                _vis = _crop.copy()
                cv2.rectangle(_vis, (int(_tr[0]), int(_tr[1])),
                              (int(_tr[0] + _tr[2]), int(_tr[1] + _tr[3])),
                              (244, 207, 78), max(2, int(_out_size / 250)))
                _shown += 1
                st.caption(
                    f"`{Path(_ip).name}` — 切り出し {_g['crop_rect_in_source'][2]}px "
                    f"→ 出力 {_g['output_size'][0]}×{_g['output_size'][1]}px"
                    + ("　⚠ 拡大の上限に当たり、指定より小さくなっています"
                       if _g["max_upscale_applied"] else "")
                )
                st.image(cv2.cvtColor(_vis, cv2.COLOR_BGR2RGB),
                         caption=f"青枠 = target_scale {_tgt_scale} の範囲",
                         use_column_width=True)

    # ── ④ 書き出し ──────────────────────────────────────────────────────
    st.markdown("#### ④ 書き出す")
    st.session_state.setdefault("cr_outname",
                                f"crops_{datetime.now():%Y%m%d_%H%M}")
    _out_name = st.text_input("出力先（data/ 配下の名前）", key="cr_outname")
    _out_dir = DATA_DIR / _out_name.strip() if _out_name.strip() else None
    if _out_dir is None:
        st.warning("出力先の名前を入れてください。")
        return
    st.caption(f"書き出し先: `{_out_dir}`")

    st.session_state.setdefault("cr_bgon", False)
    _bg_on = st.checkbox(
        f"対象の出ない画像を背景タイルにする（{len(_bg)} 枚）", key="cr_bgon",
        help="空タイルの自動除去はしません。採否は人が決めてください",
        disabled=not _bg)
    _tile, _overlap = int(_out_size), 0.0
    if _bg_on:
        _bg1, _bg2, _bg3 = st.columns(3)
        with _bg1:
            _tile = st.number_input("タイルの一辺", 128, 2048, step=64,
                                    key="cr_tile")
        with _bg2:
            _overlap = st.slider("重なり", 0.0, 0.5, step=0.05, key="cr_ovl")
        with _bg3:
            _bg_prev_n = st.select_slider(
                "プレビュー枚数", options=[0, 2, 4, 6, 9], key="cr_bg_prev_n")

        # 何枚に割れるかを先に見せる（作ってから多すぎたと気づくのを避ける）
        if int(_bg_prev_n) > 0:
            import cv2
            _step = max(1, int(_tile * (1.0 - min(0.5, _overlap))))
            _sample = cv2.imread(_bg[0])
            if _sample is not None:
                _sh, _sw = _sample.shape[:2]
                _rows = len([y for y in range(0, max(1, _sh - 1), _step)
                             if min(_sh, y + _tile) - y >= _tile // 2])
                _cols = len([x for x in range(0, max(1, _sw - 1), _step)
                             if min(_sw, x + _tile) - x >= _tile // 2])
                st.caption(
                    f"1 枚あたり {_rows}×{_cols} = {_rows * _cols} タイル"
                    f"　→ 全体で約 {_rows * _cols * len(_bg)} タイル"
                )
                _shown = 0
                _grid = st.columns(min(3, int(_bg_prev_n)))
                for _r in range(_rows):
                    for _c in range(_cols):
                        if _shown >= int(_bg_prev_n):
                            break
                        _y, _x = _r * _step, _c * _step
                        _t = _sample[_y:min(_sh, _y + _tile),
                                     _x:min(_sw, _x + _tile)]
                        if _t.size:
                            _grid[_shown % len(_grid)].image(
                                cv2.cvtColor(_t, cv2.COLOR_BGR2RGB),
                                caption=f"r{_r}_c{_c}", use_column_width=True)
                            _shown += 1
                st.caption(
                    "中身が薄く見えるタイルも自動では捨てません"
                    "（「何も無い」の判定は対象によって変わるため）。採否は人が決めてください。"
                )

    if st.button(f"✂️ {_n_obj} 件のクロップを作る", type="primary",
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
