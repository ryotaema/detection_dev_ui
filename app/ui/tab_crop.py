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
from typing import Optional

import streamlit as st

from core import *  # noqa: F401,F403
from core import _find_image_dirs  # noqa: F401
from .widgets import empty_state, metric_row, open_folder, show_error


def _model_meta(model_path: Path, conf: Optional[float] = None) -> dict:
    """メタに残すモデルの素性。あとでどの重みで作ったか辿れるようにする。

    **重みのハッシュを必ず入れる。** モデルを更新しても
    `models/<run>/weights/best.pt` というパスは変わらないので、
    パスだけでは「どの時点の重みで作ったクロップか」を区別できない。
    """
    _run = (model_path.parent.parent if model_path.parent.name == "weights"
            else model_path.parent)
    prov = read_provenance(_run)
    _imgsz = (prov or {}).get("params", {}).get("imgsz") or 640

    info = build_model_info(model_path, infer_input_size=int(_imgsz),
                            conf_threshold=conf)
    if prov:
        info["trained_at"] = prov.get("trained_at") or info.get("trained_at")
        info["dataset"] = (prov.get("dataset") or {}).get("name")
    meta = read_model_meta(model_path)
    if meta and meta.get("ok"):
        info["classes"] = meta.get("names") or []
    return info


def render_crop() -> None:
    """クロップ生成タブ。

    生成と選別は分けて呼ぶ。生成側は「検出していない」等で途中 return するが、
    **選別は検出をやり直さずに開けなければならない**
    （あとから採否を見直すのが本来の使い方のため）。
    """
    _render_generate()
    _render_bg_selection()
    _render_recut()


def _render_generate() -> None:
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

    # クロップ出力の内部（確認画像・未採用タイル・メタ）は元画像ではない。
    # 選べてしまうと、誤って自分の出力を入力にしてしまう（実際に起きた）
    _imgs_dirs = [d for d in _find_image_dirs(DATA_DIR)
                  if not {"debug", "_unused", "contact_sheet", "meta",
                          "_backup_original"} & set(d.parts)]
    _models = model_weight_files()
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
        # 名前順に並べると、1 エポックだけ回した試し撃ちのモデルが
        # 先頭に来てしまい、既定のまま押すと 1 件も検出しない（実際に起きた）。
        # 精度のあるものを先に出す。
        _ranked = sort_models(_models, how="recommended")
        _mmap, _labels = {}, []
        for _it in _ranked:
            _k = str(_it["key"])
            _mmap[_k] = MODELS_DIR / _k
            _m = _it.get("map")
            _labels.append(
                f"{_k}　{'⭐ ' if _it.get('favorite') else ''}"
                + (f"mAP50-95 {_m:.3f}" if _m is not None else "評価まだ"))
        _lab2key = dict(zip(_labels, _mmap))
        _msel_lab = st.selectbox("BBOX 検出モデル", _labels, key="cr_model")
        _msel = _lab2key[_msel_lab]
        _model = _mmap[_msel]
        if _ranked and _ranked[[l for l in _labels].index(_msel_lab)].get("map") is None:
            st.caption("⚠ このモデルはまだ評価していません。"
                       "検出が 0 件なら、精度のあるモデルを選び直してください。")

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
            record_use(_model, "crop")
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

        _bg6, _bg7 = st.columns([2, 1])
        with _bg6:
            st.session_state.setdefault("cr_bg_how", "even")
            _bg_how = st.radio(
                "選び方", list(BG_SAMPLING), horizontal=True, key="cr_bg_how",
                format_func=lambda k: BG_SAMPLING[k],
                help="「元画像ごとに均等」は、同じ場所を写した 1 枚から"
                     "大量に採ってしまうのを防ぎます")
        with _bg7:
            st.session_state.setdefault("cr_bg_seed", 0)
            _bg_seed = st.number_input(
                "抽選の種", 0, 9999, step=1, key="cr_bg_seed",
                help="同じ種なら毎回同じ選び方になります。"
                     "選び直したいときは数字を変えてください")

        _bg4, _bg5 = st.columns([2, 1])
        with _bg4:
            st.session_state.setdefault("cr_bg_ratio", 0.15)
            _bg_ratio = st.slider(
                "背景がデータセットに占める割合", 0.0, 0.5, step=0.05,
                key="cr_bg_ratio",
                help="全タイルを作ったうえで、この割合になるぶんだけ採用します。"
                     "背景ばかり増えると対象の学習に使える割合が下がります。"
                     "0 にすると全部採用します")
        with _bg5:
            st.session_state.setdefault("cr_bg_keep", True)
            _bg_keep = st.checkbox("採らなかったぶんも残す", key="cr_bg_keep",
                                   help="`background/_unused/` に置きます。"
                                        "あとから足せます")

        # 何枚採ることになるかを先に出す
        _n_obj_est = sum(len(v) for v in (_found or {}).values())
        _n_bg_want = target_tile_count(_n_obj_est, float(_bg_ratio))
        st.caption(
            f"対象クロップ {_n_obj_est} 件に対し、背景は "
            + (f"**{_n_bg_want} 枚**を{BG_SAMPLING[_bg_how]}で採用します（残りは "
               + ("`_unused/` へ" if _bg_keep else "破棄") + "）"
               if _bg_ratio > 0 else "**全タイルを採用**します")
            + "　作ったあとで下の「⑤ 背景タイルを選別する」から人の目で直せます。")

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

    _dbg = st.checkbox(
        "🔎 確認用の重ね描きも出す（debug/）", key="cr_debug",
        help="切り出しが意図どおりか（対象が中心にいるか、余白は適切か）を"
             "目で確かめるための画像を数枚だけ出します。"
             "緑=検出した対象　青=学習で使う内側の範囲　灰=写り込んだ他の対象")

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
            model_info=_model_meta(_model, conf=float(_conf)),
            debug_overlay=bool(_dbg), seed=0, on_progress=_prog)
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
            open_folder(_out_dir, "crop_out", "📂 書き出し先を開く")
        else:
            show_error(_res["error"], prefix="❌ 作れませんでした: ")

        for _p, _why in _res["skipped"][:20]:
            st.caption(f"・{Path(_p).name} — {_why}")

        _b = None
        if _bg_on and _bg:
            _b = generate_background_tiles(
                _bg, _out_dir / "background", tile_size=int(_tile),
                tile_overlap=float(_overlap), out_format=_fmt,
                background_ratio=float(_bg_ratio),
                bg_sampling=("all" if float(_bg_ratio) <= 0 else _bg_how),
                keep_unused_tiles=bool(_bg_keep),
                n_object_crops=int(_res.get("crops", 0)), seed=int(_bg_seed))
            if _b["ok"]:
                st.success(
                    f"✅ 背景タイルを {_b['generated']} 枚作り、"
                    f"うち {_b['tiles']} 枚を採用しました"
                    + (f"（{_b['unused']} 枚は `background/_unused/` に残しています）"
                       if _b.get("unused") else ""))
            else:
                show_error(_b["error"], prefix="⚠ 背景タイル: ")

        # 何をどう作ったかを出力先に残す（画面の表示は次の操作で消えるため）。
        # **失敗した実行では書かない。** 出力先が既にある等で 0 枚に終わった回が
        # 前回の成功時の記録を上書きすると、「13660 枚あるのにログは 0 枚」
        # という食い違いが起きる（実際に起きた）。
        _params = {
            "source_dir": str(_src_dir), "model": str(_model),
            "conf_threshold": float(_conf),
            "annotation_scale": float(_ann_scale),
            "target_scale": float(_tgt_scale),
            "scale_basis": _basis, "square": bool(_square),
            "pad_mode": _pad_mode, "out_size": int(_out_size),
            "max_upscale": float(_max_up), "out_format": _fmt,
            "dedup_center_dist": float(_dedup),
        }
        _cfg = None
        if _res.get("crops"):
            write_crop_log(_out_dir, _res, _params, _b)
            _cfg = write_crop_config(_out_dir, _params)
        if _cfg:
            st.caption(
                f"📄 `crop_log.txt` と `{CROP_CONFIG}` を書き出しました。"
                "実機側は次のように読み込むと、切り出しの規則が確実にそろいます:"
            )
            st.code(
                "from core.crop import make_crop, load_crop_config\n"
                f"cfg = load_crop_config(r\"{_out_dir}\")\n"
                "crop, geom = make_crop(frame, bbox_xyxy, **cfg)",
                language="python")

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



def _bg_dirs() -> list:
    """背景タイルを持つ出力先を新しい順に返す"""
    return sorted((p.parent for p in DATA_DIR.glob("*/background/manifest.jsonl")),
                  key=lambda p: p.stat().st_mtime, reverse=True)


def _render_bg_selection() -> None:
    import cv2

    st.markdown("#### ⑤ 背景タイルを選別する")
    _dirs = _bg_dirs()
    if not _dirs:
        st.caption("背景タイルを作ると、ここで採否を見直せます。")
        return

    _sel = st.selectbox(
        "見直す出力先", [str(d.parent.relative_to(DATA_DIR)) for d in _dirs],
        key="bgsel_dir")
    _bgdir = DATA_DIR / _sel / "background"
    for _k, _v in {"cs_cols": 6, "cs_rows": 6, "cs_thumb": 140}.items():
        st.session_state.setdefault(_k, _v)
    _state = read_tile_selection(_bgdir)
    if not _state:
        st.caption("タイルがありません。")
        return

    _adopted = [k for k, v in _state.items() if v]
    _unused = [k for k, v in _state.items() if not v]
    metric_row([("全タイル", len(_state)), ("採用", len(_adopted)),
                ("未採用", len(_unused))])

    _c1, _c2, _c3 = st.columns([2, 1, 1])
    with _c1:
        _view = st.radio("表示", ["採用ぶん", "未採用ぶん", "すべて"],
                         horizontal=True, key="bgsel_view")
    with _c2:
        _per = st.select_slider("1 ページの枚数", options=[6, 12, 24, 48],
                                key="bgsel_per")
    with _c3:
        _cols_n = st.select_slider("列数", options=[3, 4, 6], key="bgsel_cols")

    _names = sorted(_adopted if _view == "採用ぶん"
                    else _unused if _view == "未採用ぶん" else _state)
    if not _names:
        st.caption("表示するタイルがありません。")
        return

    _pages = (len(_names) - 1) // int(_per) + 1
    _page = st.number_input(f"ページ（全 {_pages}）", 1, _pages, step=1,
                            key="bgsel_page") if _pages > 1 else 1
    _shown = _names[(int(_page) - 1) * int(_per): int(_page) * int(_per)]

    # チェックの初期値は現在の採否。ここで触ったものだけ上書きする
    _edits = st.session_state.setdefault("bgsel_edits", {})
    if st.session_state.get("bgsel_dir_prev") != _sel:
        _edits.clear()
        st.session_state["bgsel_dir_prev"] = _sel

    _cols = st.columns(int(_cols_n))
    for _i, _stem in enumerate(_shown):
        _cur = _edits.get(_stem, _state.get(_stem, False))
        _base = _bgdir if _state.get(_stem) else _bgdir / "_unused"
        _hit = list((_base / "images").glob(f"{_stem}.*"))
        with _cols[_i % int(_cols_n)]:
            if _hit:
                _im = cv2.imread(str(_hit[0]))
                if _im is not None:
                    st.image(cv2.cvtColor(_im, cv2.COLOR_BGR2RGB),
                             use_column_width=True)
            _new = st.checkbox(f"採用　`{_stem[-22:]}`", value=_cur,
                               key=f"bgsel_cb_{_sel}_{_stem}")
            if _new != _state.get(_stem, False):
                _edits[_stem] = _new
            else:
                _edits.pop(_stem, None)

    _changed = len(_edits)
    _a1, _a2 = st.columns([1, 3])
    with _a1:
        _apply = st.button(f"✅ 反映する（{_changed} 件の変更）", key="bgsel_apply",
                           type="primary", use_container_width=True,
                           disabled=not _changed)
    with _a2:
        st.caption("外したタイルは消えず `_unused/` に移るだけなので、"
                   "何度でもやり直せます。")

    if _apply:
        _final = [k for k in _state if _edits.get(k, _state[k])]
        _r = apply_selection(_bgdir, _final)
        if _r["ok"]:
            _edits.clear()
            st.success(f"✅ 採用 {_r['adopted']} 枚 / 未採用 {_r['unused']} 枚に"
                       f"更新しました（{_r['moved']} 枚を移動）")
            st.rerun()
        else:
            show_error(_r["error"], prefix="⚠ 反映できません: ")

    _render_contact_sheet(_bgdir, _state)


def _render_contact_sheet(bgdir, state: dict) -> None:
    """一覧を 1 枚の画像にまとめ、番号で採否を受け取る（仕様 §7.3）。

    画面でスクロールしながら選ぶより、紙に出す・別の端末で見る・
    複数人で確認するときに向く。
    """
    with st.expander("📄 コンタクトシート（番号で控えて採否を渡す）"):
        st.caption(
            "タイルを縮小して 1 枚に並べます。緑枠 = いま採用中、灰枠 = 未採用。"
            "番号を控えて、下の欄に書いてください。"
        )
        _s1, _s2, _s3, _s4 = st.columns(4)
        with _s1:
            _cs_cols = st.select_slider("列数", options=[4, 6, 8, 10],
                                        key="cs_cols")
        with _s2:
            _cs_rows = st.select_slider("1 枚の行数", options=[4, 6, 8, 12],
                                        key="cs_rows")
        with _s3:
            _cs_thumb = st.select_slider("縮小後の一辺", options=[96, 140, 200, 260],
                                         key="cs_thumb")
        with _s4:
            _cs_which = st.radio("並べる対象", ["すべて", "採用ぶん", "未採用ぶん"],
                                 key="cs_which")

        if st.button("📄 シートを作る", key="cs_build", use_container_width=True):
            _names = (None if _cs_which == "すべて"
                      else [k for k, v in state.items()
                            if v == (_cs_which == "採用ぶん")])
            _r = build_contact_sheet(bgdir, _names, cols=int(_cs_cols),
                                     rows=int(_cs_rows), thumb=int(_cs_thumb))
            if _r["ok"]:
                st.success(f"✅ {_r['total']} 枚を {len(_r['sheets'])} 枚の"
                           f"シートにまとめました")
            else:
                show_error(_r["error"], prefix="⚠ シートを作れません: ")

        _sheets = sorted((bgdir / CONTACT_DIR).glob("sheet_*"))
        if _sheets:
            for _f in _sheets:
                st.image(str(_f), use_column_width=True, caption=_f.name)
                st.download_button(f"⬇ {_f.name}", _f.read_bytes(),
                                   file_name=_f.name, key=f"cs_dl_{_f.name}")

        _index = read_contact_index(bgdir)
        if not _index:
            return

        st.markdown("**控えた番号を反映する**")
        _txt = st.text_area(
            "番号（`3` / 範囲 `5-12` / ファイル名 も可）", key="cs_list",
            placeholder="例）1 3 5-12, 20", height=80)
        _mode = st.radio("扱い", ["書いた番号だけを採用にする", "書いた番号を採用から外す"],
                         key="cs_mode", horizontal=True)

        _picked, _bad = parse_selection_list(_txt, _index)
        if _bad:
            st.warning("読み取れなかったもの: " + "　".join(_bad[:20]))
        if _picked:
            _n = (len(_picked) if _mode.startswith("書いた番号だけ")
                  else sum(1 for k, v in state.items() if v and k not in _picked))
            st.caption(f"適用すると採用は **{_n} 枚** になります。")

        if st.button("✅ この内容にする", key="cs_apply", type="primary",
                     disabled=not _picked, use_container_width=True):
            if _mode.startswith("書いた番号だけ"):
                _final = list(_picked)
            else:
                _final = [k for k, v in state.items() if v and k not in _picked]
            _r = apply_selection(bgdir, _final)
            if _r["ok"]:
                st.success(f"✅ 採用 {_r['adopted']} 枚 / 未採用 {_r['unused']} 枚に"
                           f"更新しました")
                st.rerun()
            else:
                show_error(_r["error"], prefix="⚠ 反映できません: ")


def _render_recut() -> None:
    """アノテーション倍率で切ったものを、学習（＝実機）の倍率へ切り直す。

    ここを飛ばすと、学習では対象が画面の 1/2、実機では 2/3 を占めることになる。
    実測では占有率が 0.506 対 0.672 と食い違った。
    """
    st.markdown("#### ⑥ 学習用に切り直す")
    st.caption(
        "アノテーション用は周りが見えるよう大きめ（annotation_scale）に切っています。"
        "**学習は実機と同じ倍率（target_scale）に揃える**必要があるので、"
        "セグメンテーションを付け終えたら、ここで内側へ切り直してください。"
    )

    _dss = sorted([d for d in DATA_DIR.iterdir()
                   if d.is_dir() and (d / "data.yaml").exists()],
                  key=lambda p: p.stat().st_mtime, reverse=True)
    if not _dss:
        st.caption("切り直せるデータセットがありません。")
        return

    _c1, _c2, _c3 = st.columns([3, 1, 1])
    with _c1:
        _src_name = st.selectbox("切り直すデータセット",
                                 [d.name for d in _dss], key="rc_src")
    _src = DATA_DIR / _src_name

    # 作ったときの倍率が残っていれば、それを初期値にする
    _prov = (read_provenance(_src) or {}).get("crop") or {}
    with _c2:
        _from = st.number_input("いまの倍率", 1.0, 6.0, step=0.1, format="%.2f",
                                value=float(_prov.get("annotation_scale", 2.0)),
                                key="rc_from")
    with _c3:
        _to = st.number_input("学習で使う倍率", 1.0, 6.0, step=0.1, format="%.2f",
                              value=float(_prov.get("target_scale", 1.5)),
                              key="rc_to")

    if _prov:
        st.caption(f"　このデータセットは annotation_scale "
                   f"{_prov.get('annotation_scale')} / target_scale "
                   f"{_prov.get('target_scale')} で作られています。")
    else:
        # **クロップ出力以外を切り直すと、素の写真の中央を切り抜くことになる。**
        # 気づきにくいので、注意ではなく確認を求める（実際に素のデータセットへ
        # 実行された）
        st.warning(
            f"⚠ `{_src_name}` は**クロップ生成で作られたものではありません**"
            "（作成時の倍率が記録されていません）。"
            "このまま実行すると、素の画像の中央だけを切り抜いた"
            "別のデータセットができます。"
        )
        if not st.checkbox("それでも切り直す", key="rc_force"):
            return

    _c4, _c5 = st.columns([2, 2])
    with _c4:
        _keep = st.checkbox("実機の出力サイズに合わせてリサイズする", value=True,
                            key="rc_resize",
                            help="実機が out_size で出すなら合わせておくと、"
                                 "見た目の質感まで揃います")
    with _c5:
        _rsize = st.number_input("そのサイズ", 128, 2048, step=64,
                                 value=int(_prov.get("out_size", DEFAULT_OUT_SIZE)),
                                 key="rc_size", disabled=not _keep)

    if _to >= _from:
        st.warning("学習で使う倍率は、いまの倍率より小さくしてください。")
        return

    _shrink = float(_to) / float(_from)
    st.caption(f"内側 **{_shrink * 100:.0f}%** を切り出します"
               f"（対象は相対的に {1 / _shrink:.2f} 倍の大きさに写ります）。")

    _out_name = st.text_input("出力先（data/ 配下の名前）",
                              value=f"{_src_name}_t{_to:g}".replace(".", ""),
                              key="rc_out")
    _out = DATA_DIR / _out_name.strip() if _out_name.strip() else None

    if st.button("✂️ 切り直す", key="rc_run", type="primary",
                 use_container_width=True, disabled=_out is None):
        _bar = st.progress(0.0, text="切り直しています…")

        def _p(done, total):
            if total:
                _bar.progress(min(done / total, 1.0), text=f"{done} / {total} 枚")

        _r = recut_dataset(_src, _out, float(_from), float(_to),
                           out_size=int(_rsize) if _keep else None,
                           on_progress=_p)
        _bar.empty()
        if _r["ok"]:
            st.success(
                f"✅ {_r['images']} 枚を切り直しました"
                f"（ラベル {_r['labels']} 件"
                + (f"　枠の外に出た {_r['dropped']} 件は落としました" if _r["dropped"] else "")
                + f"）→ `{_out.name}`")
            record_dataset_provenance(
                _out, source="recut", task_type=dataset_task_type(_src) or "segment",
                labels=[], status="draft", tags=["クロップ", "切り直し"],
                extra={"recut": {"from": str(_src), "from_scale": float(_from),
                                 "target_scale": float(_to),
                                 "out_size": int(_rsize) if _keep else None,
                                 "images": _r["images"]}})
        else:
            show_error(_r["error"], prefix="⚠ 切り直せません: ")
        for _p2, _why in _r["skipped"][:10]:
            st.caption(f"・{Path(_p2).name} — {_why}")
