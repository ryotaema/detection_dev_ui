# =============================================================================
# Step4: 推論・評価
# =============================================================================
from __future__ import annotations

import io
import json
import os
import shutil
import time
import zipfile
from datetime import datetime
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from core import *  # noqa: F401,F403
from core import (  # noqa: F401
    _box_iou, _collect_prediction_items, _deploy_worker, _DOC_AUG, _DOC_TRAIN,
    _draw_predictions, _eval_worker, _find_image_dirs, _get_deploy_shared,
    _get_eval_shared, _get_train_shared, _iou, _MODEL_OPTS, _nuctl,
    _StdoutCapture, _train_worker, _yolo_txt_to_xyxy,
)
from .widgets import show_error




def render_evaluate() -> None:
    _mdl_count3 = len(list(MODELS_DIR.rglob("*.pt")))
    _prev_info3 = (f"← 前のステップ: ✅ 学習済みモデルが {_mdl_count3} 件あります"
                   if _mdl_count3 > 0 else "← 前のステップ: ⚠ Step3でモデルを先に学習してください")
    st.markdown(f"""
    <div class="step-banner">
      <div class="sb-title">🔭 STEP 4: 推論・評価</div>
      <div class="sb-prev">{_prev_info3}</div>
      <div class="sb-desc">→ ここでやること: ①推論する → ②精度を測る → ③結果を深掘りする</div>
    </div>""", unsafe_allow_html=True)

    # 3つの内部タブで共通して使う情報。ここで一度だけ集める
    current_model = st.session_state.last_model_path or ""
    _all_models   = list(MODELS_DIR.rglob("*.pt"))
    _model_map    = {str(p.relative_to(MODELS_DIR)): str(p) for p in _all_models}
    _pred_jsons   = sorted(PREDICTIONS_DIR.glob("*.json"))

    # 推論が主役。精度測定と深掘り分析は「推論したあとに使うもの」なので後ろに置く
    _t_infer, _t_eval, _t_deep = st.tabs([
        "▶ ① 推論する",
        "📊 ② 精度を測る",
        "🔬 ③ 深掘り分析",
    ])

    # =======================================================================
    # ① 推論する
    # =======================================================================
    with _t_infer:
        st.markdown('<div class="section-head"><h3>▶ 推論して結果を見る</h3></div>',
                    unsafe_allow_html=True)

        model_display = current_model if current_model else "（未設定）"
        st.info(f"メインモデル: `{model_display}`")

        st.markdown("#### ① 推論する対象を選ぶ")
        compare_mode = st.checkbox("🔀 複数モデル比較モード", value=False, key="cmp_mode")
        if compare_mode and _model_map:
            selected_compare_models = st.multiselect(
                "比較するモデルを選択",
                list(_model_map.keys()),
                default=list(_model_map.keys())[:min(2, len(_model_map))],
                key="cmp_models",
            )
        else:
            selected_compare_models = []
        # --- 推論対象ソース ---
        _infer_src = st.radio(
            "推論対象ソース",
            ["📂 data/ のディレクトリ", "📤 画像をアップロード", "🎬 動画をアップロード"],
            horizontal=True,
            key="infer_src_mode",
        )

        test_image_dir = ""
        test_video_path = ""

        if _infer_src == "📤 画像をアップロード":
            import io as _io_infer
            _infer_files = st.file_uploader(
                "推論したい画像ファイル（複数選択可）",
                type=["jpg", "jpeg", "png", "bmp", "tiff"],
                accept_multiple_files=True,
                key="infer_upload_files",
            )
            if _infer_files:
                st.caption(f"✅ {len(_infer_files)} ファイル選択中")
                _tmp_infer = PREDICTIONS_DIR / "_tmp_uploads"
                _tmp_infer.mkdir(exist_ok=True)
                _cur_names = {f.name for f in _infer_files}
                _saved_names = {f.name for f in _tmp_infer.iterdir() if f.is_file()}
                if _cur_names != _saved_names:
                    for _tf in list(_tmp_infer.iterdir()):
                        _tf.unlink()
                    for _f in _infer_files:
                        (_tmp_infer / _f.name).write_bytes(_f.getbuffer())
                test_image_dir = str(_tmp_infer)

        elif _infer_src == "🎬 動画をアップロード":
            _infer_video = st.file_uploader(
                "推論したい動画ファイル（MP4 / AVI / MOV / MKV）",
                type=["mp4", "avi", "mov", "webm", "mkv"],
                accept_multiple_files=False,
                key="infer_upload_video",
            )
            if _infer_video:
                st.caption(f"✅ {_infer_video.name} 選択中")
                PREDICTIONS_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
                _saved_video = PREDICTIONS_VIDEOS_DIR / _infer_video.name
                _saved_video.write_bytes(_infer_video.getbuffer())
                test_video_path = str(_saved_video)

            # トラッキング設定
            _track_enabled = st.checkbox("🔄 オブジェクトトラッキングを有効にする", value=False, key="video_track_enabled")
            if _track_enabled:
                _tracker_choice = st.radio(
                    "トラッカー",
                    ["ByteTrack", "BoT-SORT"],
                    horizontal=True,
                    key="video_tracker_choice",
                    help="ByteTrack: 高速・位置ベース。BoT-SORT: 高精度・外観特徴も使用（遮蔽に強い）",
                )
                _tracker_yaml = "/app/bytetrack.yaml" if _tracker_choice == "ByteTrack" else "/app/botsort.yaml"
            else:
                _tracker_yaml = "/app/bytetrack.yaml"

            # テンポラル平滑化設定
            _smooth_enabled = st.checkbox(
                "🕐 テンポラル平滑化（ちらつき抑制）",
                value=False,
                key="video_smooth_enabled",
                help="直近Nフレームの検出を記憶し、一時的に消えた検出をグレーで補完描画します",
            )
            if _smooth_enabled:
                _smooth_frames = st.slider(
                    "補完フレーム数",
                    min_value=1, max_value=30, value=5, step=1,
                    key="video_smooth_frames",
                    help="検出が消えてから何フレームまで補完するか。大きいほどちらつきが減るが残像が増える",
                )
            else:
                _smooth_frames = 0

            if compare_mode:
                st.info("ℹ 動画モードでは複数モデル比較は使用できません。メインモデルで推論します。")

        else:
            _img_dirs = _find_image_dirs(DATA_DIR)
            _dir_labels = [str(d.relative_to(DATA_DIR)) for d in _img_dirs]
            _MANUAL = "（手動入力）"
            _dir_options = _dir_labels + [_MANUAL]

            _sel = st.selectbox(
                "テスト画像ディレクトリを選択",
                _dir_options,
                index=0 if _dir_labels else len(_dir_options) - 1,
                help=f"スキャン元: {DATA_DIR}",
            )
            if _sel == _MANUAL:
                test_image_dir = st.text_input(
                    "パスを直接入力 (コンテナ内絶対パス)",
                    value=str(DATA_DIR / "test/images"),
                )
            else:
                test_image_dir = str(DATA_DIR / _sel)
                st.code(test_image_dir, language="text")

        inf_conf = st.slider("確信度しきい値", 0.05, 0.95, 0.25, step=0.05, key="inf_conf")

        st.markdown("#### ② 推論を実行する")
        col_run, col_vis = st.columns(2)

        # --- 推論実行ボタン ---
        with col_run:
            _infer_disabled = (
                (not current_model and not (compare_mode and selected_compare_models))
                or (not test_image_dir and not test_video_path)
            )
            if st.button("▶ 推論実行", type="primary", use_container_width=True, key="infer_run",
                        disabled=_infer_disabled):

                # ── 動画推論 ──────────────────────────────────────────────
                if _infer_src == "🎬 動画をアップロード":
                    if not test_video_path:
                        st.error("動画ファイルを選択してください")
                    elif not current_model:
                        st.error("モデルが未設定です。Step3 で学習するか、データ管理タブで選択してください")
                    else:
                        _prog_bar = st.progress(0.0, text="動画推論中…")
                        def _video_prog(fi, tot):
                            if tot > 0:
                                _prog_bar.progress(min(fi / tot, 1.0),
                                                   text=f"フレーム {fi}/{tot} 処理中…")
                        video_result = run_video_inference(
                            current_model,
                            Path(test_video_path),
                            PREDICTIONS_VIDEOS_DIR,
                            conf_threshold=inf_conf,
                            enable_tracking=_track_enabled,
                            tracker=_tracker_yaml,
                            temporal_smoothing=_smooth_enabled,
                            smooth_frames=_smooth_frames,
                            progress_cb=_video_prog,
                        )
                        _prog_bar.empty()
                        if video_result:
                            st.success(
                                f"✅ 動画推論完了: {video_result['total_frames']} フレーム処理"
                            )
                            st.session_state.last_video_result = video_result

                # ── 画像推論 ──────────────────────────────────────────────
                else:
                    img_dir = Path(test_image_dir)
                    if not img_dir.exists():
                        st.error(f"画像ディレクトリが存在しません: {img_dir}")
                    elif compare_mode and selected_compare_models:
                        # 複数モデル比較推論
                        compare_results = []
                        for model_rel in selected_compare_models:
                            model_abs = _model_map[model_rel]
                            with st.spinner(f"推論中: {model_rel}…"):
                                saved = run_inference(
                                    model_abs,
                                    img_dir,
                                    PREDICTIONS_DIR,
                                    conf_threshold=inf_conf,
                                )
                            total_detections = 0
                            total_conf = 0.0
                            conf_count = 0
                            for jf in saved:
                                with open(jf) as f:
                                    pred = json.load(f)
                                boxes = pred.get("boxes", [])
                                total_detections += len(boxes)
                                for b in boxes:
                                    total_conf += b.get("confidence", 0.0)
                                    conf_count += 1
                            avg_conf = total_conf / conf_count if conf_count > 0 else 0.0
                            compare_results.append({
                                "モデル": model_rel,
                                "検出数（合計）": total_detections,
                                "平均信頼度": round(avg_conf, 4),
                                "画像数": len(saved),
                            })
                        if compare_results:
                            st.success("✅ 比較推論完了")
                            import pandas as pd
                            df_cmp = pd.DataFrame(compare_results)
                            st.dataframe(df_cmp, use_container_width=True, hide_index=True)
                    else:
                        with st.spinner("推論中…"):
                            saved = run_inference(
                                current_model,
                                img_dir,
                                PREDICTIONS_DIR,
                                conf_threshold=inf_conf,
                            )
                        if saved:
                            st.success(f"✅ 推論完了: {len(saved)} 件のJSONを保存")
                            with st.expander("保存されたJSON (先頭1件)"):
                                with open(saved[0]) as f:
                                    st.json(json.load(f))

        # --- FiftyOne 起動ボタン ---
        with col_vis:
            fo_dataset_name = st.text_input("FiftyOneデータセット名", value="yolo_predictions", key="fo_name")
            if st.button("🔭 FiftyOne で可視化", use_container_width=True, key="fo_launch"):
                with st.spinner("FiftyOne App を起動中…"):
                    port = launch_fiftyone(fo_dataset_name, PREDICTIONS_DIR)
                if port:
                    fo_url = f"http://localhost:{port}"
                    st.success(f"FiftyOne App が起動しました: {fo_url}")
                    st.session_state.fiftyone_port = port

        # --- FiftyOne iframe 埋め込み ---
        if st.session_state.fiftyone_port:
            fo_url = f"http://localhost:{st.session_state.fiftyone_port}"
            st.markdown(f"""
        <div style="margin-top:16px;">
        <p style="color:var(--text-muted); font-size:.85rem;">
            FiftyOne App が別ポートで起動中。同一ホストの場合は以下から直接アクセスできます。
        </p>
        <a href="{fo_url}" target="_blank" style="color:var(--accent); font-family:'JetBrains Mono',monospace;">
            🔗 FiftyOne App を開く → {fo_url}
        </a>
        </div>
        <iframe src="{fo_url}" width="100%" height="600px"
        style="border:1px solid var(--border); border-radius:8px; margin-top:12px;"
        allow="fullscreen">
        </iframe>
        """, unsafe_allow_html=True)

        st.markdown("#### ③ 結果を確認する")
        # --- 動画推論結果 ---
        _vr = st.session_state.get("last_video_result")
        if _vr:
            st.markdown("##### 🎬 動画推論結果")
            _out_video = _vr.get("video_path")
            _frame_stats = _vr.get("frame_stats", [])
            _total_frames = _vr.get("total_frames", 0)

            _vc1, _vc2 = st.columns([3, 2])
            with _vc1:
                if _out_video and Path(_out_video).exists():
                    with open(_out_video, "rb") as _vf:
                        _video_bytes = _vf.read()
                    st.video(_video_bytes)
                    st.download_button(
                        "⬇ アノテーション済み動画をダウンロード",
                        _video_bytes,
                        key="dl_video",
                        file_name=Path(_out_video).name,
                        mime="video/mp4",
                        use_container_width=True,
                    )
                else:
                    st.warning("出力動画ファイルが見つかりません")

            with _vc2:
                if _frame_stats:
                    import pandas as pd
                    _df_frames = pd.DataFrame([
                        {"フレーム": s["frame"], "検出数": s["detections"]}
                        for s in _frame_stats
                    ])
                    _total_det = sum(s["detections"] for s in _frame_stats)
                    _det_frames = sum(1 for s in _frame_stats if s["detections"] > 0)

                    # トラッキング時: ユニーク ID 数を集計
                    _all_track_ids = {
                        b["track_id"]
                        for s in _frame_stats
                        for b in s["boxes"]
                        if "track_id" in b
                    }
                    _is_tracked = len(_all_track_ids) > 0

                    st.metric("総フレーム数", _total_frames)
                    st.metric("検出フレーム数", _det_frames)
                    st.metric("総検出数", _total_det)
                    if _is_tracked:
                        st.metric("ユニークトラック数", len(_all_track_ids))

                    st.markdown("**フレームごとの検出数**")
                    st.line_chart(_df_frames.set_index("フレーム"))

            if st.button("🗑 この動画結果をクリア", key="clear_video_result"):
                st.session_state.last_video_result = None
                st.rerun()
        # --- 推論結果 画像プレビュー ---
        if _pred_jsons:
            _reanno_count = len(st.session_state.reanno_set)
            _prev_header_c1, _prev_header_c2 = st.columns([4, 2])
            with _prev_header_c1:
                st.markdown("##### 🖼 推論結果プレビュー")
            with _prev_header_c2:
                if _reanno_count > 0:
                    st.markdown(
                        f'<div style="padding-top:10px; color:var(--warning); font-size:.85rem;">'
                        f'🚩 再アノテーション: <b>{_reanno_count}</b> 件</div>',
                        unsafe_allow_html=True,
                    )
            _preview_jsons = _pred_jsons[:9]
            for _row_start in range(0, len(_preview_jsons), 3):
                _row_files = _preview_jsons[_row_start:_row_start + 3]
                _row_cols = st.columns(3)
                for _col, _jf in zip(_row_cols, _row_files):
                    _res = _draw_predictions(_jf)
                    with _col:
                        if _res:
                            _img, _n_boxes, _stem = _res
                            st.image(_img, caption=f"{_stem} ({_n_boxes}件検出)",
                                     use_column_width=True)
                        else:
                            st.caption(prediction_display_name(_jf))
                        _is_flagged = _jf.name in st.session_state.reanno_set
                        _flag_label = "🚩 フラグ解除" if _is_flagged else "🚩 再アノテーション要"
                        if st.button(_flag_label, key=f"prev_flag_{_jf.name}",
                                     use_container_width=True,
                                     type="secondary"):
                            if _is_flagged:
                                st.session_state.reanno_set.discard(_jf.name)
                            else:
                                st.session_state.reanno_set.add(_jf.name)
                            st.rerun()
            if len(_pred_jsons) > 9:
                st.caption(f"（他 {len(_pred_jsons) - 9} 件は省略。全件エクスポートの選択グリッドからフラグ付け可能）")

        st.markdown("#### ④ 書き出す")
        # --- 推論結果 画像エクスポート ---
        if _pred_jsons:
            _exp_mode = st.radio(
                "エクスポート範囲",
                ["すべて書き出す", "選択して書き出す"],
                horizontal=True,
                key="exp_mode",
            )

            _exp_target_files: Optional[list[Path]] = None
            if _exp_mode == "選択して書き出す":
                _SEL_PAGE_SIZE = 12

                # 選択状態はウィジェットキーではなく set で管理（ページ切替後も保持）
                if "exp_sel_set" not in st.session_state:
                    st.session_state.exp_sel_set = set()
                if "exp_sel_page" not in st.session_state:
                    st.session_state.exp_sel_page = 0

                _total_pages = max(1, (len(_pred_jsons) + _SEL_PAGE_SIZE - 1) // _SEL_PAGE_SIZE)
                _cur_page    = min(st.session_state.exp_sel_page, _total_pages - 1)
                _page_jsons  = _pred_jsons[_cur_page * _SEL_PAGE_SIZE : (_cur_page + 1) * _SEL_PAGE_SIZE]
                _sel_count   = len(st.session_state.exp_sel_set)

                # ── ツールバー ──
                _tb1, _tb2, _tb3, _tb4 = st.columns([2, 2, 2, 4])
                with _tb1:
                    if st.button("☑ 全件選択", key="exp_sel_all", use_container_width=True):
                        st.session_state.exp_sel_set = {jf.name for jf in _pred_jsons}
                        st.rerun()
                with _tb2:
                    if st.button("☑ このページ", key="exp_sel_page_btn", use_container_width=True):
                        st.session_state.exp_sel_set.update(jf.name for jf in _page_jsons)
                        st.rerun()
                with _tb3:
                    if st.button("☐ すべて解除", key="exp_desel_all", use_container_width=True):
                        st.session_state.exp_sel_set = set()
                        st.rerun()
                with _tb4:
                    st.markdown(
                        f'<div style="padding-top:8px; color:var(--accent); font-size:.85rem;">'
                        f'選択中: <b>{_sel_count}</b> / {len(_pred_jsons)} 件 &nbsp;|&nbsp; '
                        f'ページ {_cur_page + 1} / {_total_pages}</div>',
                        unsafe_allow_html=True,
                    )

                # ── 画像グリッド + チェックボックス + フラグ ──
                _GRID_COLS = 3
                for _row_start in range(0, len(_page_jsons), _GRID_COLS):
                    _row_files = _page_jsons[_row_start : _row_start + _GRID_COLS]
                    _row_cols  = st.columns(_GRID_COLS)
                    for _col, _jf in zip(_row_cols, _row_files):
                        with _col:
                            _res = _draw_predictions(_jf)
                            if _res:
                                _img, _n_boxes, _stem = _res
                                st.image(_img, caption=f"{_stem} ({_n_boxes}件)", use_column_width=True)
                            else:
                                st.caption(prediction_display_name(_jf))
                                st.markdown("_(プレビュー不可)_")
                            _chk_result = st.checkbox(
                                "選択",
                                value=(_jf.name in st.session_state.exp_sel_set),
                                key=f"exp_chk_{_cur_page}_{_jf.name}",
                            )
                            if _chk_result:
                                st.session_state.exp_sel_set.add(_jf.name)
                            else:
                                st.session_state.exp_sel_set.discard(_jf.name)
                            _is_flagged_sel = _jf.name in st.session_state.reanno_set
                            _flag_lbl_sel = "🚩 解除" if _is_flagged_sel else "🚩 要再アノテ"
                            if st.button(_flag_lbl_sel, key=f"sel_flag_{_cur_page}_{_jf.name}",
                                         use_container_width=True, type="secondary"):
                                if _is_flagged_sel:
                                    st.session_state.reanno_set.discard(_jf.name)
                                else:
                                    st.session_state.reanno_set.add(_jf.name)
                                st.rerun()

                # ── ページネーション ──
                _pn1, _pn2, _pn3 = st.columns([1, 2, 1])
                with _pn1:
                    if st.button("← 前へ", disabled=(_cur_page == 0),
                                 key="exp_pg_prev", use_container_width=True):
                        st.session_state.exp_sel_page = _cur_page - 1
                        st.rerun()
                with _pn2:
                    st.markdown(
                        f'<div style="text-align:center; padding-top:8px; color:var(--text-muted); font-size:.82rem;">'
                        f'{_cur_page + 1} / {_total_pages}</div>',
                        unsafe_allow_html=True,
                    )
                with _pn3:
                    if st.button("次へ →", disabled=(_cur_page == _total_pages - 1),
                                 key="exp_pg_next", use_container_width=True):
                        st.session_state.exp_sel_page = _cur_page + 1
                        st.rerun()

                _exp_target_files = [PREDICTIONS_DIR / n for n in st.session_state.exp_sel_set
                                     if (PREDICTIONS_DIR / n).exists()]
                _exp_count = len(_exp_target_files)
            else:
                # モード切替時に選択をリセット
                if "exp_sel_set" in st.session_state:
                    st.session_state.exp_sel_set = set()
                _exp_count = len(_pred_jsons)

            # ── フォーマット・品質 ──────────────────────────────────────────────────
            _exp_c1, _exp_c2 = st.columns(2)
            with _exp_c1:
                _exp_fmt = st.selectbox("フォーマット", ["PNG", "JPEG"], key="exp_fmt")
            with _exp_c2:
                _exp_q = st.slider("品質 (JPEGのみ)", 60, 100, 95, step=5,
                                   disabled=(_exp_fmt != "JPEG"), key="exp_quality")

            # ── 保存先フォルダ選択（ブラウザUI） ──────────────────────────────────
            st.markdown("**📁 保存先フォルダ**")
            if "exp_dest_dir" not in st.session_state:
                st.session_state.exp_dest_dir = str(PREDICTIONS_DIR / "exports")

            _exp_dest = Path(st.session_state.exp_dest_dir)
            _BROWSE_ROOT = Path("/workspace")

            # 現在パス表示 + 「↑ 上へ」ボタン
            _nav1, _nav2 = st.columns([5, 1])
            with _nav1:
                st.code(str(_exp_dest), language="text")
            with _nav2:
                _can_up = (
                    _exp_dest != _BROWSE_ROOT
                    and str(_exp_dest).startswith(str(_BROWSE_ROOT))
                )
                if st.button("↑ 上へ", key="exp_nav_up", use_container_width=True,
                             disabled=not _can_up):
                    st.session_state.exp_dest_dir = str(_exp_dest.parent)
                    st.rerun()

            # サブフォルダ一覧（クリックで移動）
            try:
                _subdirs = sorted(
                    [d for d in _exp_dest.iterdir() if d.is_dir()]
                ) if _exp_dest.exists() else []
            except Exception:
                _subdirs = []

            if _subdirs:
                _COLS = 4
                _sd_cols = st.columns(_COLS)
                for _ci, _sd in enumerate(_subdirs[:12]):
                    with _sd_cols[_ci % _COLS]:
                        if st.button(f"📁 {_sd.name}", key=f"exp_sd_{_ci}",
                                     use_container_width=True):
                            st.session_state.exp_dest_dir = str(_sd)
                            st.rerun()
            else:
                st.caption("サブフォルダなし")

            # 新規フォルダ作成
            _nf1, _nf2 = st.columns([4, 1])
            with _nf1:
                _exp_new_folder = st.text_input(
                    "新しいフォルダ名", key="exp_new_folder",
                    placeholder="フォルダ名を入力して ＋ 作成",
                    label_visibility="collapsed",
                )
            with _nf2:
                if st.button("＋ 作成", key="exp_mkdir", use_container_width=True):
                    if _exp_new_folder.strip():
                        _nd = _exp_dest / _exp_new_folder.strip()
                        _nd.mkdir(parents=True, exist_ok=True)
                        st.session_state.exp_dest_dir = str(_nd)
                        st.rerun()

            # ── 書き出しボタン ──────────────────────────────────────────────────────
            _btn_disabled = (_exp_mode == "選択して書き出す" and _exp_count == 0)
            if st.button(f"📥 {_exp_count} 件を書き出す", use_container_width=True,
                         type="primary", disabled=_btn_disabled, key="exp_images_run"):
                _exp_out = _exp_dest
                _prog_bar  = st.progress(0, text="書き出し準備中…")
                _prog_text = st.empty()

                def _on_progress(cur, total, fname):
                    _prog_bar.progress(cur / total,
                                       text=f"{cur} / {total} 件処理中")
                    _prog_text.caption(f"→ {fname}")

                _ok, _ng = export_prediction_images(
                    _exp_out, _exp_fmt, _exp_q, _exp_target_files, _on_progress
                )
                _prog_bar.empty()
                _prog_text.empty()
                if _ok > 0:
                    st.success(f"✅ {_ok} 件を保存しました → `{_exp_out}`")
                    # ZIPにまとめてブラウザからダウンロード
                    import io as _io_exp
                    _exp_glob = sorted(_exp_out.glob("*.*"))
                    if _exp_glob:
                        _zip_buf = _io_exp.BytesIO()
                        with zipfile.ZipFile(_zip_buf, "w", zipfile.ZIP_DEFLATED) as _zf:
                            for _ef in _exp_glob:
                                _zf.write(_ef, _ef.name)
                        _zip_buf.seek(0)
                        st.download_button(
                            f"⬇️ ZIPでダウンロード ({_ok}件)",
                            _zip_buf.getvalue(),
                            key="dl_exported_zip",
                            file_name=f"exports_{datetime.now():%Y%m%d_%H%M}.zip",
                            mime="application/zip",
                            use_container_width=True,
                        )
                if _ng > 0:
                    st.warning(f"⚠ {_ng} 件スキップ（元画像が見つからないため）")

        # --- 推論結果 JSON ブラウザ ---
        with st.expander("📋 predictions/ の結果ファイル一覧"):
            json_files = sorted(PREDICTIONS_DIR.glob("*.json"))
            if json_files:
                for jf in json_files[:20]:  # 最大20件表示
                    with st.container():
                        c1, c2 = st.columns([4, 1])
                        with c1:
                            st.text(jf.name)
                        with c2:
                            if st.button("👁", key=f"view_{jf.name}"):
                                with open(jf) as f:
                                    st.json(json.load(f))
            else:
                st.info("predictions/ にJSONファイルがありません。先に推論を実行してください。")

    # =======================================================================
    # ② 精度を測る
    # =======================================================================
    with _t_eval:
        st.markdown('<div class="section-head"><h3>📊 mAP を同じ条件で測って比べる</h3></div>',
                    unsafe_allow_html=True)
        st.caption(
            "学習時の results.csv は「そのモデルが自分の val で出した値」なので、"
            "別環境で学習したモデルとは比較できません。"
            "ここで同じデータセット・同じ条件で val を回すと、同じ土俵で比べられます。"
        )

        _ev_state, _ev_lock = _get_eval_shared()
        with _ev_lock:
            _ev_running  = _ev_state["running"]
            _ev_log      = list(_ev_state["log"])
            _ev_total    = _ev_state["total"]
            _ev_done     = _ev_state["done"]
            _ev_current  = _ev_state["current"]
            _ev_finished = _ev_state["finished"]
            _ev_error    = _ev_state["error"]

        _ev_yamls = sorted(DATA_DIR.rglob("data.yaml"), key=lambda p: p.stat().st_mtime,
                           reverse=True)
        if not _ev_yamls:
            st.info("評価に使える data.yaml がありません。Step2 でデータセットを作成してください。")
        elif not _model_map:
            st.info("models/ に .pt がありません。")
        else:
            _ev_c1, _ev_c2, _ev_c3 = st.columns([3, 1, 1])
            with _ev_c1:
                _ev_yaml_sel = st.selectbox(
                    "評価に使うデータセット (data.yaml)",
                    [str(p.relative_to(DATA_DIR)) for p in _ev_yamls],
                    key="ev_yaml_sel",
                )
                _ev_yaml_path = str(DATA_DIR / _ev_yaml_sel)
            with _ev_c2:
                _ev_split = st.selectbox("スプリット", ["val", "train"], key="ev_split")
            with _ev_c3:
                _ev_imgsz = st.selectbox("imgsz", [640, 960, 1280], key="ev_imgsz")

            _ev_models_sel = st.multiselect(
                "評価するモデル（複数選択で比較できます）",
                list(_model_map.keys()),
                default=([str(Path(current_model).relative_to(MODELS_DIR))]
                         if current_model and Path(current_model).exists() else []),
                key="ev_models_sel",
            )

            _ev_a1, _ev_a2, _ev_a3 = st.columns(3)
            with _ev_a1:
                _ev_batch = st.number_input("バッチサイズ", 1, 64, 8, key="ev_batch")
            with _ev_a2:
                _ev_conf = st.number_input("conf しきい値", 0.0001, 0.9, 0.001,
                                           format="%.4f", key="ev_conf",
                                           help="mAP は全信頼度域の PR 曲線から計算するため、"
                                                "低い値（既定 0.001）を使うのが正しい計測です")
            with _ev_a3:
                _ev_iou = st.number_input("NMS IoU", 0.1, 0.95, 0.6, key="ev_iou")

            _ev_key = f"{Path(_ev_yaml_path).parent.name}:{_ev_split}"

            if st.button(f"📊 {len(_ev_models_sel)} 件のモデルを評価",
                         type="primary", use_container_width=True,
                         disabled=_ev_running or not _ev_models_sel, key="ev_run"):
                start_evaluation(
                    [_model_map[m] for m in _ev_models_sel],
                    _ev_yaml_path, _ev_split, int(_ev_imgsz),
                    int(_ev_batch), float(_ev_conf), float(_ev_iou),
                )
                st.rerun()

            # --- 実行中 / 完了ログ ---
            if _ev_running or _ev_finished:
                if _ev_running:
                    st.progress(_ev_done / max(_ev_total, 1),
                                text=f"評価中 {_ev_done}/{_ev_total}　{_ev_current}")
                elif _ev_error:
                    show_error(_ev_error, prefix="❌ 評価に失敗しました: ")
                else:
                    st.success("✅ 評価が完了しました")
                st.code("\n".join(_ev_log[-20:]) or "(実行待ち)", language="text")

            # --- 比較表（保存済みの評価結果を横断で集める）---
            _ev_rows = collect_model_evals(_ev_key)
            if _ev_rows:
                st.markdown(f"**📋 比較表 — `{_ev_key}` で評価済みの {len(_ev_rows)} モデル**")
                import pandas as _pd_ev

                _ev_tbl = []
                _has_mask_metric = any(_r.get("mask_map50") is not None for _r in _ev_rows)
                _is_cls_eval = all(_r.get("task") == "classify" for _r in _ev_rows)
                for _r in _ev_rows:
                    _mp = _r["model_path"]
                    _spd = (_r.get("speed_ms") or {}).get("inference")
                    if _is_cls_eval:
                        # 画像分類は mAP ではなく accuracy で比較する
                        _ev_tbl.append({
                            "モデル": str(_mp.relative_to(MODELS_DIR)),
                            "top1 accuracy": round(_r.get("top1") or 0.0, 4),
                            "top5 accuracy": round(_r.get("top5") or 0.0, 4),
                            "推論(ms)": _spd,
                            "サイズ(MB)": round(_mp.stat().st_size / 1024 / 1024, 1),
                            "評価日時": _r.get("evaluated_at", ""),
                        })
                        continue
                    _row = {
                        "モデル": str(_mp.relative_to(MODELS_DIR)),
                        "mAP50": round(_r["map50"], 4),
                        "mAP50-95": round(_r["map50_95"], 4),
                    }
                    # セグメンテーションモデルはマスク基準の mAP も並べる
                    if _has_mask_metric:
                        _row["mask mAP50"] = (round(_r["mask_map50"], 4)
                                              if _r.get("mask_map50") is not None else None)
                        _row["mask mAP50-95"] = (round(_r["mask_map50_95"], 4)
                                                 if _r.get("mask_map50_95") is not None else None)
                    _row.update({
                        "Precision": round(_r["precision"], 3),
                        "Recall": round(_r["recall"], 3),
                        "推論(ms)": _spd,
                        "サイズ(MB)": round(_mp.stat().st_size / 1024 / 1024, 1),
                        "評価日時": _r.get("evaluated_at", ""),
                    })
                    _ev_tbl.append(_row)
                _sort_key = "top1 accuracy" if _is_cls_eval else "mAP50-95"
                _df_ev = _pd_ev.DataFrame(_ev_tbl).sort_values(_sort_key, ascending=False)
                st.dataframe(_df_ev, use_container_width=True, hide_index=True)

                _best = _df_ev.iloc[0]
                if _is_cls_eval:
                    st.success(
                        f"🏆 このデータセットで最も精度が高いのは **{_best['モデル']}** "
                        f"（top1 = {_best['top1 accuracy']:.4f} / "
                        f"top5 = {_best['top5 accuracy']:.4f}）"
                    )
                else:
                    st.success(
                        f"🏆 このデータセットで最も精度が高いのは **{_best['モデル']}** "
                        f"（mAP50-95 = {_best['mAP50-95']:.4f} / mAP50 = {_best['mAP50']:.4f}）"
                    )

                # クラス別 AP と成果物プロット
                _ev_detail_sel = st.selectbox(
                    "詳細を見るモデル", _df_ev["モデル"].tolist(), key="ev_detail_sel")
                _ev_detail = next(
                    (r for r in _ev_rows
                     if str(r["model_path"].relative_to(MODELS_DIR)) == _ev_detail_sel), None)
                if _ev_detail:
                    if _ev_detail.get("per_class"):
                        st.markdown("**クラス別**")
                        st.dataframe(
                            _pd_ev.DataFrame([{
                                "クラス": c["class"],
                                "AP50": round(c["ap50"], 4),
                                "AP50-95": round(c["ap50_95"], 4),
                                "Precision": round(c["precision"], 3),
                                "Recall": round(c["recall"], 3),
                            } for c in _ev_detail["per_class"]]),
                            use_container_width=True, hide_index=True,
                        )
                    _pd_dir = _ev_detail.get("plots_dir")
                    if _pd_dir and Path(_pd_dir).exists():
                        _cm = Path(_pd_dir) / "confusion_matrix_normalized.png"
                        _pr = Path(_pd_dir) / "BoxPR_curve.png"
                        _pcols = st.columns(2)
                        if _cm.exists():
                            _pcols[0].image(str(_cm), caption="混同行列（正規化）",
                                            use_column_width=True)
                        if _pr.exists():
                            _pcols[1].image(str(_pr), caption="Precision-Recall 曲線",
                                            use_column_width=True)
            else:
                st.caption(f"`{_ev_key}` での評価結果はまだありません。")

        if _ev_running:
            # ここで st.rerun() すると以降のタブが描画されないため予約だけする
            request_rerun_poll()

    # =======================================================================
    # ③ 深掘り分析
    # =======================================================================
    with _t_deep:
        st.markdown('<div class="section-head"><h3>🔬 結果を深掘りして次の一手を決める</h3></div>',
                    unsafe_allow_html=True)
        st.caption("推論結果とデータセットを突き合わせて、"
                   "「次にどの画像をアノテーションし直すか」「どの conf で運用するか」を決めます。")

        # --- 要確認画像の自動抽出 ---
        if _pred_jsons:
            st.markdown("#### 🔍 要確認画像の自動抽出")
            st.caption(
                "推論結果を分析して「モデルが自信を持てていない画像」を機械的に拾います。"
                "全画像を目視して 🚩 を立てる代わりに、ここで一括フラグできます。"
            )

            _aa_c1, _aa_c2 = st.columns([2, 3])
            with _aa_c1:
                _aa_conf = st.slider("低信頼度のしきい値", 0.05, 0.95, 0.50, 0.05,
                                     key="aa_conf_low",
                                     help="この値未満の検出を含む画像を要確認とみなします")
            with _aa_c2:
                st.markdown("**抽出する条件**")
                _ac1, _ac2 = st.columns(2)
                with _ac1:
                    _aa_zero     = st.checkbox("検出ゼロ（見逃しの疑い）", value=True, key="aa_zero")
                    _aa_low      = st.checkbox("低信頼度を含む", value=True, key="aa_low")
                with _ac2:
                    _aa_conflict = st.checkbox("クラス競合（迷っている）", value=True, key="aa_conflict")
                    _aa_tiny     = st.checkbox("極小ボックス（ノイズの疑い）", value=False, key="aa_tiny")

            if st.button("🔍 要確認画像を抽出", use_container_width=True, key="aa_run"):
                with st.spinner(f"{len(_pred_jsons)} 件を分析中…"):
                    st.session_state["aa_rows"] = analyze_predictions(
                        _pred_jsons, conf_low=_aa_conf)

            _aa_rows = st.session_state.get("aa_rows") or []
            if _aa_rows:
                # チェックした条件だけを採用する
                _aa_want = set()
                if _aa_zero:     _aa_want.add("検出ゼロ")
                if _aa_low:      _aa_want.add("低信頼度")
                if _aa_conflict: _aa_want.add("クラス競合")
                if _aa_tiny:     _aa_want.add("極小ボックス")

                _aa_hits = []
                for _r in _aa_rows:
                    _kinds = {x.split("(")[0] for x in _r["reasons"]}
                    if _kinds & _aa_want:
                        _aa_hits.append({**_r, "matched": sorted(_kinds & _aa_want)})

                _am1, _am2, _am3 = st.columns(3)
                _am1.metric("分析した画像", len(_aa_rows))
                _am2.metric("要確認", len(_aa_hits))
                _am3.metric("要確認の割合",
                            f"{len(_aa_hits) / len(_aa_rows) * 100:.1f}%" if _aa_rows else "—")

                # 理由別の内訳
                _aa_agg: dict[str, int] = {}
                for _h in _aa_hits:
                    for _k in _h["matched"]:
                        _aa_agg[_k] = _aa_agg.get(_k, 0) + 1
                if _aa_agg:
                    st.markdown("　".join(f"`{k}` {v}件" for k, v in sorted(_aa_agg.items())))

                if _aa_hits:
                    import pandas as _pd_aa
                    _df_aa = _pd_aa.DataFrame([{
                        "ファイル": _h.get("display_name") or _h["name"],
                        "検出数": _h["n_boxes"],
                        "最低conf": (f"{_h['min_conf']:.2f}" if _h["min_conf"] is not None else "—"),
                        "理由": ", ".join(_h["reasons"]),
                    } for _h in _aa_hits])
                    st.dataframe(_df_aa, use_container_width=True, hide_index=True, height=260)

                    _ab1, _ab2 = st.columns(2)
                    with _ab1:
                        if st.button(f"🚩 {len(_aa_hits)} 件にまとめてフラグを立てる",
                                     type="primary", use_container_width=True, key="aa_flag_all"):
                            for _h in _aa_hits:
                                st.session_state.reanno_set.add(_h["name"])
                            st.success(f"{len(_aa_hits)} 件にフラグを立てました")
                            st.rerun()
                    with _ab2:
                        if st.button("抽出結果をクリア", use_container_width=True, key="aa_clear"):
                            st.session_state["aa_rows"] = []
                            st.rerun()
                else:
                    st.success("✅ 選択した条件に該当する画像はありませんでした。")

        # --- 再アノテーション用エクスポート ---
        if _pred_jsons:
            st.markdown("#### 🚩 再アノテーション用エクスポート")
            _ra_set  = st.session_state.reanno_set
            _ra_jsons = [PREDICTIONS_DIR / n for n in sorted(_ra_set)
                         if (PREDICTIONS_DIR / n).exists()]

            if not _ra_jsons:
                st.info("上の自動抽出、またはプレビューの 🚩 ボタンで画像にフラグを立てると、ここに表示されます。")
            else:
                st.markdown(
                    f'<div style="color:var(--warning); font-size:.9rem; margin-bottom:8px;">'
                    f'🚩 フラグ済み: <b>{len(_ra_jsons)}</b> 件</div>',
                    unsafe_allow_html=True,
                )

                # フラグ済み画像のサムネイル一覧
                with st.expander(f"フラグ済み画像を確認する（{len(_ra_jsons)} 件）", expanded=False):
                    for _ra_row in range(0, len(_ra_jsons), 3):
                        _ra_cols = st.columns(3)
                        for _rc, _rj in zip(_ra_cols, _ra_jsons[_ra_row:_ra_row + 3]):
                            with _rc:
                                _rr = _draw_predictions(_rj)
                                if _rr:
                                    _ri, _rn, _rs = _rr
                                    st.image(_ri, caption=f"{_rs} ({_rn}件)", use_column_width=True)
                                else:
                                    st.caption(_rj.stem)

                # ── CVAT へ直接送る（ZIP ダウンロード → 手動アップロードを不要にする）──
                with st.expander("📤 CVAT に新規タスクとして送る（推奨）", expanded=True):
                    st.caption(
                        "フラグ済み画像を CVAT のタスクとして直接作成します。"
                        "予測ボックスを事前アノテーションとして入れておけば、"
                        "作業者はゼロから引くのではなく「直す」だけで済みます。"
                    )
                    _pu_name = st.text_input(
                        "CVAT タスク名",
                        value=f"recheck_{datetime.now():%Y%m%d_%H%M}",
                        key="push_task_name",
                    )
                    _pu_c1, _pu_c2 = st.columns(2)
                    with _pu_c1:
                        _pu_with_ann = st.checkbox(
                            "予測ボックスを事前アノテーションとして入れる",
                            value=True, key="push_with_ann",
                        )
                    with _pu_c2:
                        _pu_extra = st.text_input(
                            "追加ラベル（カンマ区切り・任意）",
                            value="", key="push_extra_labels",
                            help="検出ゼロの画像だけを送る場合や、"
                                 "予測に出てこないクラスを後から付けたい場合に指定します",
                        )

                    if len(_ra_jsons) > 200:
                        st.warning(f"⚠ {len(_ra_jsons)} 件を送信します。画像のアップロードに時間がかかります。")

                    if st.button(f"📤 CVAT に {len(_ra_jsons)} 件を送る",
                                 type="primary", use_container_width=True,
                                 disabled=not _pu_name.strip(), key="push_run"):
                        _pu_labels = [s.strip() for s in _pu_extra.split(",") if s.strip()]
                        with st.spinner("CVAT にタスクを作成中…（画像アップロード中）"):
                            _pu_res = push_predictions_to_cvat(
                                _ra_jsons,
                                task_name=_pu_name.strip(),
                                extra_labels=_pu_labels,
                                with_annotations=_pu_with_ann,
                            )
                        st.session_state["push_result"] = _pu_res

                    _pu_last = st.session_state.get("push_result")
                    if _pu_last:
                        if _pu_last["ok"]:
                            st.success(
                                f"✅ タスクを作成しました（ID: {_pu_last['task_id']} / "
                                f"{_pu_last['n_images']} 枚 / ラベル: {', '.join(_pu_last['labels'])}）"
                            )
                            st.markdown(f"👉 [CVAT でこのタスクを開く]({_pu_last['url']})")
                            st.caption("作業が終わったら「📤 Step2: データ取込」でエクスポートして学習に回せます。")
                        else:
                            show_error(_pu_last["error"], prefix="❌ 送信に失敗しました: ")

                st.caption(
                    "ZIP 出力形式: 元画像 (`images/`) + YOLO txt ラベル (`labels/`) "
                    "+ `classes.txt` + CVAT for images 1.1 XML (`annotations.xml`)"
                )
                _ra_c1, _ra_c2 = st.columns(2)
                with _ra_c1:
                    if st.button("⬇ 再アノテーション用 ZIP を生成",
                                 type="primary", use_container_width=True,
                                 key="reanno_zip"):
                        with st.spinner("ZIP を生成中…"):
                            _zip_bytes, _ok, _ng = build_reannotation_zip(_ra_jsons)
                        if _ok > 0:
                            st.download_button(
                                f"⬇ ダウンロード（{_ok} 件）",
                                _zip_bytes,
                                key="dl_reanno_zip",
                                file_name=f"reannotation_{datetime.now():%Y%m%d_%H%M}.zip",
                                mime="application/zip",
                                use_container_width=True,
                            )
                        if _ng > 0:
                            st.warning(f"⚠ {_ng} 件は元画像が見つからずスキップしました")
                with _ra_c2:
                    if st.button("🗑 フラグをすべてクリア", use_container_width=True, key="reanno_clear"):
                        st.session_state.reanno_set = set()
                        st.rerun()

        st.markdown("---")
        # --- 実運用の conf を決める ---
        with st.expander("🎚 最適な信頼度しきい値 (conf) を探す", expanded=False):
            st.caption(
                "mAP は「モデルの実力」を測る指標ですが、実際に使うときは "
                "**どの conf で運用するか**を決める必要があります。"
                "しきい値を振って Precision / Recall / F1 を測り、判断材料を出します。"
            )

            _sw_yamls = sorted(DATA_DIR.rglob("data.yaml"), key=lambda p: p.stat().st_mtime,
                               reverse=True)
            if not _sw_yamls or not _model_map:
                st.info("data.yaml と学習済みモデルの両方が必要です。")
            else:
                _swc1, _swc2 = st.columns([3, 2])
                with _swc1:
                    _sw_yaml_sel = st.selectbox(
                        "データセット (data.yaml)",
                        [str(p.relative_to(DATA_DIR)) for p in _sw_yamls], key="sw_yaml")
                with _swc2:
                    _sw_model_sel = st.selectbox(
                        "モデル", list(_model_map.keys()),
                        index=(list(_model_map.values()).index(current_model)
                               if current_model in _model_map.values() else 0),
                        key="sw_model")

                _swp1, _swp2, _swp3 = st.columns(3)
                with _swp1:
                    _sw_split = st.selectbox("スプリット", ["val", "train"], key="sw_split")
                with _swp2:
                    _sw_iou = st.slider("一致とみなす IoU", 0.1, 0.9, 0.5, 0.05, key="sw_iou")
                with _swp3:
                    _sw_max = st.number_input("最大画像数", 0, 100000, 300, 100, key="sw_max",
                                              help="0 で全画像")

                if st.button("🎚 しきい値を振って測る", type="primary",
                             use_container_width=True, key="sw_run"):
                    with st.spinner("推論して各しきい値で評価しています…"):
                        st.session_state["sw_result"] = sweep_confidence(
                            Path(_model_map[_sw_model_sel]),
                            str(DATA_DIR / _sw_yaml_sel), split=_sw_split,
                            iou_match=float(_sw_iou), max_images=int(_sw_max),
                        )

                _sw = st.session_state.get("sw_result")
                if _sw and not _sw["ok"]:
                    show_error(_sw["error"], prefix="❌ 測定に失敗しました: ")
                elif _sw:
                    import pandas as _pd_sw

                    _df_sw = _pd_sw.DataFrame([{
                        "conf": r["conf"], "Precision": round(r["precision"], 3),
                        "Recall": round(r["recall"], 3), "F1": round(r["f1"], 3),
                        "TP": r["tp"], "FP": r["fp"], "FN": r["fn"],
                    } for r in _sw["rows"]])

                    st.markdown(f"**{_sw['n_images']} 枚で測定**（IoU {_sw['iou_match']} で一致判定）")
                    st.line_chart(_df_sw.set_index("conf")[["Precision", "Recall", "F1"]])

                    _b = _sw["best_f1"]
                    _hp, _hr = _sw["high_precision"], _sw["high_recall"]
                    _rc1, _rc2, _rc3 = st.columns(3)
                    with _rc1:
                        st.metric("バランス重視 (F1最大)", f"{_b['conf']:.2f}" if _b else "—")
                        if _b:
                            st.caption(f"P {_b['precision']:.3f} / R {_b['recall']:.3f} / "
                                       f"F1 {_b['f1']:.3f}")
                    with _rc2:
                        st.metric("誤検出を避ける", f"{_hp['conf']:.2f}" if _hp else "—")
                        st.caption(f"P {_hp['precision']:.3f} / R {_hp['recall']:.3f}"
                                   if _hp else "Precision 0.95 以上に届く点がありません")
                    with _rc3:
                        st.metric("見逃しを避ける", f"{_hr['conf']:.2f}" if _hr else "—")
                        st.caption(f"P {_hr['precision']:.3f} / R {_hr['recall']:.3f}"
                                   if _hr else "Recall 0.95 以上を保てる点がありません")

                    st.caption(
                        "用途に合わせて選んでください。"
                        "検査や安全用途で見逃したくないなら Recall 寄り（低め）、"
                        "自動処理で誤検出を出したくないなら Precision 寄り（高め）、"
                        "自動アノテーションの下書きなら少し低めが便利です"
                        "（消す方が描くより速いため）。"
                    )
                    st.dataframe(_df_sw, use_container_width=True, hide_index=True, height=260)

        with st.expander("🔬 正解ラベルとの差分分析（アノテーション漏れを探す）", expanded=False):
            st.caption(
                "モデルの予測を正解ラベル(GT)と突き合わせ、画像ごとに "
                "**FN（取りこぼし）** と **FP（余計な検出）** を数えます。"
                "精度の高いモデルが FN を出す画像は、モデルの誤りではなく "
                "**GT 側のアノテーションが漏れている**ことがよくあります。"
            )

            _gd_yamls = sorted(DATA_DIR.rglob("data.yaml"), key=lambda p: p.stat().st_mtime,
                               reverse=True)
            if not _gd_yamls or not _model_map:
                st.info("data.yaml と学習済みモデルの両方が必要です。")
            else:
                _gd_c1, _gd_c2 = st.columns([3, 2])
                with _gd_c1:
                    _gd_yaml_sel = st.selectbox(
                        "データセット (data.yaml)",
                        [str(p.relative_to(DATA_DIR)) for p in _gd_yamls], key="gd_yaml")
                    _gd_yaml = str(DATA_DIR / _gd_yaml_sel)
                with _gd_c2:
                    _gd_model_sel = st.selectbox(
                        "使用するモデル", list(_model_map.keys()),
                        index=(list(_model_map.values()).index(current_model)
                               if current_model in _model_map.values() else 0),
                        key="gd_model")

                _gd_p1, _gd_p2, _gd_p3, _gd_p4 = st.columns(4)
                with _gd_p1:
                    _gd_split = st.selectbox("スプリット", ["val", "train"], key="gd_split")
                with _gd_p2:
                    _gd_conf = st.slider("推論 conf", 0.05, 0.9, 0.25, 0.05, key="gd_conf")
                with _gd_p3:
                    _gd_iou = st.slider("一致とみなす IoU", 0.1, 0.9, 0.5, 0.05, key="gd_iou")
                with _gd_p4:
                    _gd_max = st.number_input("最大画像数", 0, 100000, 500, 100, key="gd_max",
                                              help="0 で全画像。多いほど時間がかかります")

                if st.button("🔬 差分を分析", type="primary", use_container_width=True,
                             key="gd_run"):
                    with st.spinner("推論して GT と突き合わせています…"):
                        st.session_state["gd_result"] = compare_with_ground_truth(
                            Path(_model_map[_gd_model_sel]), _gd_yaml, split=_gd_split,
                            conf=float(_gd_conf), iou_match=float(_gd_iou),
                            max_images=int(_gd_max),
                        )

                _gd = st.session_state.get("gd_result")
                if _gd and not _gd["ok"]:
                    show_error(_gd["error"], prefix="❌ 分析に失敗しました: ")
                elif _gd:
                    import pandas as _pd_gd

                    _gd_imgs = _gd["per_image"]
                    _n_clean = sum(1 for p in _gd_imgs if p["fp"] == 0 and p["fn"] == 0)

                    _gm = st.columns(5)
                    _gm[0].metric("画像数", _gd["n_images"])
                    _gm[1].metric("TP（一致）", _gd["tp"])
                    _gm[2].metric("FN（取りこぼし）", _gd["fn"])
                    _gm[3].metric("FP（余計な検出）", _gd["fp"])
                    _gm[4].metric("完全一致", f"{_n_clean}/{_gd['n_images']}")
                    if _gd["precision"] is not None:
                        st.caption(f"Precision {_gd['precision']:.3f} / "
                                   f"Recall {_gd['recall']:.3f}"
                                   f"（conf={_gd['conf']}, IoU={_gd['iou_match']} での実測）")

                    if _gd["by_class"]:
                        st.markdown("**クラス別**")
                        st.dataframe(_pd_gd.DataFrame([
                            {"クラス": k, "TP": v["tp"], "FP": v["fp"], "FN": v["fn"]}
                            for k, v in _gd["by_class"].items()
                        ]), use_container_width=True, hide_index=True)

                    # 要確認画像の抽出条件
                    st.markdown("**要確認画像の抽出**")
                    _gf1, _gf2 = st.columns(2)
                    with _gf1:
                        _gd_min_fn = st.number_input("FN が この件数以上", 0, 50, 1, key="gd_min_fn")
                    with _gf2:
                        _gd_min_fp = st.number_input("または FP が この件数以上", 0, 50, 2,
                                                     key="gd_min_fp")

                    _gd_hits = [p for p in _gd_imgs
                                if (_gd_min_fn and p["fn"] >= _gd_min_fn)
                                or (_gd_min_fp and p["fp"] >= _gd_min_fp)]
                    _gd_hits.sort(key=lambda d: -(d["fn"] * 2 + d["fp"]))

                    st.markdown(f"該当: **{len(_gd_hits)}** 件"
                                f"（差分の大きい順。FN を重く重み付けしています）")
                    if _gd_hits:
                        st.dataframe(_pd_gd.DataFrame([{
                            "ファイル": p["name"], "GT": p["n_gt"], "予測": p["n_pred"],
                            "TP": p["tp"], "FP": p["fp"], "FN": p["fn"],
                        } for p in _gd_hits]), use_container_width=True, hide_index=True,
                            height=260)

                        _ga1, _ga2 = st.columns(2)
                        with _ga1:
                            _gd_task = st.text_input(
                                "CVAT タスク名",
                                value=f"labelfix_{datetime.now():%Y%m%d_%H%M}",
                                key="gd_task_name")
                            if st.button(f"📤 {len(_gd_hits)} 件を CVAT に送る",
                                         type="primary", use_container_width=True,
                                         disabled=not _gd_task.strip(), key="gd_push"):
                                _gd_items = [{
                                    "path": Path(p["image"]), "width": p["width"],
                                    "height": p["height"], "boxes": p["pred_boxes"],
                                } for p in _gd_hits]
                                _gd_labels = sorted({b["label"] for it in _gd_items
                                                     for b in it["boxes"]})
                                if not _gd_labels:
                                    _gd_labels = list(_gd["by_class"].keys())
                                with st.spinner("CVAT にタスクを作成中…"):
                                    st.session_state["gd_push_result"] = push_items_to_cvat(
                                        _gd_items, _gd_labels, _gd_task.strip(),
                                        with_annotations=True)
                            _gdp = st.session_state.get("gd_push_result")
                            if _gdp:
                                if _gdp["ok"]:
                                    st.success(f"✅ タスク作成（ID: {_gdp['task_id']} / "
                                               f"{_gdp['n_images']} 枚）")
                                    st.markdown(f"👉 [CVAT で開く]({_gdp['url']})")
                                else:
                                    st.error(f"❌ {_gdp['error']}")
                        with _ga2:
                            _gd_fo_name = st.text_input(
                                "FiftyOne データセット名", value="gt_vs_pred",
                                key="gd_fo_name")
                            if st.button("🔭 FiftyOne で GT と予測を見比べる",
                                         use_container_width=True, key="gd_fo"):
                                with st.spinner("FiftyOne App を起動中…"):
                                    _gd_port = launch_fiftyone_comparison(
                                        _gd_fo_name.strip() or "gt_vs_pred", _gd_hits)
                                if _gd_port:
                                    st.success(f"起動しました → http://localhost:{_gd_port}")
                                    st.caption("`ground_truth`（正解）と `predictions`（予測）を"
                                               "重ねて表示できます。`n_fn` / `n_fp` でソートも可能です。")
                    else:
                        st.success("✅ 条件に該当する画像はありませんでした。")
