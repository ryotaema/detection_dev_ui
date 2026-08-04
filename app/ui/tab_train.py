# =============================================================================
# Step3: モデル学習
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
from .widgets import _ckw, _nw, _ph, _selw, _sw, empty_state, metric_row, show_error
from .presets import (_apply_preset, _BUILTIN_PRESETS, _collect_current_params,
                      _load_user_presets, _save_user_presets, _USER_PRESETS_FILE)
from .theme import active_theme




def render_train() -> None:
    # 学習の進捗はスレッドと共有する。cache_resource なので同じ実体が返る
    _train_state, _train_log_lock = _get_train_shared()
    _yaml_count = len(list(DATA_DIR.rglob("data.yaml")))
    _prev_info2 = (f"← 前のステップ: ✅ data.yaml が {_yaml_count} 件あります"
                   if _yaml_count > 0 else "← 前のステップ: ⚠ Step2でデータセットを先に生成してください")
    st.markdown(f"""
    <div class="step-banner">
      <div class="sb-title">🚀 STEP 3: モデル学習</div>
      <div class="sb-prev">{_prev_info2}</div>
      <div class="sb-desc">→ ここでやること: モデルサイズ・学習パラメータを設定して学習開始</div>
    </div>""", unsafe_allow_html=True)
    st.markdown('<div class="section-head"><h3>🚀 YOLO 学習設定</h3></div>', unsafe_allow_html=True)

    # ── いまの学習の状態 ─────────────────────────────────────────────────────
    #   学習が走っている間は「進捗とログ」が主役。設定より前に出す。
    #   設定ウィジェットは下に残す（描画をやめると Streamlit が
    #   ウィジェットの状態を破棄してしまい、学習後に設定が初期値へ戻るため）。
    # --- _train_state → st.session_state に同期 ---
    with _train_log_lock:
        st.session_state.training_log = list(_train_state["log"])
        st.session_state.training_progress = _train_state["progress"]
        st.session_state.training_running = _train_state["running"]
        st.session_state.training_metrics_history = list(_train_state["metrics_history"])
        if _train_state["error"]:
            st.session_state.training_error = _train_state["error"]
        if _train_state["model_path"]:
            st.session_state.last_model_path = _train_state["model_path"]

    # --- 学習完了トースト（1回だけ） ---
    if (st.session_state.training_progress == 100
            and not st.session_state.training_running
            and not st.session_state.training_notified):
        st.toast("🎉 学習が完了しました！", icon="✅")
        st.balloons()
        st.session_state.training_notified = True

    # --- 進捗表示 ---
    if st.session_state.training_running:
        # ── 学習中: プログレスバー＋リアルタイムグラフ＋自動スクロールログ ──
        prog = st.session_state.training_progress
        st.progress(prog / 100, text=f"進捗: {prog}%")

        # ── 停止（エポック末で安全に打ち切る）──
        with _train_log_lock:
            _stop_pending = _train_state.get("stop_requested", False)
        _stop_c1, _stop_c2 = st.columns([1, 3])
        with _stop_c1:
            if st.button("⏹ 学習を停止", type="secondary", use_container_width=True,
                         disabled=_stop_pending, key="train_stop_btn"):
                with _train_log_lock:
                    _train_state["stop_requested"] = True
                st.rerun()
        with _stop_c2:
            if _stop_pending:
                st.warning("⏳ 停止要求を受け付けました。現在のエポックが終わり次第停止します。")
            else:
                st.caption("停止してもその時点までの `best.pt` / `last.pt` は保存されます。"
                           "`last.pt` があれば下の「中断した学習を再開」から続きから再開できます。")

        _mh = st.session_state.training_metrics_history
        if _mh:
            import pandas as pd
            df_live = pd.DataFrame(_mh)
            if "epoch" in df_live.columns:
                df_live = df_live.set_index("epoch")
                _live_cols = [c for c in df_live.columns
                              if any(k in c.lower() for k in ["map50", "loss"])
                              and "95" not in c.lower()]
                if _live_cols:
                    st.markdown("**📊 学習進捗グラフ（リアルタイム）**")
                    st.line_chart(df_live[_live_cols])

        log_lines = st.session_state.training_log[-500:]
        log_text_escaped = "\n".join(log_lines).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
        # iframe の中は親ドキュメントの CSS 変数を継承しないため、
        # ここだけはテーマから実際の色を取り出して埋め込む
        _th = active_theme()
        components.html(f"""
    <style>
      body{{margin:0;background:{_th['bg_log']};}}
      #log-box{{
    background:{_th['bg_log']};color:{_th['text_secondary']};
    font-family:'JetBrains Mono',monospace;font-size:12px;
    height:380px;overflow-y:auto;
    padding:12px;border:1px solid {_th['border']};border-radius:8px;
    white-space:pre-wrap;word-break:break-all;
      }}
    </style>
    <div id="log-box">{log_text_escaped}</div>
    <script>
      var box = document.getElementById('log-box');
      var dist = 0;
      try {{ dist = parseInt(window.parent.localStorage.getItem('log_dist_bottom') || '0'); }} catch(e) {{}}
      setTimeout(function() {{
    if (dist > 100) {{
      box.scrollTop = box.scrollHeight - box.clientHeight - dist;
    }} else {{
      box.scrollTop = box.scrollHeight;
    }}
      }}, 0);
      var t = null;
      box.addEventListener('scroll', function() {{
    clearTimeout(t);
    t = setTimeout(function() {{
      var d = Math.max(0, box.scrollHeight - box.scrollTop - box.clientHeight);
      try {{ window.parent.localStorage.setItem('log_dist_bottom', d); }} catch(e) {{}}
    }}, 100);
      }}, {{passive:true}});
    </script>
    """, height=400)

        # ここで st.rerun() すると以降のタブが描画されないため予約だけする
        request_rerun_poll()

    elif st.session_state.training_progress == 100:
        # ── 学習完了: プログレスバー＋ログ（expander / 静的表示） ──
        st.progress(1.0, text="進捗: 100% — 完了")
        if st.session_state.training_log:
            with st.expander("📋 学習ログ（完了）", expanded=False):
                st.text("\n".join(st.session_state.training_log[-500:]))

    elif st.session_state.training_progress > 0:
        # ── 途中停止: 止まった時点のプログレスバー＋ログ ──
        prog = st.session_state.training_progress
        st.progress(prog / 100, text=f"進捗: {prog}% — 停止")
        if st.session_state.training_log:
            with st.expander("📋 学習ログ（停止時点）", expanded=False):
                st.text("\n".join(st.session_state.training_log[-500:]))

    if st.session_state.training_error:
        show_error(st.session_state.training_error, prefix="学習エラー: ")

    # --- 完了後: モデル選択 ---
    if st.session_state.last_model_path:
        st.success(f"✅ 最新モデル: `{st.session_state.last_model_path}`")

    # results.csv の可視化
    if st.session_state.last_model_path:
        results_csv = Path(st.session_state.last_model_path).parent.parent / "results.csv"
        if results_csv.exists():
            import pandas as pd
            st.markdown("#### 📈 学習メトリクス")
            df_r = pd.read_csv(results_csv)
            df_r.columns = [c.strip() for c in df_r.columns]
            metric_cols = [c for c in df_r.columns
                           if any(k in c.lower() for k in ["map","precision","recall","loss"])]
            _last = df_r.iloc[-1]
            _map_col  = next((c for c in df_r.columns if "map50" in c.lower() and "95" not in c.lower()), None)
            _loss_col = next((c for c in df_r.columns if "val" in c.lower() and "loss" in c.lower()), None)
            _mc = st.columns(3)
            if _map_col:  _mc[0].metric("mAP50 (最終)", f"{_last[_map_col]:.4f}")
            if _loss_col: _mc[1].metric("Val Loss (最終)", f"{_last[_loss_col]:.4f}")
            _mc[2].metric("エポック数", len(df_r))
            if metric_cols:
                st.line_chart(df_r[metric_cols])
            with st.expander("📄 生データ（末尾5行）"):
                st.dataframe(df_r.tail(5), use_container_width=True)

    # ── 学習の設定 ───────────────────────────────────────────────────────────
    if st.session_state.training_running:
        st.info(
            "⏳ 学習を実行中です。以下の設定を変えても、いま走っている学習には反映されません"
            "（次に「▶ 学習開始」を押したときから有効になります）。"
        )

    st.markdown("#### ① 学習プリセットを選ぶ（任意）")
    st.caption("よく使う設定の組み合わせです。迷ったらここから選んで、必要なら②③で微調整します。")
    # ── プリセット ───────────────────────────────────────────────────────────
    _user_presets  = _load_user_presets()
    _all_presets   = {**_BUILTIN_PRESETS,
                      **{f"👤 {k}": v for k, v in _user_presets.items()}}
    _PRESET_NONE   = "（選択してください）"

    _pr1, _pr2, _pr3 = st.columns([4, 1, 2])
    with _pr1:
        _preset_sel = st.selectbox(
            "プリセット",
            [_PRESET_NONE] + list(_all_presets.keys()),
            key="preset_sel",
            label_visibility="collapsed",
        )
    with _pr2:
        if st.button("▶ 適用", key="preset_apply", use_container_width=True,
                     disabled=(_preset_sel == _PRESET_NONE)):
            _preset_target = _all_presets.get(_preset_sel)
            if _preset_target:
                _apply_preset(_preset_target)
            else:
                st.warning("適用するプリセットを選んでください。")
    with _pr3:
        if st.button("💾 現在の設定を保存", key="preset_save_btn", use_container_width=True):
            st.session_state["preset_save_mode"] = True

    if st.session_state.get("preset_save_mode", False):
        with st.container(border=True):
            st.caption("保存するプリセット名を入力してください")
            _sv1, _sv2, _sv3 = st.columns([4, 1, 1])
            with _sv1:
                _new_pname = st.text_input(
                    "プリセット名", key="preset_new_name",
                    placeholder="例: 小物体むけ 高解像度", label_visibility="collapsed",
                )
            with _sv2:
                if st.button("✅ 保存", key="preset_save_confirm", use_container_width=True):
                    if _new_pname.strip():
                        _ups = _load_user_presets()
                        _ups[_new_pname.strip()] = _collect_current_params()
                        _save_user_presets(_ups)
                        st.session_state["preset_save_mode"] = False
                        st.toast(f"✅ プリセット「{_new_pname.strip()}」を保存しました", icon="💾")
                        st.rerun()
                    else:
                        st.warning("プリセット名を入力してください")
            with _sv3:
                if st.button("✕ キャンセル", key="preset_save_cancel", use_container_width=True):
                    st.session_state["preset_save_mode"] = False
                    st.rerun()

    if _user_presets:
        with st.expander("🗂 ユーザープリセット管理"):
            _editing = st.session_state.get("preset_editing_name", None)

            for _uname, _uparams in list(_user_presets.items()):
                st.markdown(f"**{_uname}**")
                _up1, _up2, _up3, _up4 = st.columns([1, 1, 1, 1])
                _param_summary = (
                    f"`{_uparams.get('model','?')} · {_uparams.get('epochs','?')}ep · "
                    f"{_uparams.get('imgsz','?')}px`"
                )
                st.caption(_param_summary)
                with _up1:
                    if st.button("▶ 適用", key=f"upr_apply_{_uname}", use_container_width=True):
                        _apply_preset(_uparams)
                with _up2:
                    if st.button("✏️ 編集", key=f"upr_edit_{_uname}", use_container_width=True):
                        st.session_state["preset_editing_name"] = _uname
                        st.session_state["preset_editing_vals"] = dict(_uparams)
                        st.rerun()
                with _up3:
                    if st.button("🗑 削除", key=f"upr_del_{_uname}", use_container_width=True):
                        _ups = _load_user_presets()
                        _ups.pop(_uname, None)
                        _save_user_presets(_ups)
                        if st.session_state.get("preset_editing_name") == _uname:
                            st.session_state.pop("preset_editing_name", None)
                            st.session_state.pop("preset_editing_vals", None)
                        st.rerun()
                st.markdown("---")

            # ── 編集フォーム ──────────────────────────────────────────────────
            if _editing and _editing in _user_presets:
                _ev = st.session_state.get("preset_editing_vals", {})
                st.markdown(f"#### ✏️ 編集中: **{_editing}**")
                _OPTS_OPT = ["auto","SGD","Adam","AdamW","NAdam","RAdam"]
                _ef1, _ef2, _ef3 = st.columns(3)
                with _ef1:
                    _ev["model"]   = st.selectbox("モデル", _MODEL_OPTS,
                        index=_MODEL_OPTS.index(_ev.get("model","yolo11s")) if _ev.get("model","yolo11s") in _MODEL_OPTS else _MODEL_OPTS.index("その他"),
                        key="pe_model")
                    _ev["epochs"]  = st.number_input("エポック数", 1, 5000, int(_ev.get("epochs",100)), step=10, key="pe_epochs")
                    _ev["batch"]   = st.select_slider("バッチサイズ", [-1,4,8,16,32,64,128],
                        value=_ev.get("batch",8), key="pe_batch")
                with _ef2:
                    _ev["imgsz"]   = st.select_slider("imgsz", [320,416,512,640,768,1024,1280],
                        value=int(_ev.get("imgsz",640)) if int(_ev.get("imgsz",640)) in [320,416,512,640,768,1024,1280] else 640,
                        key="pe_imgsz")
                    _ev["patience"]= st.number_input("patience", 0, 1000, int(_ev.get("patience",50)), step=10, key="pe_patience")
                    _ev["optimizer"]= st.selectbox("optimizer", _OPTS_OPT,
                        index=_OPTS_OPT.index(_ev.get("optimizer","auto")) if _ev.get("optimizer","auto") in _OPTS_OPT else 0,
                        key="pe_optimizer")
                with _ef3:
                    _ev["lr0"]     = st.number_input("lr0", 1e-5, 1.0, float(_ev.get("lr0",0.01)), format="%.5f", step=0.001, key="pe_lr0")
                    _ev["cos_lr"]  = st.checkbox("cos_lr", value=bool(_ev.get("cos_lr",False)), key="pe_cos_lr")
                    _ev["warmup_epochs"] = st.number_input("warmup_epochs", 0, 50, int(_ev.get("warmup_epochs",3)), key="pe_warmup")
                    _ev["dropout"] = st.slider("dropout", 0.0, 0.5, float(_ev.get("dropout",0.0)), step=0.05, key="pe_dropout")

                st.markdown("**拡張設定**")
                _ea1, _ea2, _ea3 = st.columns(3)
                with _ea1:
                    _ev["degrees"]  = st.slider("degrees", 0.0, 180.0, float(_ev.get("degrees",0.0)), step=1.0, key="pe_degrees")
                    _ev["scale"]    = st.slider("scale", 0.0, 0.9, float(_ev.get("scale",0.5)), step=0.05, key="pe_scale")
                    _ev["mosaic"]   = st.slider("mosaic", 0.0, 1.0, float(_ev.get("mosaic",1.0)), step=0.05, key="pe_mosaic")
                with _ea2:
                    _ev["fliplr"]   = st.slider("fliplr", 0.0, 1.0, float(_ev.get("fliplr",0.5)), step=0.05, key="pe_fliplr")
                    _ev["flipud"]   = st.slider("flipud", 0.0, 1.0, float(_ev.get("flipud",0.0)), step=0.05, key="pe_flipud")
                    _ev["mixup"]    = st.slider("mixup", 0.0, 1.0, float(_ev.get("mixup",0.0)), step=0.05, key="pe_mixup")
                with _ea3:
                    _ev["hsv_h"]    = st.slider("hsv_h", 0.0, 0.1, float(_ev.get("hsv_h",0.015)), step=0.005, key="pe_hsv_h")
                    _ev["hsv_s"]    = st.slider("hsv_s", 0.0, 1.0, float(_ev.get("hsv_s",0.7)), step=0.05, key="pe_hsv_s")
                    _ev["hsv_v"]    = st.slider("hsv_v", 0.0, 1.0, float(_ev.get("hsv_v",0.4)), step=0.05, key="pe_hsv_v")
                _eb1, _eb2, _eb3 = st.columns(3)
                with _eb1:
                    _ev["translate"]  = st.slider("translate", 0.0, 0.9, float(_ev.get("translate",0.1)), step=0.05, key="pe_translate")
                with _eb2:
                    _ev["erasing"]    = st.slider("erasing", 0.0, 0.9, float(_ev.get("erasing",0.4)), step=0.05, key="pe_erasing")
                with _eb3:
                    _ev["close_mosaic"]= st.number_input("close_mosaic", 0, 200, int(_ev.get("close_mosaic",10)), step=5, key="pe_close")

                _ec1, _ec2 = st.columns(2)
                with _ec1:
                    if st.button("✅ 変更を保存", key="pe_save", use_container_width=True, type="primary"):
                        _ups = _load_user_presets()
                        _ups[_editing] = dict(_ev)
                        _save_user_presets(_ups)
                        st.session_state.pop("preset_editing_name", None)
                        st.session_state.pop("preset_editing_vals", None)
                        st.toast(f"✅ プリセット「{_editing}」を更新しました", icon="💾")
                        st.rerun()
                with _ec2:
                    if st.button("✕ キャンセル", key="pe_cancel", use_container_width=True):
                        st.session_state.pop("preset_editing_name", None)
                        st.session_state.pop("preset_editing_vals", None)
                        st.rerun()

    st.markdown("---")

    st.markdown("#### ② モデルとデータセットを選ぶ")
    # ── 基本設定 ────────────────────────────────────────────────────────────
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        _model_preset = st.selectbox(
            "モデル",
            _MODEL_OPTS,
            key="tp_model",
        )
    with col_b:
        epochs = st.number_input("エポック数", min_value=1, max_value=5000, step=10,
                                 key="tp_epochs")
    with col_c:
        batch_size = st.select_slider(
            "バッチサイズ",
            options=[-1, 4, 8, 16, 32, 64, 128],
            help="-1 = AutoBatch",
            key="tp_batch",
        )

    if _model_preset == "その他":
        model_name = st.text_input(
            "モデルファイル名 (.pt)",
            value="yolo26n.pt",
            help="例: yolo26n.pt, yolo11x.pt, rtdetr-x.pt",
            key="tp_model_custom",
        )
        if model_name and not model_name.endswith(".pt"):
            model_name = model_name + ".pt"
        if model_name:
            _local_candidates = list(MODELS_DIR.rglob(model_name)) + [Path(model_name)]
            if any(p.exists() for p in _local_candidates):
                st.success(f"✅ ローカルに `{model_name}` が見つかりました — ローカルファイルを使用します")
            else:
                st.info(f"⬇️ 学習開始時に Ultralytics が `{model_name}` を自動ダウンロードします（初回のみ）")
        else:
            model_name = "yolo26n.pt"
    else:
        model_name = f"{_model_preset}.pt"
        st.code(f"モデル: {model_name}", language="text")

    # data.yaml 選択（最終更新順に列挙）
    _yaml_candidates = sorted(
        DATA_DIR.rglob("data.yaml"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    _yaml_labels = [str(p.relative_to(DATA_DIR)) for p in _yaml_candidates]
    _YAML_MANUAL = "（手動入力）"
    _yaml_options = _yaml_labels + [_YAML_MANUAL]
    if not _yaml_labels:
        empty_state(
            "学習に使える data.yaml がまだありません",
            "「📤 Step2: データ取込」でデータセットを生成してください。",
            "すでに YOLO 形式のデータが手元にあるなら、"
            "同タブの「📁 ローカルからデータを直接追加」から ZIP で取り込めます。",
        )

    _yc1, _yc2 = st.columns([10, 1])
    with _yc1:
        # 選択肢に状態を出しておくと、テスト用を誤って学習に使う事故を防げる
        def _yaml_option_label(v: str) -> str:
            if v == _YAML_MANUAL:
                return v
            return f"{v}　{status_label(read_status((DATA_DIR / v).parent))}"

        _yaml_sel = st.selectbox(
            "data.yaml",
            _yaml_options,
            index=0 if _yaml_labels else len(_yaml_options) - 1,
            format_func=_yaml_option_label,
        )
    with _yc2:
        st.markdown('<div style="margin-top:24px"></div>', unsafe_allow_html=True)
        if st.button("📂", key="btn_ds_dir", help="ディレクトリ内容を表示 / 非表示"):
            st.session_state["_ds_dir_open"] = not st.session_state.get("_ds_dir_open", False)

    if _yaml_sel == _YAML_MANUAL:
        data_yaml_path = st.text_input(
            "data.yaml パスを直接入力 (コンテナ内絶対パス)",
            value=str(DATA_DIR / "dataset/data.yaml"),
        )
    else:
        data_yaml_path = str(DATA_DIR / _yaml_sel)
        _ds_dir_path = Path(data_yaml_path).parent

        # 学習に向かない状態のものを選んでいたら知らせる（止めはしない）
        _ds_status = read_status(_ds_dir_path)
        if _ds_status == "test_only":
            st.warning(
                "⚠ このデータセットは **🔵 テスト用** です。評価専用として登録されています。"
                "学習に使うと、あとで同じデータで評価したときに精度が実力より高く出ます。"
            )
        elif _ds_status == "archived":
            st.warning("⚠ このデータセットは **⚪ 保管** です。もう使わない想定で登録されています。")
        elif _ds_status == "auto_annotated":
            st.info(
                "ℹ このデータセットは **🟠 自動アノテのみ**（人の目が入っていない）です。"
                "モデルの誤りをそのまま学習してしまうので、"
                "「🔭 Step4」でアノテーションを確認してからのほうが確実です。"
            )
        elif _ds_status == "draft":
            st.caption(
                "ℹ 状態が **🟡 作成中** のままです。"
                "精査が済んだら「📁 データ管理」で状態を更新しておくと、"
                "あとで何を使ったか分かりやすくなります。"
            )

        # ── データセット詳細パネル ──
        try:
            import yaml as _yaml_mod
            _yaml_content = _yaml_mod.safe_load(Path(data_yaml_path).read_text())
            _nc     = _yaml_content.get("nc", "?")
            _names  = _yaml_content.get("names", [])
            _tr_dir = _ds_dir_path / "images" / "train"
            _vl_dir = _ds_dir_path / "images" / "val"
            _n_tr   = len(list(_tr_dir.glob("*.*"))) if _tr_dir.exists() else "—"
            _n_vl   = len(list(_vl_dir.glob("*.*"))) if _vl_dir.exists() else "—"
            _nm_str = ", ".join(str(n) for n in _names[:10]) + ("…" if len(_names) > 10 else "")
            st.markdown(f"""
    <div style="background:var(--bg-card-inner);border:1px solid var(--success-border);border-left:4px solid var(--success);
     border-radius:6px;padding:10px 16px;margin:6px 0 10px;">
      <div style="color:var(--success);font-size:.87rem;font-weight:700;margin-bottom:6px;">
    📁 {_ds_dir_path.name}
      </div>
      <div style="display:flex;gap:24px;flex-wrap:wrap;margin-bottom:4px;">
    <span style="color:var(--text-secondary);font-size:.82rem;">クラス数: <b style="color:var(--text-primary)">{_nc}</b></span>
    <span style="color:var(--text-secondary);font-size:.82rem;">Train 画像: <b style="color:var(--text-primary)">{_n_tr}</b></span>
    <span style="color:var(--text-secondary);font-size:.82rem;">Val 画像: <b style="color:var(--text-primary)">{_n_vl}</b></span>
      </div>
      <div style="color:var(--text-secondary);font-size:.8rem;">
    ラベル: <span style="color:var(--text-primary)">{_nm_str if _nm_str else "—"}</span>
      </div>
      <div style="color:var(--text-muted);font-size:.75rem;margin-top:4px;">{data_yaml_path}</div>
    </div>""", unsafe_allow_html=True)
        except Exception:
            st.code(data_yaml_path, language="text")

        # ── ディレクトリビューア（📂 ボタンでトグル）──
        if st.session_state.get("_ds_dir_open", False):
            with st.container(border=True):
                st.caption(f"📂 {_ds_dir_path}")
                try:
                    _entries = sorted(
                        _ds_dir_path.iterdir(),
                        key=lambda p: (p.is_file(), p.name),
                    )
                    for _e in _entries:
                        if _e.is_dir():
                            _sub_cnt = len(list(_e.iterdir()))
                            st.markdown(
                                f"📁 **{_e.name}/**"
                                f"<span style='color:var(--text-muted);font-size:.78rem'> ({_sub_cnt} 件)</span>",
                                unsafe_allow_html=True,
                            )
                        else:
                            _sz = _e.stat().st_size
                            _sz_s = (f"{_sz/1024:.1f} KB" if _sz < 1_048_576
                                     else f"{_sz/1048576:.1f} MB")
                            st.markdown(
                                f"📄 {_e.name}"
                                f"<span style='color:var(--text-muted);font-size:.78rem'> {_sz_s}</span>",
                                unsafe_allow_html=True,
                            )
                except Exception as _dir_err:
                    st.warning(str(_dir_err))

    col_p, col_q = st.columns(2)
    with col_p:
        mlflow_project = st.text_input("MLflow プロジェクト名", value="YOLO-Detection")
    with col_q:
        run_name = st.text_input(
            "ラン名",
            value=f"{_model_preset.replace('その他','custom')}_ep{epochs}_{datetime.now():%H%M}",
        )

    st.markdown("---")
    st.markdown("#### ③ 詳細を調整する（任意）")
    st.caption("既定値のままでも学習できます。精度が伸び悩んだときに開いてください。")
    # ── 学習設定（最適化・正則化）────────────────────────────────────────────
    with st.expander("⚙️ 学習設定（最適化・正則化）", expanded=False):
        _oc1, _oc2, _oc3, _oc4 = st.columns(4)
        with _oc1:
            imgsz       = _nw("imgsz", 128, 1280, 640, step=32,
                               name="imgsz", key="tp_imgsz",
                               desc="学習・推論時の画像サイズ（ピクセル）。大きいほど精度が上がるが計算コストが増加する。")
            patience    = _nw("patience（0=無効）", 0, 1000, 50, step=10,
                               name="patience", key="tp_patience",
                               desc="EarlyStopping の待機エポック数。N エポック間 val metrics が改善しなければ自動終了。0 で無効。")
            save_period = _nw("save_period（0=無効）", 0, 500, 0, step=10,
                               name="save_period",
                               desc="N エポックごとにチェックポイントを保存する間隔。0 で無効。長期学習での途中確認に便利。")
            workers     = _nw("workers", 0, 32, 8, step=1,
                               name="workers", key="tp_workers",
                               desc="DataLoader の CPU ワーカースレッド数。多すぎるとメモリ不足になることがある。")
        with _oc2:
            optimizer   = _selw("optimizer", ["auto","SGD","Adam","AdamW","NAdam","RAdam"], 0,
                                 name="optimizer", key="tp_optimizer",
                                 desc="`auto` はモデルに応じて自動選択。細かく制御する場合は SGD または AdamW 推奨。")
            lr0         = _nw("lr0（初期学習率）", 1e-5, 1.0, 0.01, format="%.5f", step=0.001,
                               name="lr0", key="tp_lr0",
                               desc="初期学習率。SGD では 0.01、Adam/AdamW では 0.001 が一般的な推奨値。")
            lrf         = _nw("lrf（最終LR係数）", 1e-4, 1.0, 0.01, format="%.4f", step=0.001,
                               name="lrf",
                               desc="学習率スケジューラの終端係数。最終学習率 = `lr0 × lrf`。")
            cos_lr      = _ckw("cos_lr（コサイン学習率）", False,
                                name="cos_lr", key="tp_cos_lr",
                                desc="True でコサイン学習率スケジューラを使用。学習後半を滑らかに減衰させる。")
        with _oc3:
            momentum    = _nw("momentum（SGD/Adam β1）", 0.5, 0.999, 0.937, format="%.3f", step=0.01,
                               name="momentum",
                               desc="SGD のモメンタム係数、または Adam 系の β1 パラメータ。")
            warmup_epochs = _nw("warmup_epochs", 0, 50, 3, step=1,
                                 name="warmup_epochs", key="tp_warmup_epochs",
                                 desc="ウォームアップのエポック数。最初の N エポックで学習率を 0 から lr0 まで徐々に増加させる。")
            warmup_momentum = _nw("warmup_momentum", 0.0, 1.0, 0.8, format="%.2f", step=0.05,
                                   name="warmup_momentum",
                                   desc="ウォームアップ中の初期モメンタム値。")
            warmup_bias_lr  = _nw("warmup_bias_lr", 0.0, 1.0, 0.1, format="%.3f", step=0.01,
                                   name="warmup_bias_lr",
                                   desc="ウォームアップ中のバイアス層の学習率。")
        with _oc4:
            weight_decay = _nw("weight_decay", 0.0, 0.01, 0.0005, format="%.5f", step=0.0001,
                                name="weight_decay", key="tp_weight_decay",
                                desc="L2 正則化（重み減衰）の強度。過学習の抑制に効果的。")
            dropout      = _sw("dropout", 0.0, 0.5, 0.0, step=0.05,
                                name="dropout", key="tp_dropout",
                                desc="Dropout の確率。0 で無効。学習時にランダムにユニットを無効化して汎化性を高める。",
                                url=_DOC_TRAIN)
            nbs          = _nw("nbs（損失正規化基準バッチ）", 1, 256, 64, step=8,
                                name="nbs",
                                desc="名目バッチサイズ。実バッチサイズが異なる場合に損失をスケーリングする基準値。")
            amp          = _ckw("AMP（混合精度学習）", True,
                                 name="amp",
                                 desc="True で FP16 演算を混在させ GPU メモリを節約しつつ速度を向上させる。Blackwell GPU では有効推奨。")
            cache        = _ckw("cache（画像キャッシュ）", False,
                                 name="cache",
                                 desc="学習画像を RAM/disk にキャッシュ。繰り返しのディスク読み込みを削減して高速化。大規模データセットでは RAM 不足に注意。")

    # ── データ拡張（Augmentation）────────────────────────────────────────────
    with st.expander("🎨 データ拡張（Augmentation）", expanded=False):
        _mn      = model_name.lower()
        _is_seg  = "-seg"  in _mn
        _is_pose = "-pose" in _mn
        _is_cls  = "-cls"  in _mn
        _is_obb  = "-obb"  in _mn
        _has_box = not _is_cls
        _task_label = ("segment" if _is_seg else "pose" if _is_pose
                       else "classify" if _is_cls else "obb" if _is_obb else "detect")
        st.caption(
            f"推定タスク: **{_task_label}** — タスクに適用されないパラメータはグレーアウトされます"
        )

        # ── 幾何変換 ──────────────────────────────────────────────────────────
        st.markdown("##### 🔁 幾何変換")
        _g1, _g2, _g3, _g4 = st.columns(4)
        with _g1:
            degrees = _sw("degrees（回転 ±°）", 0.0, 180.0, 0.0, step=1.0,
                           name="degrees", key="tp_degrees",
                           desc="画像をランダムに回転させる角度範囲（±degrees°）。0 で無効。ロボット視点など姿勢が変化する環境で有効。",
                           disabled=not _has_box)
            shear   = _sw("shear（せん断 ±°）", 0.0, 10.0, 0.0, step=0.5,
                           name="shear",
                           desc="せん断変形（ずれ歪み）の角度範囲（±degrees°）。画像を平行四辺形状に歪める。",
                           disabled=not _has_box)
        with _g2:
            scale     = _sw("scale（拡大縮小）", 0.0, 0.9, 0.5, step=0.05,
                             name="scale", key="tp_scale",
                             desc="ランダムスケーリングの変化幅。0.5 なら画像サイズが ×0.5〜×1.5 の範囲で変化。距離・解像度の変動に対応。",
                             disabled=not _has_box)
            translate = _sw("translate（平行移動）", 0.0, 0.9, 0.1, step=0.05,
                             name="translate", key="tp_translate",
                             desc="水平・垂直方向の平行移動量（画像サイズ比）。物体が画像端にある場合への対応。",
                             disabled=not _has_box)
        with _g3:
            fliplr = _sw("fliplr（左右反転）", 0.0, 1.0, 0.5, step=0.05,
                          name="fliplr", key="tp_fliplr",
                          desc="水平（左右）反転の確率。文字・数字など向きが意味を持つタスクでは 0.0 を推奨。")
            flipud = _sw("flipud（上下反転）", 0.0, 1.0, 0.0, step=0.05,
                          name="flipud", key="tp_flipud",
                          desc="垂直（上下）反転の確率。重力方向が重要なタスクでは 0.0 を推奨。")
        with _g4:
            perspective = _nw("perspective（透視変換）", 0.0, 0.001, 0.0,
                               format="%.4f", step=0.0001,
                               name="perspective", key="tp_perspective",
                               desc="透視投影変換の強度（0〜0.001 程度）。平面を斜めから見たような 3D 的歪みを追加。",
                               url=_DOC_AUG, disabled=not _has_box)
            bgr = _sw("bgr（BGR↔RGB 反転確率）", 0.0, 1.0, 0.0, step=0.05,
                       name="bgr",
                       desc="BGR と RGB のチャンネル順をランダムに入れ替える確率。色に依存しない特徴を学習させる。")

        # ── 色調・明度変換 ───────────────────────────────────────────────────
        st.markdown("##### 🌈 色調・明度変換")
        _c1, _c2, _c3 = st.columns(3)
        with _c1:
            hsv_h = _sw("hsv_h（色相変動）", 0.0, 0.10, 0.015, step=0.005,
                         name="hsv_h", key="tp_hsv_h",
                         desc="HSV 色空間の色相（Hue）の変動量。照明条件の変化や異なる色帯域への汎化に効果的。")
        with _c2:
            hsv_s = _sw("hsv_s（彩度変動）", 0.0, 1.0, 0.7, step=0.05,
                         name="hsv_s", key="tp_hsv_s",
                         desc="HSV 色空間の彩度（Saturation）の変動量。色の鮮やかさをランダムに変化させる。")
        with _c3:
            hsv_v = _sw("hsv_v（明度変動）", 0.0, 1.0, 0.4, step=0.05,
                         name="hsv_v", key="tp_hsv_v",
                         desc="HSV 色空間の明度（Value）の変動量。屋内外の照明差や露出変化に対応させる。")

        # ── 合成拡張 ─────────────────────────────────────────────────────────
        st.markdown("##### 🔀 合成拡張")
        _m1, _m2, _m3, _m4 = st.columns(4)
        with _m1:
            mosaic = _sw("mosaic（4 画像合成）", 0.0, 1.0,
                          1.0 if _has_box else 0.0, step=0.05,
                          name="mosaic", key="tp_mosaic",
                          desc="4 枚の画像をランダムにモザイク結合する確率。小物体の検出精度向上に非常に効果的。detect/segment/pose 向け。",
                          disabled=not _has_box)
            close_mosaic = _nw("close_mosaic（終盤N エポックOFF）", 0, 200, 10, step=5,
                                name="close_mosaic", key="tp_close_mosaic",
                                desc="最後の N エポックでモザイク拡張を OFF にする。学習終盤に拡張なしの本来の分布で収束させ精度を安定させる。",
                                url=_DOC_AUG, disabled=not _has_box)
        with _m2:
            mixup  = _sw("mixup", 0.0, 1.0, 0.0, step=0.05,
                          name="mixup", key="tp_mixup",
                          desc="2 枚の画像とラベルを α ブレンドで混合する確率。クラス境界付近の汎化性向上に有効。detect/segment 向け。",
                          disabled=not _has_box)
            cutmix = _sw("cutmix", 0.0, 1.0, 0.0, step=0.05,
                          name="cutmix",
                          desc="ランダムに切り抜いた領域を別画像で置き換える確率。MixUp の空間的バリアント。",
                          disabled=not _has_box)
        with _m3:
            copy_paste = _sw("copy_paste（セグのみ）", 0.0, 1.0, 0.0, step=0.05,
                              name="copy_paste",
                              desc="【セグメンテーション専用】別画像のセグメント済みオブジェクトをコピーして貼り付ける確率。クラス不均衡の解消やレアオブジェクト増強に有効。",
                              disabled=not _is_seg)
            _cp_c, _cp_h = st.columns([5, 1])
            with _cp_c:
                copy_paste_mode = st.selectbox(
                    "copy_paste_mode", ["flip", "mixup"],
                    disabled=not (_is_seg and copy_paste > 0.0),
                )
            with _cp_h:
                st.markdown('<div style="margin-top:28px"></div>', unsafe_allow_html=True)
                _ph("copy_paste_mode",
                    "`flip` は対象を反転して貼り付け、`mixup` はブレンドして貼り付け。copy_paste > 0 のときのみ有効。",
                    _DOC_AUG)
        with _m4:
            erasing = _sw("erasing（ランダム消去）", 0.0, 0.9, 0.4, step=0.05,
                           name="erasing", key="tp_erasing",
                           desc="ランダムな矩形領域を消去する確率（Random Erasing）。オクルージョン（物体が部分的に隠れる）への耐性を向上させる。")

        # ── 分類専用 ─────────────────────────────────────────────────────────
        if _is_cls:
            st.markdown("##### 📋 分類専用")
            _cl1, _cl2 = st.columns(2)
            with _cl1:
                crop_fraction = _sw("crop_fraction（ランダムクロップ割合）",
                                    0.1, 1.0, 1.0, step=0.05,
                                    name="crop_fraction",
                                    desc="分類タスク専用。画像を中心からランダムにクロップする際の最小割合。",
                                    url=_DOC_AUG)
            with _cl2:
                auto_augment = _selw(
                    "auto_augment", ["randaugment", "autoaugment", "augmix"], 0,
                    name="auto_augment",
                    desc="分類タスク専用の自動拡張ポリシー。`randaugment`（ランダム操作）、`autoaugment`（AutoAugment）、`augmix`（AugMix）から選択。",
                    url=_DOC_AUG,
                )
        else:
            crop_fraction = 1.0
            auto_augment  = "randaugment"

    # ── データ拡張のプレビュー ────────────────────────────────────────────────
    with st.expander("👁 データ拡張を目で確認する", expanded=False):
        st.caption(
            "上で設定した拡張が画像に何をするかを、学習前に確認できます。"
            "パラメータの意味を掴むためのものです。"
        )

        _ap_params = {
            "hsv_h": float(hsv_h), "hsv_s": float(hsv_s), "hsv_v": float(hsv_v),
            "degrees": float(degrees), "translate": float(translate),
            "scale": float(scale), "shear": float(shear),
            "fliplr": float(fliplr), "flipud": float(flipud),
            "mosaic": float(mosaic), "erasing": float(erasing),
        }

        _ap_active = describe_augment(_ap_params)
        if not _ap_active:
            st.info("有効な拡張がありません。上の「🎨 データ拡張」で値を設定してください。")
        else:
            st.markdown("**有効になっている拡張**")
            for _label, _val, _desc in _ap_active:
                st.caption(f"・**{_label}** = `{_val}` … {_desc}")

        _ap_ds = Path(data_yaml_path).parent if data_yaml_path else None
        _ap_imgs = list_sample_images(_ap_ds) if _ap_ds and _ap_ds.exists() else []

        if not _ap_imgs:
            st.warning("プレビューに使える画像が見つかりません。"
                       "先にデータセットを選択してください。")
        else:
            _apc1, _apc2 = st.columns([1, 2])
            with _apc1:
                _ap_seed = st.number_input("乱数シード", 0, 9999, 0, key="ap_seed",
                                           help="変えると別のかかり方を試せます")
            with _apc2:
                _ap_n = st.slider("表示するパターン数", 1, 4, 3, key="ap_n")

            if st.button("👁 プレビューを作る", use_container_width=True, key="ap_run"):
                with st.spinner("生成中…"):
                    st.session_state["ap_preview"] = build_augment_preview(
                        _ap_imgs, _ap_params, seed=int(_ap_seed), n_variants=int(_ap_n))

            _ap_res = st.session_state.get("ap_preview")
            if _ap_res:
                _orig, _vars = _ap_res
                if _orig is None:
                    st.error("画像を読み込めませんでした。")
                else:
                    _cols = st.columns(len(_vars) + 1)
                    _cols[0].image(_orig, caption="元画像", use_column_width=True)
                    for _c, (_lbl, _im) in zip(_cols[1:], _vars):
                        _c.image(_im, caption=_lbl, use_column_width=True)
                    st.caption(
                        "⚠ 実際の学習では拡張が**確率的に**適用され、"
                        "ここでは効果が見えるよう必ず適用しています。"
                        "見え方の傾向を掴むための近似表示です。"
                    )

    st.markdown("---")
    st.markdown("#### ④ 設定を確認して学習を開始する")
    # ── 学習設定サマリー ──────────────────────────────────────────────────────
    _ds_disp = Path(data_yaml_path).parent.name if data_yaml_path else "—"
    st.markdown("##### 📋 学習設定サマリー")
    metric_row([
        ("モデル",     model_name),
        ("エポック数", epochs),
        ("バッチ",     batch_size if batch_size != -1 else "Auto"),
        ("imgsz",      imgsz),
        ("patience",   patience if patience > 0 else "OFF"),
        ("optimizer",  optimizer),
        ("lr0",        lr0),
        ("warmup",     warmup_epochs),
        ("dropout",    dropout),
        ("AMP",        "ON" if amp else "OFF"),
    ])
    st.markdown(
        f'<div style="background:var(--bg-card-inner);border:1px solid var(--border);border-radius:6px;'
        f'padding:8px 14px;margin:8px 0 16px;font-size:.82rem;color:var(--text-secondary);">'
        f'📁 データセット: <b style="color:var(--text-primary)">{_ds_disp}</b>'
        f'<span style="color:var(--text-muted);font-size:.75rem"> &nbsp;—&nbsp; {data_yaml_path}</span></div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── 中断した学習の再開 ───────────────────────────────────────────────────
    # last.pt があり、results.csv のエポック数が設定より少ない run を候補にする
    _resume_cands = []
    if MODELS_DIR.exists():
        for _last in sorted(MODELS_DIR.glob("*/weights/last.pt"),
                            key=lambda p: p.stat().st_mtime, reverse=True):
            _run_dir = _last.parent.parent
            _done_ep, _total_ep = None, None
            _rcsv = _run_dir / "results.csv"
            if _rcsv.exists():
                try:
                    import pandas as _pd_rs
                    _done_ep = len(_pd_rs.read_csv(_rcsv))
                except Exception:
                    pass
            _args_y = _run_dir / "args.yaml"
            if _args_y.exists():
                try:
                    import yaml as _yml_rs
                    _total_ep = (_yml_rs.safe_load(_args_y.read_text()) or {}).get("epochs")
                except Exception:
                    pass
            # 完走していれば候補から外す（判定できない場合は候補に残す）
            if _done_ep is not None and _total_ep is not None and _done_ep >= int(_total_ep):
                continue
            _resume_cands.append({
                "run": _run_dir.name, "last": _last,
                "done": _done_ep, "total": _total_ep,
            })

    if _resume_cands:
        with st.expander(f"⏯ 中断した学習を再開する（候補 {len(_resume_cands)} 件）"):
            st.caption(
                "停止・クラッシュなどで途中終了した学習を `last.pt` から続きから再開します。"
                "エポック数や学習率などの設定は中断時のものが引き継がれます"
                "（上で設定した値は使われません）。"
            )
            _rs_labels = [
                f"{c['run']}　"
                + (f"({c['done']}/{c['total']} エポック完了)"
                   if c["done"] is not None and c["total"] is not None
                   else f"({c['done']} エポック完了)" if c["done"] is not None else "")
                for c in _resume_cands
            ]
            _rs_sel = st.selectbox("再開する学習", _rs_labels, key="resume_sel")
            _rs_target = _resume_cands[_rs_labels.index(_rs_sel)]
            st.caption(f"再開元: `{_rs_target['last']}`")

            if st.button("⏯ この学習を再開する", type="primary", use_container_width=True,
                         disabled=st.session_state.training_running, key="resume_btn"):
                with _train_log_lock:
                    _train_state["log"] = []
                    _train_state["progress"] = 0
                    _train_state["running"] = True
                    _train_state["error"] = None
                    _train_state["model_path"] = None
                    _train_state["metrics_history"] = []
                    _train_state["stop_requested"] = False
                threading.Thread(
                    target=_train_worker,
                    # resume=True のとき data / epochs 等は last.pt 側の設定が使われる
                    args=(data_yaml_path, str(_rs_target["last"]), 0, 0,
                          mlflow_project, _rs_target["run"], {"resume": True}),
                    daemon=True,
                ).start()
                st.session_state.training_notified = False
                st.rerun()

    # ── 学習ボタン ───────────────────────────────────────────────────────────
    btn_col1, btn_col2 = st.columns([2, 1])
    with btn_col1:
        start_btn = st.button(
            "▶ 学習開始",
            type="primary",
            disabled=st.session_state.training_running,
            use_container_width=True,
            key="train_start",
        )
    with btn_col2:
        if st.session_state.training_running:
            st.markdown('<span class="badge-warn">RUNNING</span>', unsafe_allow_html=True)
        elif st.session_state.training_progress == 100:
            st.markdown('<span class="badge-ok">COMPLETED</span>', unsafe_allow_html=True)

    if start_btn:
        yaml_p = Path(data_yaml_path)
        if not yaml_p.exists():
            st.error(f"data.yaml が見つかりません: {yaml_p}")
        else:
            _train_kwargs: dict = {
                # ── 基本 ──────────────────────────────────────────────────
                "imgsz": int(imgsz),
                "device": 0,
                "workers": int(workers),
                "nbs": int(nbs),
                # ── 最適化 ────────────────────────────────────────────────
                "optimizer": optimizer,
                "lr0": float(lr0),
                "lrf": float(lrf),
                "momentum": float(momentum),
                "warmup_epochs": float(warmup_epochs),
                "warmup_momentum": float(warmup_momentum),
                "warmup_bias_lr": float(warmup_bias_lr),
                "weight_decay": float(weight_decay),
                "dropout": float(dropout),
                "cos_lr": cos_lr,
                "amp": amp,
                "cache": cache,
                # ── 幾何変換 ──────────────────────────────────────────────
                "degrees": float(degrees),
                "scale": float(scale),
                "translate": float(translate),
                "shear": float(shear),
                "perspective": float(perspective),
                "flipud": float(flipud),
                "fliplr": float(fliplr),
                "bgr": float(bgr),
                # ── 色調変換 ──────────────────────────────────────────────
                "hsv_h": float(hsv_h),
                "hsv_s": float(hsv_s),
                "hsv_v": float(hsv_v),
                # ── 合成拡張 ──────────────────────────────────────────────
                "mosaic": float(mosaic),
                "mixup": float(mixup),
                "cutmix": float(cutmix),
                "erasing": float(erasing),
                "close_mosaic": int(close_mosaic),
            }
            # セグメンテーション専用
            if _is_seg:
                _train_kwargs["copy_paste"] = float(copy_paste)
                _train_kwargs["copy_paste_mode"] = copy_paste_mode
            # 分類専用
            if _is_cls:
                _train_kwargs["crop_fraction"] = float(crop_fraction)
                _train_kwargs["auto_augment"] = auto_augment
            # 条件付き
            if patience > 0:
                _train_kwargs["patience"] = int(patience)
            if save_period > 0:
                _train_kwargs["save_period"] = int(save_period)

            with _train_log_lock:
                _train_state["log"] = []
                _train_state["progress"] = 0
                _train_state["running"] = True
                _train_state["error"] = None
                _train_state["model_path"] = None
                _train_state["metrics_history"] = []

            t = threading.Thread(
                target=_train_worker,
                args=(data_yaml_path, model_name, epochs, batch_size,
                      mlflow_project, run_name, _train_kwargs),
                daemon=True,
            )
            t.start()
            st.session_state.training_notified = False   # 新規学習開始 → 通知リセット
            st.rerun()


    # --- 既存モデル選択 ---
    # ── 学習履歴の比較（MLflow）────────────────────────────────────────────
    with st.expander("📊 過去の学習を比較する（MLflow）", expanded=False):
        st.caption(
            "これまでの学習を並べて、設定の違いが精度にどう効いたかを確認できます。"
            "学習曲線を重ねて表示できるので、過学習の始まりや伸び止まりも見比べられます。"
        )

        _mf_ok, _mf_err = mlflow_available()
        if not _mf_ok:
            st.warning(f"MLflow に接続できません（{_mf_err}）")
            st.caption("`docker compose up -d mlflow` で起動できます。"
                       "下の「results.csv から比較」は MLflow なしでも使えます。")
        else:
            _mf_exps = list_experiments()
            _mf_exps = [e for e in _mf_exps if e["n_runs"] > 0]
            if not _mf_exps:
                st.info("記録された学習がまだありません。")
            else:
                _mf_sel = st.multiselect(
                    "実験",
                    [f"{e['name']} ({e['n_runs']} runs)" for e in _mf_exps],
                    default=[f"{_mf_exps[0]['name']} ({_mf_exps[0]['n_runs']} runs)"],
                    key="mf_exp_sel",
                )
                _mf_ids = [e["id"] for e in _mf_exps
                           if f"{e['name']} ({e['n_runs']} runs)" in _mf_sel]
                _mf_runs = list_runs(_mf_ids)

                if not _mf_runs:
                    st.info("選んだ実験に run がありません。")
                else:
                    import pandas as _pd_mf

                    _df_mf = _pd_mf.DataFrame(_mf_runs)
                    _show_cols = [c for c in
                                  ["run_name", "status", "mAP50", "mAP50-95",
                                   "precision", "recall", "top1", "model", "epochs",
                                   "batch", "imgsz", "optimizer", "lr0",
                                   "duration_min"]
                                  if c in _df_mf.columns]
                    st.markdown(f"**{len(_mf_runs)} 件の学習**")
                    st.dataframe(_df_mf[_show_cols], use_container_width=True,
                                 hide_index=True)

                    _mf_pick = st.multiselect(
                        "学習曲線を比較する run（複数選択）",
                        _df_mf["run_name"].tolist(),
                        default=_df_mf["run_name"].tolist()[:min(3, len(_df_mf))],
                        key="mf_run_pick",
                    )
                    _mf_pick_ids = _df_mf[_df_mf["run_name"].isin(_mf_pick)]["run_id"].tolist()

                    if _mf_pick_ids:
                        _mf_metrics = available_metrics(_mf_pick_ids)
                        if not _mf_metrics:
                            st.info("比較できるメトリクスがありません。")
                        else:
                            _mf_metric = st.selectbox(
                                "メトリクス", _mf_metrics,
                                index=preferred_metric_index(_mf_metrics),
                                key="mf_metric")
                            _hist = metric_history(_mf_pick_ids, _mf_metric)
                            if _hist:
                                _df_curve = _pd_mf.DataFrame(_hist).sort_index()
                                _df_curve.index.name = "epoch"
                                st.line_chart(_df_curve)
                                st.caption(
                                    "曲線が途中で平らになっていれば、それ以上エポックを"
                                    "増やしても伸びにくいサインです。"
                                    "val の loss が上がり始めていれば過学習を疑ってください。"
                                )
                            else:
                                st.info("この指標の履歴が記録されていません。")

                    _mf_detail = st.selectbox("設定の詳細を見る run",
                                              _df_mf["run_name"].tolist(),
                                              key="mf_detail_sel")
                    _det_id = _df_mf[_df_mf["run_name"] == _mf_detail]["run_id"].iloc[0]
                    _det = run_detail(_det_id)
                    if _det:
                        _dc1, _dc2 = st.columns(2)
                        with _dc1:
                            st.markdown("**パラメータ**")
                            st.dataframe(
                                _pd_mf.DataFrame(sorted(_det["params"].items()),
                                                 columns=["名前", "値"]),
                                use_container_width=True, hide_index=True, height=240)
                        with _dc2:
                            st.markdown("**最終メトリクス**")
                            st.dataframe(
                                _pd_mf.DataFrame(
                                    sorted((k, round(v, 5))
                                           for k, v in _det["metrics"].items()),
                                    columns=["名前", "値"]),
                                use_container_width=True, hide_index=True, height=240)

        # MLflow が使えないときでも学習経過を見られるようにする
        st.markdown("---")
        st.markdown("**results.csv から比較（MLflow なしでも動きます）**")
        _lc_runs = sorted([p.name for p in MODELS_DIR.iterdir()
                           if p.is_dir() and (p / "results.csv").exists()]) \
            if MODELS_DIR.exists() else []
        if not _lc_runs:
            st.caption("`models/<run>/results.csv` がまだありません。")
        else:
            _lc_pick = st.multiselect("比較する学習", _lc_runs,
                                      default=_lc_runs[:min(2, len(_lc_runs))],
                                      key="lc_pick")
            if _lc_pick:
                _curves = local_results_curves(_lc_pick, MODELS_DIR)
                _all_cols: list[str] = []
                for _df in _curves.values():
                    for c in _df.columns:
                        if c not in _all_cols and c != "epoch":
                            _all_cols.append(c)
                _pref = next((c for c in _all_cols
                              if "map50" in c.lower() and "95" not in c.lower()), None)
                _lc_metric = st.selectbox(
                    "指標", _all_cols,
                    index=_all_cols.index(_pref) if _pref in _all_cols else 0,
                    key="lc_metric")
                import pandas as _pd_lc
                _merged = _pd_lc.DataFrame()
                for _name, _df in _curves.items():
                    if _lc_metric in _df.columns:
                        _merged[_name] = _df[_lc_metric].reset_index(drop=True)
                if not _merged.empty:
                    _merged.index.name = "epoch"
                    st.line_chart(_merged)
                else:
                    st.info("選んだ指標を含む results.csv がありません。")

    with st.expander("📦 既存の学習済みモデルを選択"):
        existing_models = list(MODELS_DIR.rglob("*.pt"))
        if existing_models:
            model_labels = [str(p.relative_to(MODELS_DIR)) for p in existing_models]
            sel_model = st.selectbox("モデルファイル", model_labels)
            if st.button("このモデルを使用", key="use_existing_model"):
                st.session_state.last_model_path = str(MODELS_DIR / sel_model)
                st.success(f"モデルを設定: {st.session_state.last_model_path}")
        else:
            empty_state(
                "選べる学習済みモデルがありません",
                "上の「④ 設定を確認して学習を開始する」から学習すると、ここで選べるようになります。",
            )
