# =============================================================================
# Step1: アノテーション
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
from .widgets import empty_state, metric_row, open_folder, show_error




def render_annotate() -> None:
    st.markdown(f"""
    <div class="step-banner">
      <div class="sb-title">🏷 STEP 1: アノテーション</div>
      <div class="sb-prev">← 作業場所: CVAT ({CVAT_WEB}) — ここはその作業を支援する管理画面です</div>
      <div class="sb-desc">→ ここでやること: 学習済みモデルを自動アノテーションに載せる / チームの進捗を把握する</div>
    </div>""", unsafe_allow_html=True)

    # ── 自動アノテーションモデル (Nuclio) ──────────────────────────────────
    st.markdown('<div class="section-head"><h3>🤖 自動アノテーションモデル (Nuclio)</h3></div>',
                unsafe_allow_html=True)
    st.caption(
        "ここにデプロイしたモデルが、CVAT の「Actions → Automatic annotation」の選択肢に現れます。"
        "手作業のアノテーションを、モデルの推論結果を修正する作業に置き換えられます。"
    )

    _sl = serverless_status()
    if not serverless_ready():
        _sl_miss = []
        if not _sl["deploy_sh"]:   _sl_miss.append("`serverless/deploy.sh` が見つかりません")
        if not _sl["nuctl"]:       _sl_miss.append("`serverless/bin/nuctl` が見つかりません")
        if not _sl["docker_sock"]: _sl_miss.append("`/var/run/docker.sock` がマウントされていません")
        if not _sl["docker_cli"]:  _sl_miss.append("コンテナ内に `docker` コマンドがありません")
        st.warning(
            "⚠ このコンテナからは Nuclio へデプロイできません。\n\n"
            + "\n".join(f"- {m}" for m in _sl_miss)
            + "\n\n`docker-compose.yml` の `streamlit_app` に serverless / docker.sock の"
              "マウントを追加し、`docker compose up -d streamlit_app` を実行してください。"
        )
    else:
        _dep_state, _dep_lock = _get_deploy_shared()
        with _dep_lock:
            _dep_running  = _dep_state["running"]
            _dep_log      = list(_dep_state["log"])
            _dep_error    = _dep_state["error"]
            _dep_target   = _dep_state["target"]
            _dep_finished = _dep_state["finished"]

        # --- デプロイ実行中 / 直後のログ表示 ---
        if _dep_running or _dep_finished:
            if _dep_running:
                st.info(f"⏳ デプロイ実行中: `{_dep_target}`  "
                        "（初回はイメージのビルドに数分〜十数分かかります）")
            elif _dep_error:
                show_error(_dep_error, prefix="❌ デプロイ失敗: ")
            else:
                st.success(f"✅ デプロイ完了: `{_dep_target}`")

            st.code("\n".join(_dep_log[-40:]) or "(出力待ち)", language="bash")

            if not _dep_running:
                if st.button("ログを閉じる", key="dep_clear_log"):
                    with _dep_lock:
                        _dep_state["finished"] = False
                        _dep_state["log"] = []
                        _dep_state["error"] = None
                    st.rerun()

        # --- デプロイ済み関数の一覧 ---
        _fns  = cached_nuclio_functions()
        _defs = list_serverless_defs()
        _def_by_fn = {d["function_name"]: d for d in _defs if d["function_name"]}

        st.markdown(f"**デプロイ済み: {len(_fns)} 件**")
        if not _fns:
            st.info("まだ関数がデプロイされていません。下の「新しいモデルをデプロイ」から追加してください。")
        else:
            for _fn in _fns:
                _d = _def_by_fn.get(_fn["name"], {})
                with st.container(border=True):
                    _fc1, _fc2, _fc3, _fc4 = st.columns([5, 2, 2, 2])
                    with _fc1:
                        _badge = {"ready": "🟢", "error": "🔴", "building": "🟡"}.get(
                            _fn["state"], "⚪")
                        st.markdown(f"{_badge} **{_fn['display']}**")
                        st.caption(f"`{_fn['name']}`")
                        st.caption("🏷 ラベル: " + (", ".join(_fn["labels"]) or "—"))
                        if _d.get("model_weights"):
                            _mr_ok = "✅" if _d.get("model_exists") else "⚠ 見つかりません"
                            _mnt = "（マウント）" if _d.get("kind") == "sam3" else ""
                            st.caption(f"📦 モデル: `models/{_d['model_weights']}` {_mr_ok}{_mnt}")
                            if _d.get("deployed_at"):
                                st.caption(f"🕐 デプロイ: {_d['deployed_at'][:19].replace('T', ' ')}")
                            # 重みはビルド時に焼き込まれるので、models/ を差し替えても
                            # 再デプロイしない限り古いモデルが動き続ける。
                            # SAM 3 はマウントなので焼き込みではないが、関数プロセスは
                            # 起動時に読んだモデルを持ち続けるので、結局読み直しが要る
                            if _d.get("weights_changed"):
                                st.warning(
                                    "⚠ **重みが差し替わっています。**重みはマウントしているため"
                                    "再ビルドは不要ですが、関数は起動時に読んだモデルを"
                                    "使い続けます。右の「🔄 再デプロイ」で読み直してください。"
                                    if _d.get("kind") == "sam3" else
                                    "⚠ **重みが更新されています。**いま動いているのは"
                                    "デプロイ時点のモデルです。右の「🔄 再デプロイ」を"
                                    "押すと最新のモデルに入れ替わります。")
                            elif not _d.get("deployed_sha1"):
                                st.caption("ℹ この関数はデプロイ記録がありません"
                                           "（更新の検知は次回のデプロイから効きます）")
                    with _fc2:
                        st.metric("状態", _fn["state"] or "—")
                    with _fc3:
                        st.metric("実行", "GPU" if _fn["gpu"] else "CPU")
                    with _fc4:
                        if _d.get("dir"):
                            if st.button("🔄 再デプロイ", key=f"redeploy_{_fn['name']}",
                                         use_container_width=True, disabled=_dep_running,
                                         type="primary" if _d.get("weights_changed") else "secondary",
                                         help="モデルを差し替えた後に実行すると最新の重みが反映されます"):
                                start_deploy(_d["dir"], use_gpu=_fn["gpu"])
                                cached_nuclio_functions.clear()
                                st.rerun()
                        if st.button("🗑 削除", key=f"delfn_{_fn['name']}",
                                     use_container_width=True, disabled=_dep_running):
                            _ok, _msg = delete_nuclio_function(_fn["name"])
                            cached_nuclio_functions.clear()
                            if _ok:
                                st.success(f"削除しました: {_fn['name']}")
                            else:
                                st.error(f"削除に失敗: {_msg}")
                            st.rerun()

        # --- 新規デプロイ ---
        st.markdown("---")
        st.markdown("**➕ 新しいモデルをデプロイ**")

        _dep_models = sorted(model_weight_files(), key=lambda p: p.stat().st_mtime,
                             reverse=True) if MODELS_DIR.exists() else []

        # 他 PC で学習した .pt をここで直接取り込めるようにする。
        # 「データ管理タブで取り込んでから戻ってくる」という往復をなくすため
        _dep_src = st.radio(
            "モデルの選び方",
            ["📂 models/ から選ぶ", "📤 .pt をアップロード"],
            horizontal=True, key="dep_src",
            index=0 if _dep_models else 1,
        )

        if _dep_src == "📤 .pt をアップロード":
            _up_pt = st.file_uploader(
                "学習済みの重み (.pt)", type=["pt"], key="dep_up_pt",
                help="他の PC で学習したモデルをそのまま取り込めます",
            )
            _up_name = st.text_input(
                "モデル名（`models/` 以下に作るディレクトリ名）",
                value=f"imported_{datetime.now():%Y%m%d_%H%M}",
                key="dep_up_name",
            ).strip()
            if _up_pt:
                st.caption(f"選択中: `{_up_pt.name}` ({_up_pt.size / 1024 / 1024:.1f} MB)"
                           f" → `models/{safe_run_name(_up_name)}/weights/`")
                if st.button("📥 取り込む", key="dep_up_btn", use_container_width=True,
                             disabled=_dep_running):
                    _saved, _err = import_model_weights(
                        _up_name, [(_up_pt.name, _up_pt.getbuffer())])
                    if _err:
                        st.error(f"❌ {_err}")
                    else:
                        # 取り込んだものをそのまま選択状態にして、続けてデプロイできるようにする
                        st.session_state["dep_model_sel"] = str(
                            _saved[0].relative_to(MODELS_DIR))
                        st.session_state["dep_src"] = "📂 models/ から選ぶ"
                        st.success(f"✅ 取り込みました: `{_saved[0].name}`")
                        st.rerun()
            st.caption("取り込むとこの下でそのままデプロイできます。")

        if not _dep_models:
            st.info("models/ に .pt がありません。"
                    "上の「📤 .pt をアップロード」から取り込むか、Step3 で学習してください。")
        else:
            _dep_map = {str(p.relative_to(MODELS_DIR)): p for p in _dep_models}
            _dep_keys = list(_dep_map.keys())
            # 取り込み直後は、その重みを選んだ状態で始める。
            # key を使うウィジェットに index を併せて渡しても効かないので、
            # session_state 側を先に整えておく
            if st.session_state.get("dep_model_sel") not in _dep_keys:
                st.session_state["dep_model_sel"] = _dep_keys[0]
            _dep_sel = st.selectbox("デプロイするモデル", _dep_keys, key="dep_model_sel")
            _dep_path = _dep_map[_dep_sel]
            # 重みは models/ からの相対パスで渡す。
            # 取り込んだモデルはファイル名が best.pt とは限らないため、
            # run 名だけでは指し切れない（例: imported_xxx/weights/my_model.pt）
            _dep_rel = str(_dep_path.relative_to(MODELS_DIR))
            _dep_run = (_dep_path.parent.parent.name
                        if _dep_path.parent.name == "weights"
                        else _dep_path.parent.name)

            # クラス名はモデルから取得する（= CVAT に出るラベル定義になる）
            _dep_meta = read_model_meta(_dep_path)
            if _dep_meta is None:
                with st.spinner("モデルのクラス名を読み込み中…"):
                    _dep_meta = inspect_model_file(_dep_path)
            _dep_classes = _dep_meta.get("names") or []

            if not _dep_meta.get("ok") or not _dep_classes:
                st.error(
                    "❌ このモデルからクラス名を取得できませんでした。"
                    "ラベル定義を作れないためデプロイできません。\n\n"
                    f"{_dep_meta.get('error') or 'クラス名が空です'}"
                )
            else:
                _dep_task = _dep_meta.get("task") or "detect"
                _dep_shape = "polygon（ポリゴン）" if _dep_task == "segment" \
                    else "rectangle（矩形）"
                st.success(f"🏷 ラベル定義（モデルのクラス名から自動生成）: "
                           f"**{', '.join(_dep_classes)}**")
                st.caption(
                    f"タスク種別: `{_dep_task}` → CVAT には **{_dep_shape}** として返します。"
                )
                st.caption(
                    "⚠ CVAT タスク側のラベル名がこれと一致していないと、"
                    "自動アノテーションの結果が反映されません。"
                )

                # 名前はモデル名から機械的に決まるので、既定では見せない。
                # 「モデルを選んで押すだけ」で終わるようにして、
                # 変えたい人だけが開く（入力欄が並ぶと何を決めるべきか分からなくなる）
                _dep_dir_def  = slugify_function_name(_dep_run)
                _dep_disp_def = f"{_dep_run} (custom)"
                _dep_dir, _dep_disp = _dep_dir_def, _dep_disp_def

                if st.checkbox("⚙ 名前を変える（任意）", key="dep_rename"):
                    _dc1, _dc2 = st.columns(2)
                    with _dc1:
                        _dep_dir = st.text_input(
                            "関数ディレクトリ名 (`serverless/custom/` 以下)",
                            value=_dep_dir_def, key="dep_fn_dir",
                        ).strip() or _dep_dir_def
                    with _dc2:
                        _dep_disp = st.text_input(
                            "CVAT に表示する名前",
                            value=_dep_disp_def, key="dep_fn_disp",
                        ).strip() or _dep_disp_def

                _dep_gpu = st.radio(
                    "実行モード", ["GPU", "CPU"], horizontal=True, key="dep_gpu_mode",
                    help="GPU 実行には Docker daemon の default-runtime=nvidia が必要です"
                         "（serverless/README.md 参照）。CPU なら前提なしで動きます。",
                ) == "GPU"

                _dep_slug = slugify_function_name(_dep_dir) if _dep_dir else ""
                if _dep_slug:
                    _exists_def = (SERVERLESS_DIR / "custom" / _dep_dir).exists()
                    st.caption(f"Nuclio 関数名: `custom-{_dep_slug}`"
                               + ("　⚠ 同名の定義が既にあります（上書きされます）" if _exists_def else ""))

                if st.button("🚀 CVAT にデプロイ", type="primary", use_container_width=True,
                             disabled=_dep_running or not _dep_dir, key="dep_run_btn"):
                    _out_dir, _fn_name = generate_function_files(
                        fn_dir=_dep_dir,
                        model_run=_dep_run,
                        class_names=_dep_classes,
                        display_name=_dep_disp,
                        task=_dep_meta.get("task") or "detect",
                        weights_rel=_dep_rel,
                    )
                    start_deploy(_dep_dir, use_gpu=_dep_gpu)
                    cached_nuclio_functions.clear()
                    st.rerun()

        # --- SAM（学習不要のゼロショットモデル）---
        st.markdown("---")
        st.markdown("**🧩 学習不要のモデルを使う（SAM）**")
        st.caption(
            "SAM は学習なしで使えるセグメンテーションモデルです。"
            "自分のデータがまだ少なく、学習済みモデルを用意できない段階でも、"
            "アノテーションの下書きを作れます。"
        )

        _sm_key = st.radio(
            "モデル",
            list(SAM_MODELS.keys()),
            # 知らない値が来ても落とさない（定義から版を消したあとの
            # 古いセッションでも画面を出し続けるため）
            format_func=lambda k: SAM_MODELS.get(k, {}).get("display", str(k)),
            horizontal=True, key="sam_model",
        )
        if _sm_key not in SAM_MODELS:
            _sm_key = next(iter(SAM_MODELS))
        _sm = SAM_MODELS[_sm_key]
        _sm_st = sam_model_status(_sm_key)
        st.caption(_sm["note"])

        if not _sm_st["ready"]:
            # SAM 3 は各自が重みを取ってくる必要がある
            st.info(
                "**まだ重みが置かれていません。**\n\n"
                "SAM 3 の重みは Meta の承認が要るため、配布物に含められません。\n\n"
                "1. https://huggingface.co/facebook/sam3 でアクセス承認を受ける"
                "（**手動レビュー**なので即時ではありません）\n"
                f"2. `{_sm_st['name']}`（約 3.45GB）をダウンロードする\n"
                f"3. 下のフォルダに置く → `models/{SAM3_DIR.name}/{_sm_st['name']}`\n\n"
                "置いたらこの画面を再読み込みしてください。\n\n"
                "**承認を待つ間は SAM 2.1 が使えます**（承認も重みの配置も不要）。"
            )
            open_folder(SAM3_DIR, key="sam_weights_dir",
                        label="📂 重みの置き場所を開く")
        else:
            _sm_wname = ""
            if _sm_st.get("auto"):
                st.success("✅ 重みはデプロイ時に自動ダウンロードされます（準備作業は不要です）")
                if _sm.get("sizes"):
                    _sm_size = st.selectbox(
                        "モデルサイズ",
                        list(_sm["sizes"].keys()),
                        index=list(_sm["sizes"].keys()).index(_sm["default_size"]),
                        key="sam_size",
                        help="大きいほど精度が上がりますが、GPU メモリと待ち時間が増えます",
                    )
                    _sm_wname = _sm["sizes"][_sm_size]
                    st.caption(f"重み: `{_sm_wname}`")
            else:
                st.success(
                    f"✅ 重み: `models/{SAM3_DIR.name}/{_sm_st['name']}` "
                    f"({_sm_st['size'] / 1024**3:.2f} GB)"
                )

            # 使い方は版によって選べる数が違う（SAM 2 はクリックのみ）
            _sm_variants = _sm["variants"]
            if len(_sm_variants) == 1:
                _sv_key = _sm_variants[0]
                st.caption(f"使い方: **{SAM_VARIANTS[_sv_key]['label']}**"
                           "（SAM 2 はテキストでの検出に対応していません）")
            else:
                _sv_key = st.radio(
                    "使い方",
                    _sm_variants,
                    format_func=lambda k: SAM_VARIANTS[k]["label"],
                    horizontal=True, key="sam_variant",
                )
            _sv = SAM_VARIANTS[_sv_key]
            st.caption(f"CVAT では「**{_sv['where']}**」に現れます。")

            _sm_pairs: list[tuple[str, str]] = []
            _sm_ready = True

            if _sv_key == "concept":
                st.caption(
                    "検出したいものを 1 行に 1 つ書きます。"
                    "`CVAT のラベル名 = 英語プロンプト` の形で、左が CVAT のタスクに"
                    "付けるラベル名、右が SAM に渡す言葉です（`=` を省くと同じ語を使います）。"
                    "**SAM 3 は英語の短い名詞句を前提**にしているので、"
                    "右側は `red fruit` のように書いてください。"
                )
                # key と value を同時に渡すと Streamlit が警告を出すので、
                # 初期値は session_state 側に入れておく
                st.session_state.setdefault("sam_prompts_text", "猫 = cat")
                _sm_text = st.text_area(
                    "検出したいもの", key="sam_prompts_text", height=110,
                )
                _sm_pairs, _sm_bad = parse_sam_prompt_lines(_sm_text)
                if _sm_bad:
                    st.warning("⚠ 読み取れなかった行があります（重複も含む）:\n\n"
                               + "\n".join(f"- `{b}`" for b in _sm_bad))
                if _sm_pairs:
                    st.caption("🏷 CVAT に出るラベル: **"
                               + ", ".join(lb for lb, _ in _sm_pairs) + "**")
                    st.caption("⚠ CVAT タスク側のラベル名がこれと一致していないと、"
                               "自動アノテーションの結果が反映されません。")
                else:
                    _sm_ready = False
                    st.error("❌ 検出したいものを 1 つ以上書いてください。")
            else:
                st.caption(
                    "対象をボックスで囲むか、内側に正の点・外側に負の点を打つと、"
                    "その 1 個だけをマスクにして返します。ラベルの指定は要りません。"
                )

            _smc1, _smc2 = st.columns(2)
            with _smc1:
                _sm_gpu = st.radio(
                    "実行モード", ["GPU", "CPU"], horizontal=True, key="sam_gpu_mode",
                    help="CPU でも動きますが、1 枚あたり数十秒かかります",
                ) == "GPU"
            with _smc2:
                _sm_half = st.checkbox(
                    "FP16 で読み込む", value=False, key="sam_half",
                    help="GPU メモリの使用量を減らせます。精度がわずかに変わることがあります",
                )

            if _sm_gpu:
                st.caption(
                    "⚠ SAM は GPU メモリを使います（SAM 3 は数 GB）。"
                    "**複数の使い方を同時にデプロイすると、その分だけ常時占有します。**"
                    "学習と同時に使うとメモリが足りなくなることがあるので、"
                    "使わないほうは削除しておくのが安全です。"
                )

            st.caption(f"Nuclio 関数名: `{_sm_key}-{_sv_key}`　"
                       "初回はビルドとモデル読み込みで 10 分以上かかることがあります。")

            if st.button("🚀 デプロイ", type="primary", use_container_width=True,
                         disabled=_dep_running or not _sm_ready, key="sam_deploy_btn"):
                generate_sam_function_files(
                    version=_sm_key,
                    variant=_sv_key,
                    pairs=_sm_pairs,
                    weights_name=_sm_wname,
                    half=_sm_half,
                )
                start_deploy(f"{_sm_key}-{_sv_key}", use_gpu=_sm_gpu)
                cached_nuclio_functions.clear()
                st.rerun()

        if _dep_running:
            # ここで st.rerun() すると以降のタブが描画されないため予約だけする
            request_rerun_poll()

    st.markdown(
        f'<div style="margin-top:8px"><a href="{NUCLIO_WEB}" target="_blank">'
        f'🔗 Nuclio ダッシュボードで詳細を見る</a></div>',
        unsafe_allow_html=True,
    )

    # ── 新規タスク作成（アノテーションの入口）──────────────────────────────
    st.markdown('<div class="section-head"><h3>➕ CVAT に新しいタスクを作る</h3></div>',
                unsafe_allow_html=True)
    st.caption(
        "アノテーションしたい画像から CVAT のタスクを直接作ります。"
        "CVAT の画面を開かずに、ここからアノテーションを始められます。"
    )

    _nt_src = st.radio(
        "画像の取得元",
        ["📤 画像をアップロード", "📂 data/ のディレクトリから"],
        horizontal=True, key="nt_src",
    )

    _nt_images: list[Path] = []
    _nt_tmp = PREDICTIONS_DIR / "_newtask_uploads"

    if _nt_src == "📤 画像をアップロード":
        _nt_files = st.file_uploader(
            "アノテーションする画像（複数選択可）",
            type=["jpg", "jpeg", "png", "bmp", "tiff"],
            accept_multiple_files=True, key="nt_files",
        )
        if _nt_files:
            _nt_tmp.mkdir(parents=True, exist_ok=True)
            _cur = {f.name for f in _nt_files}
            _saved = {f.name for f in _nt_tmp.iterdir() if f.is_file()}
            if _cur != _saved:
                for _f in list(_nt_tmp.iterdir()):
                    _f.unlink()
                for _f in _nt_files:
                    (_nt_tmp / _f.name).write_bytes(_f.getbuffer())
            _nt_images = sorted(p for p in _nt_tmp.iterdir() if p.is_file())
            st.caption(f"✅ {len(_nt_images)} 枚を選択中")
    else:
        _nt_dirs = _find_image_dirs(DATA_DIR) if DATA_DIR.exists() else []
        if not _nt_dirs:
            empty_state(
                "CVAT に送れる画像が data/ にありません",
                "「📤 Step2: データ取込」の「📁 ローカルからデータを直接追加」で画像を入れてください。",
            )
        else:
            _nt_dir_sel = st.selectbox(
                "画像ディレクトリ",
                [str(d.relative_to(DATA_DIR)) for d in _nt_dirs], key="nt_dir")
            _nt_dir = DATA_DIR / _nt_dir_sel
            _nt_all = sorted(p for p in _nt_dir.iterdir()
                             if p.is_file() and p.suffix.lower() in IMG_EXTS)
            _nt_limit = st.number_input(
                "使用する枚数（先頭から）", 1, max(len(_nt_all), 1),
                min(len(_nt_all), 100), key="nt_limit",
                help="1タスクが大きすぎると作業しづらいので、分割して作るのがおすすめです")
            _nt_images = _nt_all[:int(_nt_limit)]
            st.caption(f"ディレクトリ内 {len(_nt_all)} 枚中 {len(_nt_images)} 枚を使用")

    _ntc1, _ntc2 = st.columns([2, 1])
    with _ntc1:
        _nt_name = st.text_input(
            "タスク名", value=f"annotate_{datetime.now():%Y%m%d_%H%M}", key="nt_name")
    with _ntc2:
        _nt_shape = st.selectbox(
            "アノテーション形式",
            ["rectangle", "polygon", "points", "tag", "any"], key="nt_shape",
            help="rectangle: 物体検出 / polygon: セグメンテーション / "
                 "points: キーポイント / tag: 画像分類 / any: 何でも",
        )

    # ラベルは既存タスクから引き継げるようにする（表記ゆれを防ぐ）
    _nt_known: list[str] = []
    _nt_prev = st.session_state.get("le_labels_by_task") or {}
    for _lbls in _nt_prev.values():
        for _l in _lbls:
            if _l not in _nt_known:
                _nt_known.append(_l)
    _nt_default = ", ".join(_nt_known) if _nt_known else ""
    _nt_labels_raw = st.text_input(
        "ラベル（カンマ区切り）", value=_nt_default, key="nt_labels",
        help="既存タスクと同じ名前にしてください。"
             "自動アノテーションを使う場合は、モデルのクラス名とも一致させる必要があります。",
    )
    _nt_labels = [s.strip() for s in _nt_labels_raw.split(",") if s.strip()]
    if _nt_known and not _nt_labels_raw.strip():
        st.caption(f"💡 取得済みのラベル: {', '.join(_nt_known)}")

    if st.button(f"➕ CVAT にタスクを作成（{len(_nt_images)} 枚）",
                 type="primary", use_container_width=True,
                 disabled=not _nt_images or not _nt_labels or not _nt_name.strip(),
                 key="nt_create"):
        with st.spinner("CVAT にタスクを作成中…（画像アップロード中）"):
            st.session_state["nt_result"] = create_cvat_task_from_images(
                _nt_name.strip(), _nt_images, _nt_labels, label_type=_nt_shape)

    _nt_res = st.session_state.get("nt_result")
    if _nt_res:
        if _nt_res["ok"]:
            st.success(f"✅ タスクを作成しました（ID: {_nt_res['task_id']} / "
                       f"{_nt_res['n_images']} 枚 / ラベル: {', '.join(_nt_res['labels'])}）")
            st.markdown(f"👉 [CVAT でアノテーションを始める]({_nt_res['url']})")
            st.caption(
                "自動アノテーションモデルをデプロイ済みなら、CVAT の "
                "「Actions → Automatic annotation」で下書きを作れます。"
            )
        else:
            show_error(_nt_res["error"], prefix="❌ 作成に失敗しました: ")


    # ── アノテーション進捗 ────────────────────────────────────────────────
    st.markdown('<div class="section-head"><h3>📊 アノテーション進捗</h3></div>', unsafe_allow_html=True)

    st.caption(
        "進捗はジョブ単位で集計しています。CVAT はタスクを複数のジョブに分割し、"
        "担当者もジョブ単位で割り当てるため、タスクの status より実態に近い数字が出ます。"
    )
    _pc1, _pc2 = st.columns([3, 1])
    with _pc2:
        if st.button("🔄 CVATから進捗を取得", use_container_width=True, key="anno_fetch_tasks"):
            with st.spinner("CVATからタスク・ジョブを取得中…"):
                st.session_state.cvat_tasks = fetch_cvat_tasks()
                st.session_state.cvat_jobs  = fetch_cvat_jobs()

    _anno_tasks = st.session_state.cvat_tasks
    _anno_jobs  = st.session_state.cvat_jobs
    if not _anno_jobs and not _anno_tasks:
        st.info("「CVATから進捗を取得」を押すと、ジョブ単位の進捗と担当者別の状況を表示します。")
    elif not _anno_jobs:
        st.warning("ジョブ情報を取得できませんでした。もう一度「CVATから進捗を取得」を試してください。")
    else:
        import pandas as _pd_anno

        _df_j = _pd_anno.DataFrame(_anno_jobs)
        _df_j["担当者"] = _df_j["assignee"].replace("", None).fillna("（未割当）")

        _total_f = int(_df_j["frames"].fillna(0).sum())
        _done_f  = int(_df_j.loc[_df_j["state"] == "completed", "frames"].fillna(0).sum())
        _rate    = (_done_f / _total_f * 100) if _total_f else 0.0

        metric_row([
            ("タスク数",     len(_anno_tasks) or _df_j["task_id"].nunique()),
            ("ジョブ数",     len(_df_j)),
            ("総フレーム数", f"{_total_f:,}"),
            ("完了率",       f"{_rate:.1f}%"),
        ])
        st.progress(min(_rate / 100, 1.0),
                    text=f"完了 {_done_f:,} / {_total_f:,} フレーム（{_rate:.1f}%）")

        # state / stage の内訳
        _st1, _st2 = st.columns(2)
        with _st1:
            st.markdown("**進行状態 (state)**")
            _by_state = _df_j.groupby("state").agg(
                ジョブ数=("job_id", "count"), フレーム数=("frames", "sum")
            ).reset_index().rename(columns={"state": "状態"})
            st.dataframe(_by_state, use_container_width=True, hide_index=True)
        with _st2:
            st.markdown("**工程 (stage)**")
            _by_stage = _df_j.groupby("stage").agg(
                ジョブ数=("job_id", "count"), フレーム数=("frames", "sum")
            ).reset_index().rename(columns={"stage": "工程"})
            st.dataframe(_by_stage, use_container_width=True, hide_index=True)

        # 担当者別（ジョブ単位。4人以上で分担するときの主指標）
        st.markdown("**👥 担当者別**")
        _by_user = _df_j.groupby("担当者").agg(
            担当ジョブ数=("job_id", "count"),
            フレーム数=("frames", "sum"),
            完了ジョブ数=("state", lambda s: int((s == "completed").sum())),
            完了フレーム数=("frames", "sum"),   # 後で上書きする
        ).reset_index()
        _done_by_user = (_df_j[_df_j["state"] == "completed"]
                         .groupby("担当者")["frames"].sum())
        _by_user["完了フレーム数"] = _by_user["担当者"].map(_done_by_user).fillna(0).astype(int)
        _by_user["完了率"] = (
            _by_user["完了フレーム数"] / _by_user["フレーム数"].replace(0, 1) * 100
        ).round(1).astype(str) + "%"
        st.dataframe(_by_user.sort_values("フレーム数", ascending=False),
                     use_container_width=True, hide_index=True)

        _unassigned = _df_j[_df_j["担当者"] == "（未割当）"]
        if len(_unassigned) > 0:
            st.warning(
                f"⚠ 未割当のジョブが {len(_unassigned)} 件 "
                f"（{int(_unassigned['frames'].sum()):,} フレーム）あります。"
                f"CVAT 側で担当者を割り当ててください。"
            )

        # タスク別の進捗
        st.markdown("**📋 タスク別の進捗**")
        _task_prog = _df_j.groupby(["task_id", "task_name"]).agg(
            ジョブ数=("job_id", "count"),
            フレーム数=("frames", "sum"),
        ).reset_index()
        _done_by_task = (_df_j[_df_j["state"] == "completed"]
                         .groupby("task_id")["frames"].sum())
        _task_prog["完了フレーム"] = _task_prog["task_id"].map(_done_by_task).fillna(0).astype(int)
        _task_prog["進捗"] = (
            _task_prog["完了フレーム"] / _task_prog["フレーム数"].replace(0, 1)
        ).round(3)
        _task_prog["担当者"] = _task_prog["task_id"].map(
            _df_j.groupby("task_id")["担当者"].agg(lambda s: ", ".join(sorted(set(s))))
        )
        _task_prog = _task_prog.rename(columns={"task_id": "ID", "task_name": "タスク名"})

        _only_incomplete = st.checkbox("未完了のタスクだけ表示", value=False,
                                       key="anno_only_incomplete")
        _tp_show = _task_prog[_task_prog["進捗"] < 1.0] if _only_incomplete else _task_prog
        st.dataframe(
            _tp_show[["ID", "タスク名", "担当者", "ジョブ数", "フレーム数", "完了フレーム", "進捗"]],
            use_container_width=True, hide_index=True,
            column_config={"進捗": st.column_config.ProgressColumn(
                "進捗", min_value=0.0, max_value=1.0, format="%.0f%%")},
        )

        st.caption(
            f"CVAT で作業する → [{CVAT_WEB}]({CVAT_WEB})　"
            "／ アノテーションが終わったタスクは「📤 Step2: データ取込」でエクスポートします。"
        )

