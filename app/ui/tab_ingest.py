# =============================================================================
# Step2: データ取込
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




def render_ingest() -> None:
    st.markdown("""
    <div class="step-banner">
      <div class="sb-title">📤 STEP 2: データ取込</div>
      <div class="sb-prev">← 事前準備: CVATでアノテーションを完了させてください (http://localhost:8080)</div>
      <div class="sb-desc">→ ここでやること: CVATタスクをエクスポート → YOLOデータセット形式に変換</div>
    </div>""", unsafe_allow_html=True)
    st.markdown('<div class="pipeline-card"><h3>📤 CVATタスクエクスポート</h3>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🔄 タスク一覧を取得", use_container_width=True):
            with st.spinner("CVATからタスクを取得中…"):
                st.session_state.cvat_tasks = fetch_cvat_tasks()

    tasks = st.session_state.cvat_tasks
    if not tasks:
        st.info("「タスク一覧を取得」ボタンを押してCVATに接続してください。")
    else:
        # 進捗テーブル
        if tasks:
            st.markdown("#### 📊 アノテーション進捗")
            import pandas as pd
            df = pd.DataFrame(tasks)[["id","name","status","assignee","size"]]
            df.columns = ["ID","タスク名","ステータス","担当者","画像数"]
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.markdown("---")

        # 複数タスク選択
        task_options = {f"[{t['id']}] {t['name']} ({t['size']} items)": t["id"] for t in tasks}
        selected_labels = st.multiselect(
            "エクスポートするタスクを選択（複数可）",
            list(task_options.keys()),
            default=[list(task_options.keys())[0]] if task_options else [],
        )
        selected_ids = [task_options[lbl] for lbl in selected_labels]

        _first_id = selected_ids[0] if selected_ids else "multi"
        export_dir_name = st.text_input(
            "エクスポート先サブディレクトリ名",
            value=f"dataset_{_first_id}_{datetime.now():%Y%m%d}",
        )

        # ─── 手順① CVAT for images 1.1 エクスポート ─────────────────────────
        st.markdown("#### ① CVATエクスポート")
        if st.button("⬇️ エクスポート実行 (CVAT for images 1.1)", type="primary",
                     use_container_width=True,
                     disabled=len(selected_ids) == 0):
            if not selected_ids:
                st.warning("エクスポートするタスクを選択してください。")
            else:
                out_dir = DATA_DIR / export_dir_name
                out_dir.mkdir(parents=True, exist_ok=True)
                all_raw_dirs = []
                with st.spinner("エクスポート中…（最大3分×タスク数）"):
                    for task_id in selected_ids:
                        task_out = out_dir / f"task_{task_id}"
                        task_out.mkdir(parents=True, exist_ok=True)
                        raw_dir = export_cvat_task_raw(task_id, task_out)
                        if raw_dir:
                            all_raw_dirs.append(raw_dir)
                            st.success(f"✅ タスク {task_id} エクスポート完了: `{raw_dir}`")
                        else:
                            st.error(f"タスク {task_id} のエクスポートに失敗しました")

                if all_raw_dirs:
                    # 複数タスクの場合は最初のrawディレクトリをメインとして設定
                    # マージ: 全rawディレクトリのXMLを統合して最初のrawを基準にする
                    if len(all_raw_dirs) == 1:
                        merged_raw = all_raw_dirs[0]
                    else:
                        import shutil as _shutil
                        merged_raw = out_dir / "merged_raw"
                        merged_raw.mkdir(parents=True, exist_ok=True)
                        for src_raw in all_raw_dirs:
                            for item in src_raw.rglob("*"):
                                if item.is_file():
                                    rel = item.relative_to(src_raw)
                                    dst = merged_raw / rel
                                    dst.parent.mkdir(parents=True, exist_ok=True)
                                    if not dst.exists():
                                        _shutil.copy2(item, dst)
                        st.success(f"✅ {len(all_raw_dirs)} タスクを統合: `{merged_raw}`")

                    st.session_state.cvat_raw_dir = str(merged_raw)
                    # 来歴に残すため、どの CVAT タスクから取り込んだかを覚えておく
                    st.session_state["cvat_export_tasks"] = [
                        {"id": t["id"], "name": t["name"], "size": t.get("size")}
                        for t in tasks if t["id"] in selected_ids
                    ]
                    st.session_state.cvat_xml_info = None
                    xml_info = parse_cvat_xml(merged_raw)
                    if xml_info:
                        st.session_state.cvat_xml_info = xml_info

        # ─── 手順② ラベル・タスク種別の設定 ─────────────────────────────────
        if st.session_state.cvat_raw_dir and st.session_state.cvat_xml_info:
            xml_info = st.session_state.cvat_xml_info
            st.markdown("---")
            st.markdown("#### ② ラベルとタスク種別の設定")

            col_stat1, col_stat2, col_stat3 = st.columns(3)
            with col_stat1:
                st.metric("総画像数", xml_info["image_count"])
            with col_stat2:
                st.metric("アノテーション付き", xml_info["annotated_count"])
            with col_stat3:
                ann_type_str = ", ".join(xml_info["annotation_types"]) or "なし"
                st.metric("アノテーション種別", ann_type_str)

            selected_labels = st.multiselect(
                "学習に使用するラベルを選択（順番がクラスID順になります）",
                options=xml_info["labels"],
                default=xml_info["labels"],
            )

            ann_types = set(xml_info.get("annotation_types", []))
            task_type_options = ["detect"]
            if "polygon" in ann_types:
                task_type_options.append("segment")
            if "points" in ann_types:
                task_type_options.append("pose")
            if "tag" in ann_types:
                task_type_options.append("classify")
            if "box" in ann_types or "polygon" in ann_types:
                task_type_options.append("obb")

            col_task, col_val = st.columns(2)
            with col_task:
                task_type = st.selectbox(
                    "タスク種別",
                    task_type_options,
                    help="detect: バウンディングボックス / segment: ポリゴン（box→矩形ポリゴンに変換） / "
                         "pose: キーポイント / classify: 画像分類（CVAT の「タグ」から生成） / "
                         "obb: 回転バウンディングボックス（回転付き box・4点ポリゴンから生成）",
                )
            if "tag" not in ann_types:
                st.caption(
                    "💡 画像分類 (classify) を作るには、CVAT で矩形ではなく"
                    "「タグ（画像単位のラベル）」を付けてエクスポートしてください。"
                )
            with col_val:
                val_ratio = st.slider("バリデーション割合", 0.05, 0.40, 0.20, step=0.05)

            # ─── 手順③ データセット生成 ──────────────────────────────────────
            st.markdown("---")
            st.markdown("#### ③ データセット生成")

            gen_dir_name = st.text_input(
                "生成先ディレクトリ名",
                value=f"yolo_{task_type}_{datetime.now():%Y%m%d_%H%M}",
            )

            if not selected_labels:
                st.warning("少なくとも1つ以上のラベルを選択してください。")
            else:
                if st.button("⚙️ データセット生成", type="primary", use_container_width=True):
                    raw_dir_path = Path(st.session_state.cvat_raw_dir)
                    gen_dir = DATA_DIR / gen_dir_name
                    gen_dir.mkdir(parents=True, exist_ok=True)
                    with st.spinner("YOLOデータセットを生成中…"):
                        result = generate_yolo_dataset(
                            raw_dir=raw_dir_path,
                            xml_info=xml_info,
                            selected_labels=selected_labels,
                            task_type=task_type,
                            out_dir=gen_dir,
                            val_ratio=val_ratio,
                            cvat_tasks=st.session_state.get("cvat_export_tasks"),
                        )
                    if result:
                        yaml_path = result / "data.yaml"
                        st.success("✅ データセット生成完了！")
                        st.info(
                            f"🗂 data.yaml パス（Step3: モデル学習 タブで使用）:\n`{yaml_path}`"
                        )
                        st.code(str(yaml_path), language="text")

    st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("💡 既にRAWデータがある場合（再解析）"):
        manual_raw = st.text_input(
            "既存のraw_dirパス（コンテナ内）",
            placeholder="/workspace/data/dataset_11_20260512/raw",
        )
        if st.button("🔍 XMLを解析", use_container_width=True) and manual_raw:
            raw_p = Path(manual_raw)
            if raw_p.exists():
                xml_info = parse_cvat_xml(raw_p)
                if xml_info:
                    st.session_state.cvat_raw_dir = str(raw_p)
                    st.session_state.cvat_xml_info = xml_info
                    st.success("解析完了。下に②が表示されます。")
                    st.rerun()
            else:
                st.error(f"ディレクトリが存在しません: {raw_p}")

    with st.expander("📁 ローカルからデータを直接追加（CVATなし）"):
        import io as _io_ul
        st.caption(
            "CVATを経由せず、手元の画像やYOLOデータセットZIPを直接 data/ に追加します。"
        )
        _ul_mode = st.radio(
            "アップロード形式",
            ["🗜 ZIPファイル（YOLOデータセット）", "🖼 画像ファイル（複数可）"],
            horizontal=True,
            key="ul_mode",
        )
        _ul_dir_name = st.text_input(
            "保存先ディレクトリ名（data/ 以下に作成）",
            value=f"upload_{datetime.now():%Y%m%d_%H%M}",
            key="ul_dir_name",
        )

        if _ul_mode == "🗜 ZIPファイル（YOLOデータセット）":
            _ul_zip = st.file_uploader(
                "YOLOデータセット ZIP（images/, labels/, data.yaml を含む）",
                type=["zip"],
                key="ul_zip",
            )
            if _ul_zip:
                st.caption(f"選択中: {_ul_zip.name}  ({_ul_zip.size / 1024 / 1024:.1f} MB)")
                if st.button("📤 展開して data/ に保存", key="ul_zip_btn",
                             type="primary", use_container_width=True):
                    _ul_out = DATA_DIR / _ul_dir_name
                    _ul_out.mkdir(parents=True, exist_ok=True)
                    with zipfile.ZipFile(_io_ul.BytesIO(_ul_zip.read()), "r") as _zf:
                        _zf.extractall(_ul_out)
                    st.success(f"✅ 展開完了: `{_ul_out}`")
                    record_dataset_provenance(
                        _ul_out, source="upload_zip",
                        extra={"zip_name": _ul_zip.name,
                               "note": "外部から持ち込んだ YOLO データセット ZIP"},
                    )
                    _ul_yamls = list(_ul_out.rglob("data.yaml"))
                    if _ul_yamls:
                        st.info(f"🗂 data.yaml: `{_ul_yamls[0]}`")
                    else:
                        st.warning("data.yaml が見つかりません。Step3で手動入力が必要です。")
        else:
            _ul_imgs = st.file_uploader(
                "画像ファイル（複数選択可）",
                type=["jpg", "jpeg", "png", "bmp", "tiff"],
                accept_multiple_files=True,
                key="ul_imgs",
            )
            _ul_split = st.radio(
                "保存先スプリット", ["train", "val"], horizontal=True, key="ul_split"
            )
            if _ul_imgs:
                st.caption(f"選択中: {len(_ul_imgs)} ファイル")
                _ul_dst_preview = f"data/{_ul_dir_name}/images/{_ul_split}/"
                if st.button(f"📤 {_ul_dst_preview} に保存", key="ul_imgs_btn",
                             type="primary", use_container_width=True):
                    _ul_out = DATA_DIR / _ul_dir_name / "images" / _ul_split
                    _ul_out.mkdir(parents=True, exist_ok=True)
                    for _f in _ul_imgs:
                        (_ul_out / _f.name).write_bytes(_f.getbuffer())
                    st.success(f"✅ {len(_ul_imgs)} ファイルを保存: `{_ul_out}`")
                    record_dataset_provenance(
                        DATA_DIR / _ul_dir_name, source="upload_images",
                        extra={"note": f"{_ul_split} に画像 {len(_ul_imgs)} 枚を直接アップロード"},
                    )
                    st.info("アノテーションを付与する場合は CVATにアップロード後、Step2からエクスポートしてください。")


    # ── +α: チーム共通ラベルエクスポート ──────────────────────────────────────
    st.markdown("---")
    with st.expander("🏷️ チーム共通ラベルのエクスポート（+α）", expanded=False):
        st.caption("複数のCVATタスクからラベルを収集し、チーム内で共有できる形式でダウンロードできます。")

        _le_tasks = st.session_state.cvat_tasks
        if not _le_tasks:
            st.info("先に「タスク一覧を取得」を実行してください。")
        else:
            _le_opts = {f"{t['name']}  (ID: {t['id']})": t["id"] for t in _le_tasks}
            _le_selected = st.multiselect(
                "ラベルを収集するタスクを選択（複数可）",
                options=list(_le_opts.keys()),
                key="le_task_select",
            )

            if _le_selected:
                if st.button("🔍 ラベルを取得", key="le_fetch_btn", use_container_width=True):
                    _le_ids = [_le_opts[k] for k in _le_selected]
                    with st.spinner("ラベル取得中..."):
                        st.session_state["le_labels_by_task"] = fetch_cvat_task_labels(_le_ids)

        if st.session_state.get("le_labels_by_task"):
            _le_by_task: dict = st.session_state["le_labels_by_task"]

            # 全ラベルを重複排除して収集
            _le_all: list[str] = []
            for _lbls in _le_by_task.values():
                for _l in _lbls:
                    if _l not in _le_all:
                        _le_all.append(_l)

            st.markdown("**含めるラベルを選択してください：**")
            # タスク別ラベルは参考表示のみ（チェックボックスは重複排除済みリストで一度だけ描画）
            for _tn, _lbls in _le_by_task.items():
                st.caption(f"📋 {_tn}：{', '.join(_lbls)}")
            st.markdown("---")
            _le_cols = st.columns(3)
            for _ci, _l in enumerate(_le_all):
                with _le_cols[_ci % 3]:
                    if f"le_chk_{_l}" not in st.session_state:
                        st.session_state[f"le_chk_{_l}"] = True
                    st.checkbox(_l, key=f"le_chk_{_l}")

            _le_chosen = [_l for _l in _le_all if st.session_state.get(f"le_chk_{_l}", True)]

            if _le_chosen:
                st.markdown(f"**選択中: {len(_le_chosen)} ラベル** — `{', '.join(_le_chosen)}`")
                _le_c1, _le_c2, _le_c3 = st.columns(3)
                with _le_c1:
                    _le_yaml = "names:\n" + "".join(f"  - {l}\n" for l in _le_chosen)
                    st.download_button(
                        "📥 YAML形式",
                        data=_le_yaml,
                        file_name="labels.yaml",
                        mime="text/yaml",
                        key="le_dl_yaml",
                        use_container_width=True,
                    )
                with _le_c2:
                    st.download_button(
                        "📥 TXT形式",
                        data="\n".join(_le_chosen),
                        file_name="labels.txt",
                        mime="text/plain",
                        key="le_dl_txt",
                        use_container_width=True,
                    )
                with _le_c3:
                    _le_cvat_json = json.dumps(
                        [{"name": l, "attributes": [], "type": "any", "sublabels": []} for l in _le_chosen],
                        ensure_ascii=False, indent=2
                    )
                    st.download_button(
                        "📥 CVAT JSON形式",
                        data=_le_cvat_json,
                        file_name="labels_cvat.json",
                        mime="application/json",
                        key="le_dl_cvat",
                        use_container_width=True,
                    )

                st.markdown("---")
                st.markdown("##### CVATで新規タスクを作成するときの手順")
                st.markdown(f"""
    <div class="step-banner">
      <div class="sb-title">📋 ラベルの共有方法</div>
      <div class="sb-desc">ダウンロードした <code>labels_cvat.json</code> を使うと、CVATのラベル設定を一括で読み込めます。</div>
    </div>""", unsafe_allow_html=True)
                st.markdown("""
    1. `http://localhost:8080` にアクセスしてログイン
    2. **Tasks** → **+** ボタンで新規タスク作成画面を開く
    3. タスク名・画像等を設定後、**Labels** セクションを開く
    4. **Raw** タブをクリックし、ダウンロードした `labels_cvat.json` の内容を貼り付ける
    5. **Done** をクリックしてラベルを確定する
    """)
                st.info("💡 チームメンバー全員が同じ `labels_cvat.json` を使うことで、ラベル名の表記ゆれを防げます。")
            else:
                st.warning("1つ以上のラベルを選択してください。")
