# =============================================================================
# データ管理
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




def render_manage() -> None:
    import shutil

    st.markdown('<div class="pipeline-card"><h3>📁 データ管理</h3>', unsafe_allow_html=True)

    # --- data/ データセット一覧 ---
    st.markdown("#### 学習データセット (`data/`)")
    datasets = sorted(DATA_DIR.iterdir()) if DATA_DIR.exists() else []
    datasets = [d for d in datasets if d.is_dir()]
    if not datasets:
        st.info("data/ にデータセットがありません。")
    else:
        for ds in datasets:
            all_files = [f for f in ds.rglob("*") if f.is_file()]
            file_count = len(all_files)
            size_mb = sum(f.stat().st_size for f in all_files) / (1024 * 1024)
            col1, col2, col3 = st.columns([4, 2, 1])
            with col1:
                st.text(ds.name)
            with col2:
                st.text(f"{file_count} files  /  {size_mb:.1f} MB")
            with col3:
                if st.button("🗑", key=f"del_ds_{ds.name}", help=f"{ds.name} を削除"):
                    shutil.rmtree(ds)
                    st.success(f"{ds.name} を削除しました")
                    st.rerun()
            _ds_prov = read_provenance(ds)
            if _ds_prov:
                _src_label = {
                    "cvat": "CVAT から生成", "upload_zip": "ZIP を取込",
                    "upload_images": "画像を直接アップロード", "merge": "データセット統合",
                }.get(_ds_prov.get("source", ""), _ds_prov.get("source", "不明"))
                _tasks_txt = ", ".join(
                    f"[{t.get('id')}] {t.get('name')}" for t in (_ds_prov.get("cvat_tasks") or [])
                )
                st.caption(
                    f"📚 {_src_label}"
                    + (f"（{_ds_prov.get('task_type')}）" if _ds_prov.get("task_type") else "")
                    + f"　作成: {_ds_prov.get('created_at', '不明')}"
                    + (f"　元タスク: {_tasks_txt}" if _tasks_txt else "")
                )
            else:
                st.caption("📚 来歴の記録なし（この機能を入れる前に作られたデータセットです）")

            with st.expander(f"✂️ {ds.name} の train/val を分け直す"):
                st.caption(
                    "生成時に決めた比率のままだと「val が偏っていて評価が信用できない」"
                    "ときに手が出せません。ここで混ぜ直せます。"
                    "画像とラベルは対のまま移動し、何度でもやり直せます。"
                )
                _rs_before = dataset_split_counts(ds)
                st.caption("現在: " + (" / ".join(f"{k} {v}枚" for k, v in _rs_before.items())
                                       or "（画像なし）"))
                _rsc1, _rsc2 = st.columns(2)
                with _rsc1:
                    _rs_ratio = st.slider("val の割合", 0.05, 0.50, 0.20, 0.05,
                                          key=f"rs_ratio_{ds.name}")
                with _rsc2:
                    _rs_seed = st.number_input(
                        "乱数シード", 0, 9999, 0, key=f"rs_seed_{ds.name}",
                        help="同じ値なら同じ分け方になります。変えると別の組み合わせを試せます")
                if st.button("✂️ 分け直す", key=f"rs_run_{ds.name}",
                             use_container_width=True):
                    with st.spinner("分割し直しています…"):
                        _rs = resplit_dataset(ds, val_ratio=float(_rs_ratio),
                                              seed=int(_rs_seed))
                    if _rs["error"]:
                        show_error(_rs["error"], prefix="❌ 分割に失敗しました: ")
                    else:
                        st.success(
                            "✅ 分割し直しました　"
                            + " / ".join(f"{k} {v}枚" for k, v in _rs["after"].items())
                            + f"（{_rs['moved']} 件を移動）"
                        )
                        st.rerun()

            with st.expander(f"🏷 {ds.name} のクラス名を編集・統合する"):
                _cls_names = dataset_class_names(ds)
                if not _cls_names:
                    st.info("data.yaml にクラス定義がありません。")
                else:
                    st.caption(
                        "クラス名の変更・統合・削除ができます。"
                        "複数のクラスに同じ新しい名前を付けると統合されます。"
                        "空欄にするとそのクラスのアノテーションを削除します。"
                        "ラベルは `.txt.bak` にバックアップしてから書き換えます。"
                    )
                    _mapping: dict = {}
                    for _cn in _cls_names:
                        _mapping[_cn] = st.text_input(
                            f"`{_cn}` →", value=_cn, key=f"cls_map_{ds.name}_{_cn}",
                        ).strip()

                    _new_list: list[str] = []
                    for _cn in _cls_names:
                        _nv = _mapping[_cn]
                        if _nv and _nv not in _new_list:
                            _new_list.append(_nv)
                    _removed = [c for c in _cls_names if not _mapping[c]]

                    if _new_list != _cls_names or _removed:
                        st.markdown(f"変更後のクラス: **{', '.join(_new_list) or '（なし）'}**")
                        if _removed:
                            st.warning(f"⚠ 削除されるクラス: {', '.join(_removed)}"
                                       "（該当するアノテーションが消えます）")
                        if not _new_list:
                            st.error("すべてのクラスが削除対象です。1つ以上残してください。")
                        elif st.button("🏷 クラスを更新する", key=f"cls_run_{ds.name}",
                                       type="primary", use_container_width=True):
                            with st.spinner("ラベルを書き換えています…"):
                                _rm = remap_dataset_classes(
                                    ds, {k: (v or None) for k, v in _mapping.items()})
                            if _rm["error"]:
                                show_error(_rm["error"], prefix="❌ 更新に失敗しました: ")
                            else:
                                st.success(
                                    f"✅ {', '.join(_rm['old_classes'])} → "
                                    f"{', '.join(_rm['new_classes'])}"
                                    + (f"（{_rm['files_changed']} ファイルを書換、"
                                       f"{_rm['lines_removed']} 行を除去）"
                                       if _rm["files_changed"] else "")
                                    + (f"（{_rm['dirs_merged']} ディレクトリを整理）"
                                       if _rm["dirs_merged"] else "")
                                )
                                st.rerun()
                    else:
                        st.caption("変更はありません。")

            with st.expander(f"⬇ {ds.name} を持ち出す（ZIP エクスポート）"):
                st.caption(
                    "他の PC で学習させる場合などに、データセットを ZIP で書き出します。"
                    "展開すればそのまま YOLO の学習に使える構造のままです。"
                )
                _ex_labels_only = st.checkbox(
                    "ラベルと data.yaml のみ（画像を含めない）", value=False,
                    key=f"ex_lbl_{ds.name}",
                    help="画像は既に相手側にある場合や、アノテーションだけ共有したい場合に使います",
                )
                _ex_bytes = dataset_size_bytes(ds, labels_only=_ex_labels_only)
                _ex_mb = _ex_bytes / 1024 / 1024
                st.caption(f"対象サイズ: 約 {_ex_mb:,.1f} MB（圧縮前）")
                if _ex_mb > 500:
                    st.warning(
                        f"⚠ {_ex_mb:,.0f} MB あります。ZIP の生成とダウンロードに時間がかかり、"
                        "ブラウザ側のメモリも消費します。"
                        "画像が不要なら「ラベルと data.yaml のみ」を使ってください。"
                    )

                if st.button("📦 ZIP を生成", key=f"ex_build_{ds.name}",
                             use_container_width=True):
                    _ex_out = (PREDICTIONS_DIR / "_exports" /
                               f"{ds.name}{'_labels' if _ex_labels_only else ''}.zip")
                    with st.spinner("ZIP を生成中…（サイズによっては数分かかります）"):
                        _ok_ex, _msg_ex, _n_ex = build_dataset_zip(
                            ds, _ex_out, labels_only=_ex_labels_only)
                    st.session_state[f"ex_zip_{ds.name}"] = (
                        {"path": _msg_ex, "n": _n_ex} if _ok_ex else None)
                    if not _ok_ex:
                        st.error(f"❌ 生成に失敗しました: {_msg_ex}")

                _ex_info = st.session_state.get(f"ex_zip_{ds.name}")
                if _ex_info and Path(_ex_info["path"]).exists():
                    _ex_p = Path(_ex_info["path"])
                    st.success(f"✅ {_ex_info['n']} ファイル / "
                               f"{_ex_p.stat().st_size / 1024 / 1024:,.1f} MB")
                    with open(_ex_p, "rb") as _fz:
                        st.download_button(
                            "⬇ ダウンロード", _fz, file_name=_ex_p.name,
                            mime="application/zip", use_container_width=True,
                            key=f"ex_dl_{ds.name}",
                        )
                    st.caption(f"生成先: `{_ex_p}`（不要になったら削除して構いません）")

            with st.expander(f"🔍 {ds.name} の品質チェック"):
                st.caption(
                    "画像とラベルの対応漏れ・座標の破損・クラス分布の偏りを検査します。"
                    "外部から持ち込んだデータや複数人で分担したデータほど確認する価値があります。"
                )
                if st.button("🔍 チェックを実行", key=f"qc_run_{ds.name}",
                             use_container_width=True):
                    with st.spinner(f"{ds.name} を検査中…"):
                        st.session_state[f"qc_{ds.name}"] = check_dataset_quality(ds)

                _qc = st.session_state.get(f"qc_{ds.name}")
                if _qc:
                    if _qc["error"]:
                        st.error(f"❌ {_qc['error']}")
                    else:
                        _n_err = _qc.get("n_errors", 0)
                        if _qc["n_issues"] == 0:
                            st.success("✅ 問題は見つかりませんでした。")
                        elif _n_err > 0:
                            st.error(f"❌ 要対応 {_n_err} 件 / 指摘 {_qc['n_issues']} 件")
                        else:
                            st.warning(f"⚠ 指摘 {_qc['n_issues']} 件（いずれも警告レベル）")

                        # スプリット別の内訳
                        import pandas as _pd_qc
                        _rows_qc = [{
                            "スプリット": sp,
                            "画像": v["images"], "ラベル": v["labels"],
                            "ラベル無し画像": v["missing_label"],
                            "画像無しラベル": v["orphan_label"],
                            "空ラベル": v["empty_label"],
                            "ボックス数": v["boxes"],
                        } for sp, v in _qc["splits"].items()]
                        if _rows_qc:
                            st.dataframe(_pd_qc.DataFrame(_rows_qc),
                                         use_container_width=True, hide_index=True)

                        # クラス分布
                        if _qc["class_counts"]:
                            st.markdown("**クラス分布**")
                            _df_cls = _pd_qc.DataFrame(
                                sorted(_qc["class_counts"].items(),
                                       key=lambda kv: -kv[1]),
                                columns=["クラス", "件数"],
                            )
                            st.dataframe(_df_cls, use_container_width=True, hide_index=True)

                        # 指摘の内訳と詳細
                        if _qc["issue_counts"]:
                            st.markdown("**指摘の内訳**")
                            for _k, _v in sorted(_qc["issue_counts"].items(),
                                                 key=lambda kv: -kv[1]["count"]):
                                _icon = "❌" if _v["severity"] == "error" else "⚠"
                                st.markdown(f"- {_icon} **{_k}**: {_v['count']} 件")
                            # expander の入れ子は不可のためチェックボックスで開閉する
                            if st.checkbox("詳細を表示（種別ごとに最大20件）",
                                           key=f"qc_detail_{ds.name}"):
                                with st.container(border=True):
                                    for _is in _qc["issues"]:
                                        _icon = "❌" if _is["severity"] == "error" else "⚠"
                                        st.caption(f"{_icon} `{_is['path']}` — "
                                                   f"{_is['kind']}: {_is['detail']}")

                            # --- 自動修正 ---
                            _fixable = {"サイズ不正", "座標範囲外", "行フォーマット",
                                        "数値変換", "極小ボックス", "画像無しラベル"}
                            if _fixable & set(_qc["issue_counts"].keys()):
                                st.markdown("**🔧 壊れたラベルの自動修正**")
                                st.caption(
                                    "該当する行だけを取り除きます。書き換える前に "
                                    "`<ファイル名>.txt.bak` としてバックアップを作るので元に戻せます。"
                                )
                                _fx1, _fx2 = st.columns(2)
                                with _fx1:
                                    _fx_size = st.checkbox(
                                        "幅・高さが0以下の行を除去", value=True,
                                        key=f"fx_size_{ds.name}")
                                    _fx_range = st.checkbox(
                                        "座標が0〜1の範囲外の行を除去", value=True,
                                        key=f"fx_range_{ds.name}")
                                with _fx2:
                                    _fx_tiny = st.checkbox(
                                        "極小ボックスも除去", value=False,
                                        key=f"fx_tiny_{ds.name}",
                                        help="小さな物体を意図的にアノテーションしている場合は"
                                             "OFF のままにしてください")
                                    _fx_orphan = st.checkbox(
                                        "画像が無いラベルを退避", value=False,
                                        key=f"fx_orphan_{ds.name}")

                                if st.button("🔧 修正を実行", key=f"fx_run_{ds.name}",
                                             type="primary", use_container_width=True):
                                    with st.spinner("修正中…"):
                                        _fx = fix_dataset_labels(
                                            ds,
                                            drop_invalid_size=_fx_size,
                                            drop_out_of_range=_fx_range,
                                            drop_tiny=_fx_tiny,
                                            delete_orphan_labels=_fx_orphan,
                                        )
                                    if _fx["error"]:
                                        st.error(f"❌ {_fx['error']}")
                                    else:
                                        st.success(
                                            f"✅ {_fx['files_changed']} ファイルを修正し "
                                            f"{_fx['lines_removed']} 行を除去しました"
                                            + (f"／ {_fx['orphans_deleted']} 件の迷子ラベルを退避"
                                               if _fx["orphans_deleted"] else "")
                                        )
                                        if _fx["files_emptied"]:
                                            st.warning(
                                                f"⚠ {_fx['files_emptied']} ファイルが空になりました"
                                                "（その画像は背景画像として扱われます）")
                                        for _d in _fx["details"][:20]:
                                            st.caption(f"・{_d}")
                                        # 修正後の状態で再チェック
                                        st.session_state[f"qc_{ds.name}"] = \
                                            check_dataset_quality(ds)
                                        st.rerun()

            with st.expander(f"➕ {ds.name} に画像を追加"):
                _add_imgs = st.file_uploader(
                    "追加する画像ファイル（複数選択可）",
                    type=["jpg", "jpeg", "png", "bmp", "tiff"],
                    accept_multiple_files=True,
                    key=f"add_imgs_{ds.name}",
                )
                _add_split = st.radio(
                    "追加先スプリット", ["train", "val"], horizontal=True,
                    key=f"add_split_{ds.name}",
                )
                if _add_imgs:
                    st.caption(f"選択中: {len(_add_imgs)} ファイル")
                    _add_dst = ds / "images" / _add_split
                    if st.button(
                        f"📤 images/{_add_split}/ に追加",
                        key=f"add_btn_{ds.name}",
                        type="primary",
                        use_container_width=True,
                    ):
                        _add_dst.mkdir(parents=True, exist_ok=True)
                        for _f in _add_imgs:
                            (_add_dst / _f.name).write_bytes(_f.getbuffer())
                        st.success(f"✅ {len(_add_imgs)} ファイルを追加しました → `{_add_dst}`")
                        st.rerun()

    st.markdown("---")

    # --- models/ モデル一覧（カード表示） ---
    st.markdown("#### 🤖 学習済みモデル (`models/`)")

    # --- 外部モデルの取り込み ---
    with st.expander("📤 学習済みモデル (.pt) をアップロード（他PCで学習したモデルの取り込み）"):
        import io as _io_mu

        st.caption(
            "他の環境で学習した YOLO の重みを `models/` に取り込みます。"
            "取り込み時にこの環境の ultralytics で読み込めるか検証し、クラス名を記録します。"
        )
        _mu_run = st.text_input(
            "モデル名（`models/` 以下に作成するディレクトリ名）",
            value=f"imported_{datetime.now():%Y%m%d_%H%M}",
            key="mu_run_name",
        ).strip()
        _mu_mode = st.radio(
            "アップロード形式",
            ["📦 .pt ファイル", "🗜 学習run ディレクトリ ZIP"],
            horizontal=True,
            key="mu_mode",
        )
        _mu_set_current = st.checkbox(
            "取り込み後、このモデルを「使用中」にする", value=True, key="mu_set_current"
        )

        _mu_dir = MODELS_DIR / _mu_run if _mu_run else None
        if _mu_dir and _mu_dir.exists():
            st.warning(f"⚠ `models/{_mu_run}/` は既に存在します。同名ファイルは上書きされます。")

        if _mu_mode == "📦 .pt ファイル":
            _mu_pts = st.file_uploader(
                "重みファイル（.pt、複数選択可）",
                type=["pt"],
                accept_multiple_files=True,
                key="mu_pt_files",
            )
            _mu_extras = st.file_uploader(
                "付随ファイル（任意・複数可）",
                type=["csv", "yaml", "yml", "json", "txt", "png", "jpg", "jpeg"],
                accept_multiple_files=True,
                key="mu_extra_files",
                help="results.csv を一緒に入れると下のカードに mAP50 が表示されます。"
                     "args.yaml / confusion_matrix.png なども保存できます。",
            )
            if _mu_pts and _mu_run:
                st.caption(
                    f"選択中: {len(_mu_pts)} 個の重み "
                    f"({sum(f.size for f in _mu_pts) / 1024 / 1024:.1f} MB)"
                    + (f" + 付随 {len(_mu_extras)} ファイル" if _mu_extras else "")
                )
                st.caption(f"保存先: `models/{_mu_run}/weights/`")
                if st.button("📥 models/ に取り込む", key="mu_pt_btn",
                             type="primary", use_container_width=True):
                    _mu_w = _mu_dir / "weights"
                    _mu_w.mkdir(parents=True, exist_ok=True)
                    _mu_saved = []
                    for _f in _mu_pts:
                        _dst = _mu_w / _f.name
                        _dst.write_bytes(_f.getbuffer())
                        _mu_saved.append(_dst)
                    for _f in (_mu_extras or []):
                        (_mu_dir / _f.name).write_bytes(_f.getbuffer())
                    st.session_state["mu_saved_paths"] = [str(p) for p in _mu_saved]
                    st.session_state["mu_pending_current"] = _mu_set_current
        else:
            _mu_zip = st.file_uploader(
                "学習run ディレクトリの ZIP（`weights/best.pt` を含む想定）",
                type=["zip"],
                key="mu_zip_file",
            )
            if _mu_zip and _mu_run:
                st.caption(f"選択中: {_mu_zip.name}  ({_mu_zip.size / 1024 / 1024:.1f} MB)")
                st.caption(f"展開先: `models/{_mu_run}/`")
                if st.button("📥 展開して models/ に取り込む", key="mu_zip_btn",
                             type="primary", use_container_width=True):
                    with zipfile.ZipFile(_io_mu.BytesIO(_mu_zip.read()), "r") as _zf:
                        _bad = [n for n in _zf.namelist()
                                if n.startswith("/") or ".." in Path(n).parts]
                        if _bad:
                            st.error(f"⚠ ZIP に不正なパスが含まれています: {_bad[:3]}")
                        else:
                            _mu_dir.mkdir(parents=True, exist_ok=True)
                            _zf.extractall(_mu_dir)
                            _mu_found = sorted(_mu_dir.rglob("*.pt"))
                            if not _mu_found:
                                st.error("⚠ ZIP 内に .pt ファイルが見つかりませんでした。")
                            st.session_state["mu_saved_paths"] = [str(p) for p in _mu_found]
                            st.session_state["mu_pending_current"] = _mu_set_current

        # --- 取り込み結果の検証・表示 ---
        _mu_saved_paths = [Path(p) for p in st.session_state.get("mu_saved_paths", [])]
        if _mu_saved_paths:
            st.markdown("---")
            st.markdown("**取り込み結果**")
            _mu_ok_paths = []
            for _p in _mu_saved_paths:
                if not _p.exists():
                    continue
                with st.spinner(f"{_p.name} を検証中…"):
                    _info = inspect_model_file(_p)
                if _info["ok"]:
                    _mu_ok_paths.append(_p)
                    st.success(f"✅ `{_p.relative_to(MODELS_DIR)}` — 読み込み成功")
                    _mi1, _mi2, _mi3 = st.columns(3)
                    _mi1.metric("クラス数", len(_info["names"]))
                    _mi2.metric("タスク", _info["task"] or "—")
                    _mi3.metric("学習時 ultralytics", _info["ultralytics_version"] or "—")
                    st.caption("クラス: " + (", ".join(_info["names"]) or "—"))
                    _mu_detail = [
                        f"ベースモデル: {_info['base_model']}" if _info.get("base_model") else "",
                        f"imgsz: {_info['imgsz']}" if _info.get("imgsz") else "",
                        f"epochs: {_info['epochs']}" if _info.get("epochs") else "",
                        f"学習日: {_info['trained_at']}" if _info.get("trained_at") else "",
                    ]
                    _mu_detail = [d for d in _mu_detail if d]
                    if _mu_detail:
                        st.caption(" / ".join(_mu_detail))
                else:
                    show_error(
                        _info["error"],
                        prefix=f"❌ `{_p.relative_to(MODELS_DIR)}` は読み込めませんでした: ",
                    )
            if _mu_ok_paths and st.session_state.get("mu_pending_current"):
                _mu_best = next((p for p in _mu_ok_paths if p.name == "best.pt"), _mu_ok_paths[0])
                st.session_state.last_model_path = str(_mu_best)
                st.session_state["mu_pending_current"] = False
                st.info(f"⭐ 使用中モデルに設定しました: `{_mu_best.relative_to(MODELS_DIR)}`\n\n"
                        "→ 「🔭 Step4: 推論・評価」タブで推論を実行できます。")
            if st.button("表示をクリア", key="mu_clear_result"):
                st.session_state["mu_saved_paths"] = []
                st.rerun()

    model_files = sorted(
        MODELS_DIR.rglob("*.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    ) if MODELS_DIR.exists() else []
    if not model_files:
        st.info("models/ に .pt ファイルがありません。")
    else:
        import pandas as pd

        # どのモデルが CVAT の自動アノテーションに載っているかを引くための対応表
        # （models/<run> ←→ serverless/custom/<dir> ←→ Nuclio 関数）
        _fn_states = {f["name"]: f for f in cached_nuclio_functions()} if serverless_ready() else {}
        _def_by_run = {d["model_run"]: d for d in list_serverless_defs() if d["model_run"]}

        for mp in model_files:
            size_mb  = mp.stat().st_size / (1024 * 1024)
            mod_time = datetime.fromtimestamp(mp.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            is_current = (str(mp) == st.session_state.last_model_path)

            # results.csv からmAP50を取得
            _results_csv = mp.parent.parent / "results.csv"
            _map50_val = None
            if _results_csv.exists():
                try:
                    _df_r = pd.read_csv(_results_csv)
                    _df_r.columns = [c.strip() for c in _df_r.columns]
                    _mc = next((c for c in _df_r.columns
                                if "map50" in c.lower() and "95" not in c.lower()), None)
                    if _mc:
                        _map50_val = float(_df_r.iloc[-1][_mc])
                except Exception:
                    pass

            with st.container(border=True):
                _label_col, _size_col, _map_col, _use_col, _del_col = st.columns([4, 2, 2, 2, 1])
                with _label_col:
                    if is_current:
                        st.markdown("⭐ **現在使用中**")
                    st.markdown(f"`{mp.relative_to(MODELS_DIR)}`")
                    st.caption(mod_time)
                    _meta = read_model_meta(mp)
                    if _meta and _meta.get("ok"):
                        _cls = _meta.get("names") or []
                        _cls_txt = ", ".join(_cls[:6])
                        if len(_cls) > 6:
                            _cls_txt += f" 他{len(_cls) - 6}件"
                        st.caption(f"🏷 {len(_cls)} クラス: {_cls_txt}")
                    elif _meta:
                        st.caption("⚠ この環境の ultralytics で読み込めないモデルです")
                    else:
                        if st.button("🔍 モデル情報を読み込む", key=f"insp_model_{mp}"):
                            with st.spinner(f"{mp.name} を読み込み中…"):
                                inspect_model_file(mp)
                            st.rerun()

                    # 来歴（何で学習したモデルか）
                    _prov = read_provenance(mp.parent.parent) if mp.parent.name == "weights" else None
                    if _prov:
                        _pds = _prov.get("dataset", {}) or {}
                        _pc = _pds.get("counts_at_train", {}) or {}
                        _cnt_txt = " / ".join(f"{k} {v}枚" for k, v in _pc.items())
                        st.caption(
                            f"📚 学習データ: `{_pds.get('name') or '不明'}`"
                            + (f"（{_cnt_txt}）" if _cnt_txt else "")
                            + f"　ベース: `{Path(_prov.get('base_model', '')).name or '不明'}`"
                            + ("　※再開あり" if _prov.get("resumed") else "")
                        )

                    # 評価済みなら最新の mAP を出す（results.csv とは別に、
                    # 任意データセットで測り直した値）
                    _evs = read_model_evals(mp)
                    if _evs:
                        _latest_key = max(_evs, key=lambda k: _evs[k].get("evaluated_at", ""))
                        _lv = _evs[_latest_key]
                        if _lv.get("ok"):
                            st.caption(
                                f"📊 評価 `{_latest_key}` — mAP50 {_lv['map50']:.4f} / "
                                f"mAP50-95 {_lv['map50_95']:.4f}"
                                + (f"（他 {len(_evs) - 1} 件）" if len(_evs) > 1 else "")
                            )

                    # CVAT 自動アノテーションへのデプロイ状態
                    _run_name = (mp.parent.parent.name
                                 if mp.parent.name == "weights" else "")
                    _def = _def_by_run.get(_run_name)
                    _fn_st = _fn_states.get(_def["function_name"]) if _def else None
                    if _fn_st:
                        _bdg = {"ready": "🟢", "error": "🔴"}.get(_fn_st["state"], "🟡")
                        st.caption(f"{_bdg} CVAT自動アノテーションに使用中 "
                                   f"(`{_fn_st['name']}` / {'GPU' if _fn_st['gpu'] else 'CPU'})")
                    elif mp.name == "best.pt":
                        st.caption("○ 自動アノテーション未デプロイ — 「🏷 アノテーション」タブから追加できます")
                with _size_col:
                    st.metric("サイズ", f"{size_mb:.1f} MB")
                with _map_col:
                    if _map50_val is not None:
                        st.metric("mAP50", f"{_map50_val:.4f}")
                    else:
                        st.caption("mAP50: -")
                with _use_col:
                    if st.button("✅ 使用", key=f"use_model_{mp}", use_container_width=True,
                                 type="primary" if not is_current else "secondary"):
                        st.session_state.last_model_path = str(mp)
                        st.rerun()
                with _del_col:
                    if st.button("🗑", key=f"del_model_{mp}", help=f"{mp.name} を削除"):
                        mp.unlink()
                        model_meta_path(mp).unlink(missing_ok=True)
                        model_eval_path(mp).unlink(missing_ok=True)
                        if st.session_state.last_model_path == str(mp):
                            st.session_state.last_model_path = None
                        st.rerun()

                # --- 持ち出し（他PCへ渡す）---
                _dl1, _dl2 = st.columns(2)
                with _dl1:
                    with open(mp, "rb") as _fm:
                        st.download_button(
                            f"⬇ {mp.name} をダウンロード", _fm, file_name=mp.name,
                            mime="application/octet-stream", use_container_width=True,
                            key=f"dl_pt_{mp}",
                            help="重みファイル単体。相手側の UI でそのまま取り込めます",
                        )
                with _dl2:
                    _bundle_key = f"bundle_{mp}"
                    if st.button("📦 一式ZIPを生成", key=f"mkbundle_{mp}",
                                 use_container_width=True,
                                 help="重み + results.csv + 評価結果 + プロットをまとめます"):
                        _b_out = (PREDICTIONS_DIR / "_exports" /
                                  f"{mp.parent.parent.name}_bundle.zip")
                        with st.spinner("ZIP を生成中…"):
                            _ok_b, _msg_b, _n_b = build_model_bundle_zip(mp, _b_out)
                        st.session_state[_bundle_key] = (
                            {"path": _msg_b, "n": _n_b} if _ok_b else None)
                        if not _ok_b:
                            st.error(f"❌ {_msg_b}")
                        st.rerun()
                    _b_info = st.session_state.get(_bundle_key)
                    if _b_info and Path(_b_info["path"]).exists():
                        _b_p = Path(_b_info["path"])
                        with open(_b_p, "rb") as _fb:
                            st.download_button(
                                f"⬇ 一式ZIP ({_b_p.stat().st_size / 1024 / 1024:.0f}MB)",
                                _fb, file_name=_b_p.name, mime="application/zip",
                                use_container_width=True, key=f"dl_bundle_{mp}",
                            )

    st.markdown("---")

    # --- モデルの系譜（何から何が作られたか）---
    with st.expander("📚 モデルの系譜を追跡する"):
        st.caption(
            "モデルがどのデータセットから作られたかを一覧します。"
            "データを足しながら学習を重ねると対応が分からなくなるため、"
            "学習開始時点の情報を記録しています。"
        )
        _lin_rows = []
        for _run in sorted([p for p in MODELS_DIR.iterdir() if p.is_dir()],
                           key=lambda p: p.stat().st_mtime, reverse=True) \
                if MODELS_DIR.exists() else []:
            _pv = read_provenance(_run)
            if not _pv:
                _lin_rows.append({
                    "モデル": _run.name, "学習日時": "—", "学習データ": "（記録なし）",
                    "枚数": "—", "ベースモデル": "—", "クラス": "—",
                })
                continue
            _d = _pv.get("dataset", {}) or {}
            _cnts = _d.get("counts_at_train", {}) or {}
            _lin_rows.append({
                "モデル": _run.name,
                "学習日時": _pv.get("trained_at", "—"),
                "学習データ": _d.get("name") or "—",
                "枚数": " / ".join(f"{k}:{v}" for k, v in _cnts.items()) or "—",
                "ベースモデル": Path(_pv.get("base_model", "")).name or "—",
                "クラス": ", ".join(_d.get("classes") or []) or "—",
            })

        if _lin_rows:
            import pandas as _pd_lin
            st.dataframe(_pd_lin.DataFrame(_lin_rows),
                         use_container_width=True, hide_index=True)

            # 1件を選んで詳細（データセット側の来歴まで辿る）
            _lin_names = [r["モデル"] for r in _lin_rows]
            _lin_sel = st.selectbox("詳細を見るモデル", _lin_names, key="lineage_sel")
            _lin_pv = read_provenance(MODELS_DIR / _lin_sel)
            if not _lin_pv:
                st.info("このモデルには来歴の記録がありません。"
                        "この機能を入れる前に学習されたモデルです。")
            else:
                _ld = _lin_pv.get("dataset", {}) or {}
                _dsp = _ld.get("provenance") or {}
                st.markdown("**系譜**")
                _chain = []
                if _dsp.get("cvat_tasks"):
                    _chain.append("CVAT タスク " + ", ".join(
                        f"[{t.get('id')}] {t.get('name')}" for t in _dsp["cvat_tasks"]))
                elif _dsp.get("source"):
                    _chain.append({"upload_zip": "外部 ZIP の取込",
                                   "upload_images": "画像の直接アップロード",
                                   "merge": "データセット統合"}.get(_dsp["source"], _dsp["source"]))
                if _ld.get("name"):
                    _chain.append(f"データセット `{_ld['name']}`")
                _chain.append(f"モデル `{_lin_sel}`")
                st.markdown("　→　".join(_chain))

                _lc1, _lc2 = st.columns(2)
                with _lc1:
                    st.markdown("**学習時の情報**")
                    st.caption(f"学習日時: {_lin_pv.get('trained_at', '—')}")
                    st.caption(f"ベースモデル: `{_lin_pv.get('base_model', '—')}`")
                    st.caption(f"再開: {'あり' if _lin_pv.get('resumed') else 'なし'}")
                    _pp = _lin_pv.get("params", {}) or {}
                    st.caption("主なパラメータ: " + ", ".join(
                        f"{k}={_pp[k]}" for k in ("epochs", "batch", "imgsz", "optimizer")
                        if k in _pp) or "—")
                with _lc2:
                    st.markdown("**学習に使ったデータ**")
                    st.caption(f"データセット: `{_ld.get('name') or '—'}`")
                    st.caption(f"クラス: {', '.join(_ld.get('classes') or []) or '—'}")
                    _cnts2 = _ld.get("counts_at_train", {}) or {}
                    st.caption("学習時の枚数: " + (
                        " / ".join(f"{k} {v}枚" for k, v in _cnts2.items()) or "—"))
                    # 現在のデータセットと比べて増減があれば知らせる
                    _cur_ds = Path(_ld.get("data_yaml", "")).parent if _ld.get("data_yaml") else None
                    if _cur_ds and _cur_ds.exists():
                        _now = count_dataset_items(_cur_ds)
                        if _now != _cnts2 and _cnts2:
                            st.warning(
                                "⚠ 学習後にデータセットが変わっています（現在: "
                                + " / ".join(f"{k} {v}枚" for k, v in _now.items())
                                + "）。再学習すると結果が変わります。"
                            )
                    elif _ld.get("data_yaml"):
                        st.warning("⚠ 学習に使ったデータセットは現在見つかりません。")

                # expander の入れ子は不可のためチェックボックスで開閉する
                if st.checkbox("生の来歴データ (JSON) を表示", key="lineage_raw"):
                    st.json(_lin_pv)
        else:
            st.info("models/ に学習 run がありません。")

    st.markdown("---")

    # --- predictions/ 一括クリア ---
    st.markdown("#### 推論結果 (`predictions/`)")
    pred_files = list(PREDICTIONS_DIR.glob("*.json")) if PREDICTIONS_DIR.exists() else []
    if not pred_files:
        st.info("predictions/ に結果 JSON がありません。")
    else:
        st.text(f"{len(pred_files)} 件の結果ファイル")
        if st.button("🗑 predictions/ をすべてクリア", type="secondary",
                     key="pred_clear_all"):
            for jf in pred_files:
                jf.unlink()
            st.success("predictions/ をクリアしました")
            st.rerun()

    st.markdown("---")
    st.markdown("#### 🔀 データセット統合")
    _ds_dirs = [d for d in sorted(DATA_DIR.iterdir()) if d.is_dir()] if DATA_DIR.exists() else []
    _ds_names = [d.name for d in _ds_dirs]
    if len(_ds_names) < 2:
        st.info("統合するには2つ以上のデータセットが必要です。")
    else:
        merge_targets = st.multiselect("統合するデータセットを選択（2つ以上）", _ds_names)
        merge_out_name = st.text_input("統合先ディレクトリ名", value=f"merged_{datetime.now():%Y%m%d_%H%M}")
        if st.button("🔀 統合実行", disabled=len(merge_targets) < 2,
                     key="merge_datasets_run"):
            import yaml as pyyaml
            out_dir = DATA_DIR / merge_out_name
            all_labels: list[str] = []
            # 各データセットからラベル収集
            for ds_name in merge_targets:
                src = DATA_DIR / ds_name
                yaml_f = src / "data.yaml"
                if yaml_f.exists():
                    with open(yaml_f) as f:
                        ydata = pyyaml.safe_load(f)
                    for lbl in ydata.get("names", []):
                        if lbl not in all_labels:
                            all_labels.append(lbl)
            # 画像・ラベルをコピー
            for split in ("train", "val"):
                for ds_name in merge_targets:
                    src = DATA_DIR / ds_name
                    for kind in ("images", "labels"):
                        src_dir = src / split / kind
                        if not src_dir.exists():
                            continue
                        dst_dir = out_dir / split / kind
                        dst_dir.mkdir(parents=True, exist_ok=True)
                        for f in src_dir.iterdir():
                            dst = dst_dir / f"{ds_name}_{f.name}"
                            shutil.copy2(f, dst)
            # data.yaml 生成
            data_yaml_content = {
                "path": str(out_dir),
                "train": "train/images",
                "val": "val/images",
                "names": all_labels,
                "nc": len(all_labels),
            }
            with open(out_dir / "data.yaml", "w") as f:
                pyyaml.dump(data_yaml_content, f, allow_unicode=True)
            st.success(f"✅ 統合完了: `{out_dir}` (ラベル: {all_labels})")
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)
