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
from .widgets import empty_state, show_error




def render_manage() -> None:
    import shutil

    st.markdown('<div class="section-head"><h3>📁 データ管理</h3></div>', unsafe_allow_html=True)

    # --- data/ データセット一覧 ---
    st.markdown("#### 学習データセット (`data/`)")
    datasets = sorted(DATA_DIR.iterdir()) if DATA_DIR.exists() else []
    datasets = [d for d in datasets if d.is_dir()]

    _ds_filtered = False        # 絞り込みで 0 件になったのか、元から 0 件なのか

    # --- 絞り込み（件数が増えたときに効く）---
    if len(datasets) > 1:
        _f1, _f2 = st.columns([3, 3])
        with _f1:
            _f_status = st.multiselect(
                "状態で絞り込む", list(DATASET_STATUSES),
                format_func=lambda v: status_label(v),
                key="ds_filter_status",
                placeholder="すべて",
            )
        with _f2:
            _f_tags = st.multiselect(
                "タグで絞り込む", collect_tags(datasets),
                key="ds_filter_tags", placeholder="すべて",
            )
        _all_n = len(datasets)
        if _f_status:
            datasets = [d for d in datasets if read_status(d) in _f_status]
        if _f_tags:
            # 選んだタグを1つでも持つもの
            datasets = [d for d in datasets if set(read_tags(d)) & set(_f_tags)]
        if _f_status or _f_tags:
            _ds_filtered = True
            st.caption(f"{len(datasets)} / {_all_n} 件を表示中")

    if _ds_filtered and not datasets:
        st.info("条件に合うデータセットがありません。絞り込みを外してください。")
    elif not datasets:
        empty_state(
            "まだデータセットがありません",
            "「📤 Step2: データ取込」で CVAT のタスクを取り込むか、"
            "同タブの「📁 ローカルからデータを直接追加」で手元の画像・ZIP を入れてください。",
        )
    else:
        for ds in datasets:
            # モデル一覧と同じくカードで囲む。
            # 以前はデータセット 1 件につき expander が 5 個並んでいて、
            # 件数が増えると縦に伸びるだけで目的の操作を探せなかった。
            # 操作はラジオで 1 つだけ開く方式にする。
            with st.container(border=True):
                all_files = [f for f in ds.rglob("*") if f.is_file()]
                file_count = len(all_files)
                size_mb = sum(f.stat().st_size for f in all_files) / (1024 * 1024)
                col1, col2, col3 = st.columns([4, 2, 1])
                with col1:
                    st.markdown(f"**`{ds.name}`**　{status_label(read_status(ds))}")
                    _split_counts = dataset_split_counts(ds)
                    if _split_counts:
                        st.caption(" / ".join(f"{k} {v}枚" for k, v in _split_counts.items()))
                    _ds_tags = read_tags(ds)
                    if _ds_tags:
                        st.markdown(
                            " ".join(f'<span class="tc-chip">{t}</span>' for t in _ds_tags),
                            unsafe_allow_html=True,
                        )
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

                # 統合で作られたものは、何を混ぜたのかを出す
                for _s in ((_ds_prov or {}).get("sources") or []):
                    st.caption(
                        f"　└ 🔀 `{_s.get('name', '不明')}`"
                        f"（{status_label(_s.get('status', ''))}"
                        + (f"　{' / '.join(f'{k} {v}枚' for k, v in (_s.get('counts') or {}).items())}"
                           if _s.get("counts") else "")
                        + "）統合した時点の記録"
                    )

                _DS_OP_NONE = "—"
                _ds_op = st.radio(
                    "操作",
                    [_DS_OP_NONE, "🏷 状態とタグ", "✂️ train/val を分け直す",
                     "🏷 クラス名を編集", "⬇ 持ち出す (ZIP)", "🔍 品質チェック",
                     "➕ 画像を追加"],
                    horizontal=True, key=f"ds_op_{ds.name}",
                    label_visibility="collapsed",
                )

                if _ds_op == "🏷 状態とタグ":
                    st.caption(
                        "状態は「いまどの段階か」を1つだけ選びます。"
                        "タグは性格づけ（撮影場所・被写体・用途など）を自由に付けられます。"
                    )
                    _st_keys = list(DATASET_STATUSES)
                    _cur_st = read_status(ds)
                    _new_st = st.radio(
                        "状態",
                        _st_keys,
                        index=_st_keys.index(_cur_st),
                        format_func=lambda v: f"{status_label(v)} — {DATASET_STATUSES[v]['desc']}",
                        key=f"ds_status_{ds.name}",
                    )
                    _new_tags = st.text_input(
                        "タグ（カンマ区切り）",
                        value=", ".join(read_tags(ds)),
                        key=f"ds_tags_{ds.name}",
                        placeholder="例: 屋内, ペッパー, 自動アノテ由来",
                    )
                    _new_note = st.text_area(
                        "メモ（任意）", value=read_note(ds),
                        key=f"ds_note_{ds.name}", height=70,
                        placeholder="照明条件が偏っているので追加撮影の予定あり など",
                    )
                    if st.button("💾 保存", key=f"ds_meta_save_{ds.name}",
                                 type="primary", use_container_width=True):
                        _up = update_provenance(
                            ds, kind="dataset", status=_new_st,
                            tags=_new_tags, note=_new_note)
                        if _up["ok"]:
                            st.success("✅ 保存しました")
                            st.rerun()
                        else:
                            show_error(_up["error"], prefix="❌ 保存できませんでした: ")

                elif _ds_op == "✂️ train/val を分け直す":
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

                elif _ds_op == "🏷 クラス名を編集":
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

                elif _ds_op == "⬇ 持ち出す (ZIP)":
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

                elif _ds_op == "🔍 品質チェック":
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

                elif _ds_op == "➕ 画像を追加":
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
        empty_state(
            "まだ学習済みモデルがありません",
            "「🚀 Step3: モデル学習」で学習すると、ここに一覧が出ます。",
            "他の PC で学習した `.pt` があれば、上の「📤 学習済みモデルをアップロード」から取り込めます。",
        )
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
                # 来歴（status / tags）は run ディレクトリ側に付いている。
                # weights/best.pt なら 2 つ上が run ディレクトリ。
                _run_dir = mp.parent.parent if mp.parent.name == "weights" else mp.parent

                with _label_col:
                    if is_current:
                        st.markdown("⭐ **現在使用中**")
                    st.markdown(f"`{mp.relative_to(MODELS_DIR)}`　"
                                f"{status_label(read_status(_run_dir, 'model'), 'model')}")
                    st.caption(mod_time)
                    _md_tags = read_tags(_run_dir)
                    if _md_tags:
                        st.markdown(
                            " ".join(f'<span class="tc-chip">{t}</span>' for t in _md_tags),
                            unsafe_allow_html=True,
                        )
                    _md_note = read_note(_run_dir)
                    if _md_note:
                        st.caption(f"📝 {_md_note}")
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

                # --- 状態とタグ（人が「これを使うか」を判断するための目印）---
                #     expander の入れ子は不可なのでチェックボックスで開閉する
                if st.checkbox("🏷 状態とタグを編集", key=f"md_meta_{mp}"):
                    _mst_keys = list(MODEL_STATUSES)
                    _md_cur = read_status(_run_dir, "model")
                    _md_new = st.radio(
                        "状態", _mst_keys,
                        index=_mst_keys.index(_md_cur),
                        format_func=lambda v: f"{status_label(v, 'model')} — {MODEL_STATUSES[v]['desc']}",
                        key=f"md_status_{mp}",
                    )
                    _md_new_tags = st.text_input(
                        "タグ（カンマ区切り）", value=", ".join(read_tags(_run_dir)),
                        key=f"md_tags_{mp}", placeholder="例: 近距離向け, 屋内, ベースライン",
                    )
                    _md_new_note = st.text_area(
                        "メモ（任意）", value=read_note(_run_dir),
                        key=f"md_note_{mp}", height=70,
                    )
                    if st.button("💾 保存", key=f"md_meta_save_{mp}",
                                 type="primary", use_container_width=True):
                        _mup = update_provenance(
                            _run_dir, kind="model", status=_md_new,
                            tags=_md_new_tags, note=_md_new_note)
                        if _mup["ok"]:
                            st.success("✅ 保存しました")
                            st.rerun()
                        else:
                            show_error(_mup["error"], prefix="❌ 保存できませんでした: ")

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

                # 統合データセットは親が複数あるので、一本道ではなく木で辿る。
                # 記録が入れ子になっているぶんだけ再帰的に降りていく。
                _SRC_LABELS = {
                    "cvat": "CVAT から生成", "upload_zip": "外部 ZIP の取込",
                    "upload_images": "画像の直接アップロード",
                    "merge": "データセット統合", "unknown": "出所不明",
                }

                def _render_dataset_node(prov: dict, name: str, depth: int = 0) -> None:
                    """データセット 1 つぶんを出し、統合なら親へ降りる"""
                    indent = "　" * depth + ("└ " if depth else "")
                    src = (prov or {}).get("source", "")
                    bits = [f"{indent}📁 `{name}`"]
                    if prov and prov.get("status"):
                        bits.append(status_label(prov["status"]))
                    if src:
                        bits.append(_SRC_LABELS.get(src, src))
                    if prov and prov.get("cvat_tasks"):
                        bits.append("CVAT タスク " + ", ".join(
                            f"[{t.get('id')}] {t.get('name')}"
                            for t in prov["cvat_tasks"]))
                    st.markdown("　".join(bits))

                    if depth > 5:            # 記録が壊れていても止まるように
                        st.caption("　" * (depth + 1) + "…（これ以上は辿りません）")
                        return
                    for _src_snap in (prov or {}).get("sources") or []:
                        _child_name = _src_snap.get("name", "不明")
                        _child_dir = DATA_DIR / _child_name
                        # 統合時のスナップショットを基本に、親が残っていれば今の来歴も見る
                        _child_prov = dict(_src_snap)
                        _live = read_provenance(_child_dir)
                        if _live:
                            _child_prov = {**_live, **{
                                k: v for k, v in _src_snap.items() if k in ("counts",)}}
                        _cnt = _src_snap.get("counts") or {}
                        _render_dataset_node(_child_prov, _child_name, depth + 1)
                        if _cnt:
                            st.caption("　" * (depth + 2)
                                       + "統合した時点: "
                                       + " / ".join(f"{k} {v}枚" for k, v in _cnt.items())
                                       + ("" if _child_dir.exists() else "　※ 現在は削除済み"))

                st.markdown(f"🤖 モデル `{_lin_sel}`")
                if _ld.get("name"):
                    _render_dataset_node(_dsp, _ld["name"], depth=1)
                else:
                    st.caption("　└ 学習データの記録がありません")

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
        empty_state(
            "まだ推論結果がありません",
            "「🔭 Step4: 推論・評価」の「▶ ① 推論する」で推論すると、ここに溜まります。",
        )
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
        st.caption(
            "クラス名の並びが違うデータセット同士でも、統合先の並びに合わせて"
            "クラスIDを振り直します。元のデータセットには手を加えません。"
        )
        if merge_targets:
            _mg_info = []
            for _n in merge_targets:
                _d = DATA_DIR / _n
                _mg_info.append(
                    f"- `{_n}` … {status_label(read_status(_d))}"
                    f"　{' / '.join(f'{k} {v}枚' for k, v in dataset_split_counts(_d).items()) or '（画像なし）'}"
                    f"　クラス: {', '.join(dataset_class_names(_d)) or '—'}"
                )
            st.markdown("\n".join(_mg_info))

        _mg1, _mg2 = st.columns([3, 2])
        with _mg1:
            merge_out_name = st.text_input(
                "統合先ディレクトリ名", value=f"merged_{datetime.now():%Y%m%d_%H%M}")
        with _mg2:
            _mg_status = st.selectbox(
                "統合後の状態",
                list(DATASET_STATUSES),
                format_func=lambda v: status_label(v),
                key="merge_status",
                help="精査済みどうしを混ぜたなら精査済み、"
                     "自動アノテのものが混ざるなら自動アノテのみ、と付けておく",
            )
        _mg_tags = st.text_input("タグ（カンマ区切り・任意）", key="merge_tags")

        if st.button("🔀 統合実行", disabled=len(merge_targets) < 2,
                     type="primary", key="merge_datasets_run"):
            with st.spinner("統合しています…"):
                _mg = merge_datasets(
                    [DATA_DIR / n for n in merge_targets],
                    DATA_DIR / merge_out_name,
                    status=_mg_status,
                    tags=_mg_tags,
                )
            if not _mg["ok"]:
                show_error(_mg["error"], prefix="❌ 統合できませんでした: ")
            else:
                st.success(
                    f"✅ 統合完了: `{merge_out_name}`　"
                    + " / ".join(f"{k} {v}枚" for k, v in _mg["counts"].items())
                    + f"　クラス: {', '.join(_mg['labels'])}"
                )
                for _w in _mg["warnings"]:
                    st.warning(f"⚠ {_w}")
                st.rerun()

