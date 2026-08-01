# =============================================================================
# FiftyOne セッション管理
# =============================================================================
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import streamlit as st

from .config import FIFTYONE_PORT


# ---------------------------------------------------------------------------
# FiftyOne セッション管理
# ---------------------------------------------------------------------------
def launch_fiftyone(dataset_name: str, predictions_dir: Path) -> Optional[int]:
    """
    FiftyOne データセットを作成し、Appを起動してポート番号を返す。
    既存のセッションがあれば再利用。

    Fix: remote=True → remote=False, address="0.0.0.0"
        コンテナ内で 0.0.0.0:5151 でListenさせてホストブラウザからアクセス可能にする。
    """
    try:
        import fiftyone as fo

        # 既存データセットをリセット
        if fo.dataset_exists(dataset_name):
            fo.delete_dataset(dataset_name)

        dataset = fo.Dataset(name=dataset_name)

        # predictions_dir の JSON ファイルを読み込んでサンプル追加
        json_files = list(predictions_dir.glob("*.json"))
        if not json_files:
            st.warning("predictions/ に結果JSONがありません。先に推論を実行してください。")
            return None

        samples = []
        for jf in json_files:
            with open(jf) as f:
                pred = json.load(f)

            img_path = pred.get("image_path", "")
            detections = []
            for box in pred.get("boxes", []):
                detections.append(
                    fo.Detection(
                        label=box["label"],
                        bounding_box=box["bbox_xywhn"],  # [x, y, w, h] 正規化済
                        confidence=box.get("confidence", 1.0),
                    )
                )
            sample = fo.Sample(filepath=img_path)
            sample["predictions"] = fo.Detections(detections=detections)
            samples.append(sample)

        dataset.add_samples(samples)

        # 既存セッションを閉じる
        if st.session_state.fiftyone_session:
            try:
                st.session_state.fiftyone_session.close()
            except Exception:
                pass

        # Fix: remote=False, address="0.0.0.0" でコンテナ外から直接アクセス可能に
        session = fo.launch_app(
            dataset,
            port=FIFTYONE_PORT,
            address="0.0.0.0",
            remote=False,
        )
        st.session_state.fiftyone_session = session
        st.session_state.fiftyone_port = FIFTYONE_PORT
        return FIFTYONE_PORT

    except Exception as e:
        st.error(f"FiftyOne エラー: {e}")
        return None


def launch_fiftyone_comparison(dataset_name: str, per_image: list[dict]) -> Optional[int]:
    """GT と予測の両方を載せた FiftyOne データセットを作って App を起動する。

    FiftyOne の bounding_box は [左上x, 左上y, w, h] の正規化形式。
    """
    try:
        import fiftyone as fo

        if fo.dataset_exists(dataset_name):
            fo.delete_dataset(dataset_name)
        dataset = fo.Dataset(name=dataset_name)

        samples = []
        for item in per_image:
            s = fo.Sample(filepath=item["image"])
            s["ground_truth"] = fo.Detections(detections=[
                fo.Detection(label=g["label"], bounding_box=g["bbox_xywhn"])
                for g in item["gt_boxes"]
            ])
            s["predictions"] = fo.Detections(detections=[
                fo.Detection(label=p["label"], bounding_box=p["bbox_xywhn"],
                             confidence=p.get("confidence", 1.0))
                for p in item["pred_boxes"]
            ])
            # FiftyOne 上でソート・フィルタできるようフィールドにも入れる
            s["n_fn"] = item["fn"]
            s["n_fp"] = item["fp"]
            s["n_tp"] = item["tp"]
            samples.append(s)

        dataset.add_samples(samples)

        if st.session_state.fiftyone_session:
            try:
                st.session_state.fiftyone_session.close()
            except Exception:
                pass

        session = fo.launch_app(dataset, port=FIFTYONE_PORT,
                                address="0.0.0.0", remote=False)
        st.session_state.fiftyone_session = session
        st.session_state.fiftyone_port = FIFTYONE_PORT
        return FIFTYONE_PORT
    except Exception as e:
        st.error(f"FiftyOne エラー: {e}")
        return None
