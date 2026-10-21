# =============================================================================
# セマンティックセグメンテーション（小型 U-Net）用の推論ハンドラ
# CVAT の Nuclio serverless 関数から呼ばれる。
#
# Ultralytics のモデルとは扱いが違う点:
#   - チェックポイントは state_dict のみ。アーキテクチャは smallunet.py が持つ
#   - 出力は「画素ごとのクラス」であってインスタンスではない。
#     CVAT へ返すには連結成分に分けて polygon にする（後述）
#   - 入力が RGB + 深度の 4ch。**CVAT は RGB しか渡してこない**ので、
#     深度チャンネルは DEPTH_MODE に従って埋める（既定 zero = 基準面と同一深度）
# =============================================================================
import json
import os

import cv2
import numpy as np
import torch

from smallunet import create_model

MODEL_PATH = os.environ.get("MODEL_PATH", "/opt/nuclio/model.pt")

# 深度チャンネルの埋め方。CVAT からは深度が来ないので何かで代用するしかない。
#   zero  … 全画素を 0 にする
#   const:<値> … 全画素をその値にする（学習時の値域は -1.0〜1.0）
#
# **どれを選んでも精度は大きく落ちる。埋めるのは代用であって復元ではない。**
# このモデルの第1層は深度チャンネルに RGB と同等の重み
# （L2 ノルムで RGB 平均の 1.03 倍）を割り当てており、深度は補助ではなく主入力。
# 一定値で埋めると、その値そのものが強い偽の手がかりになる（40 枚平均の実測）:
#
#     depth=-1.0 … fruit  0.003% / peduncle 0.000%
#     depth= 0.0 … fruit 24.190% / peduncle 0.032%   ← 0 埋め
#     depth= 0.5 … fruit  0.529% / peduncle 1.623%
#
# 0 は「全画素が基準面と同じ深度にある」という主張になるため、
# モデルは fruit を過剰に出し peduncle をほぼ出さなくなる。
# 「学習時も欠測画素は 0 だから」は理由にならない。
# 部分的に 0 が混じるのと全画素が 0 なのは、学習中に現れない別の入力。
#
# 深度が本当に要る用途（実機など）では、ここを埋めずに実際の深度を渡すこと。
# CVAT 経由では画像しか届かないため、この関数からは渡せない。
DEPTH_MODE = os.environ.get("DEPTH_MODE", "zero")


def _depth_fill_value(mode):
    """DEPTH_MODE から埋める値を決める。未知の指定は落とす（黙って 0 にしない）。"""
    if mode == "zero":
        return 0.0
    if mode.startswith("const:"):
        return float(mode.split(":", 1)[1])
    raise ValueError(f"未知の DEPTH_MODE: {mode}（zero か const:<値>）")

# ポリゴン化のパラメータ
MIN_AREA_RATIO = float(os.environ.get("MIN_AREA_RATIO", "0.0002"))  # 画像面積に対する下限
POLY_EPSILON_RATIO = float(os.environ.get("POLY_EPSILON_RATIO", "0.004"))  # 輪郭長に対する単純化量

# 画素ごとの分類なので、返さないクラスがある
#   background … 背景そのもの
#   ignore     … 学習時に損失から外した画素（255）
SKIP_LABELS = {"background", "ignore", "__background__"}


def _to_class_names(labels, num_classes):
    """checkpoint の labels（{名前: id}）を id 順の名前一覧に直す。

    labels が無い / 壊れている場合はクラス番号を名前として使う
    （関数が起動しないより、名前が味気ないほうがましなため）。
    """
    names = {}
    if isinstance(labels, dict):
        for name, idx in labels.items():
            try:
                names[int(idx)] = str(name)
            except (TypeError, ValueError):
                continue
    return [names.get(i, str(i)) for i in range(int(num_classes))]


class ModelHandler:
    def __init__(self, labels):
        # labels: {id: name} — function.yaml の spec から渡される（参照用）
        self.spec_labels = labels

        ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        config = dict(ckpt.get("config") or {})

        self.width = int(config.get("image_width", 320))
        self.height = int(config.get("image_height", 240))
        self.in_channels = int(config.get("in_channels", 4))
        self.num_classes = int(config.get("num_classes", 2))
        self.relative_depth_max_mm = float(config.get("relative_depth_max_mm", 150.0))
        self.names = _to_class_names(ckpt.get("labels"), self.num_classes)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = create_model(
            in_channels=self.in_channels,
            num_classes=self.num_classes,
            base_channels=int(config.get("base_channels", 24)),
        ).to(self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

    def describe(self):
        return json.dumps(
            {
                "input": [self.width, self.height, self.in_channels],
                "classes": self.names,
                "depth_mode": DEPTH_MODE,
                "depth_fill": _depth_fill_value(DEPTH_MODE) if self.in_channels > 3 else None,
                "device": str(self.device),
            },
            ensure_ascii=False,
        )

    def _make_input(self, rgb):
        """PIL 由来の RGB 配列を、学習時と同じ前処理で 4ch テンソルにする。

        学習側の make_input と同じ順序（リサイズ → 0..1 正規化 → 深度を後ろに連結）に
        すること。ここがずれると重みは読めても結果が合わない。
        """
        small = cv2.resize(rgb, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        x = small.astype(np.float32) / 255.0

        extra = self.in_channels - 3
        if extra > 0:
            filler = np.full((self.height, self.width, extra),
                             _depth_fill_value(DEPTH_MODE), dtype=np.float32)
            x = np.dstack([x, filler])

        return torch.from_numpy(x.transpose(2, 0, 1)[None])

    def infer(self, image, threshold):
        rgb = np.asarray(image.convert("RGB"))
        src_h, src_w = rgb.shape[:2]

        with torch.no_grad():
            logits = self.model(self._make_input(rgb).to(self.device))
            probs_small = torch.softmax(logits, dim=1)[0].cpu().numpy()

        # 確率のまま元の解像度へ戻す（マスクを拡大するとギザギザになるため）
        probs = np.stack(
            [
                cv2.resize(probs_small[i], (src_w, src_h), interpolation=cv2.INTER_LINEAR)
                for i in range(probs_small.shape[0])
            ]
        )
        mask = np.argmax(probs, axis=0).astype(np.int32)

        min_area = max(4.0, MIN_AREA_RATIO * src_w * src_h)
        detections = []
        for cls_id in range(probs.shape[0]):
            name = self.names[cls_id] if cls_id < len(self.names) else str(cls_id)
            if name.lower() in SKIP_LABELS:
                continue

            # 確率がしきい値に満たない画素は落とす（CVAT のスライダに対応させる）
            binary = ((mask == cls_id) & (probs[cls_id] >= float(threshold))).astype(np.uint8)
            if not binary.any():
                continue

            detections.extend(self._to_polygons(binary, probs[cls_id], name, min_area))

        return detections

    @staticmethod
    def _to_polygons(binary, prob, label, min_area):
        """クラスの2値マスクを、連結成分ごとの polygon に分ける。

        画素ごとの分類にはインスタンスの区別が無いので、
        つながっている領域を 1 個として扱う（CVAT 側で分割・結合できる）。
        輪郭の点は approxPolyDP で減らす。数百点のまま返すと人が編集できない。
        """
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for contour in contours:
            if cv2.contourArea(contour) < min_area:
                continue

            eps = POLY_EPSILON_RATIO * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, eps, True)
            if len(approx) < 3:
                continue

            # 信頼度は領域内の平均確率。画素単位の確率を 1 個の値にまとめる
            region = np.zeros(binary.shape, dtype=np.uint8)
            cv2.drawContours(region, [contour], -1, 1, thickness=cv2.FILLED)
            confidence = float(prob[region.astype(bool)].mean())

            out.append(
                {
                    "confidence": str(confidence),
                    "label": label,
                    "points": [float(v) for point in approx.reshape(-1, 2) for v in point],
                    "type": "polygon",
                }
            )
        return out
