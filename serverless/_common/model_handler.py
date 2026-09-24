# =============================================================================
# 自作 YOLO (Ultralytics) モデル用の推論ハンドラ
# CVAT の Nuclio serverless 関数から呼び出される共通実装。
# best.pt はビルド時に /opt/nuclio/best.pt へ配置される (serverless/deploy.sh が担当)。
# =============================================================================
import os

from ultralytics import YOLO

# Ultralytics が設定ファイルを書き込む先。書込可能な /tmp に固定して権限エラーを防ぐ
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

MODEL_PATH = os.environ.get("MODEL_PATH", "/opt/nuclio/best.pt")


class ModelHandler:
    def __init__(self, labels):
        # labels: {id: name} — function.yaml の spec から渡される (参照用)
        self.labels = labels
        self.model = YOLO(MODEL_PATH)
        # モデル自身が保持するクラス名 (学習時の names) を正とする
        self.names = self.model.names

    def infer(self, image, threshold):
        results = self.model.predict(source=image, conf=threshold, verbose=False)
        if not results:
            return []
        return to_cvat(results[0], self.names, threshold)


def _label(names, cls_id):
    if isinstance(names, dict):
        return names.get(cls_id, str(cls_id))
    return names[cls_id]


def to_cvat(result, names, threshold):
    """Ultralytics の推論結果 1 枚分を CVAT の detector の返り値にする。

    タスクごとに結果の入れ物が違うので、それぞれ拾う:
      detect   → rectangle
      segment  → polygon（輪郭）
      obb      → polygon（回転矩形の 4 隅。CVAT で 4 点ポリゴンとして編集できる）
      pose     → rectangle（キーポイントは CVAT の skeleton 定義が要るため未対応）
      classify → tag（画像単位のラベル。最上位クラスがしきい値以上のときだけ）
    """
    detections = []

    # ── classify ──
    probs = getattr(result, "probs", None)
    if probs is not None:
        cls_id = int(probs.top1)
        confidence = float(probs.top1conf)
        if confidence >= threshold:
            detections.append({
                "confidence": str(confidence),
                "label": _label(names, cls_id),
                "type": "tag",
            })
        return detections

    # ── obb ──
    obb = getattr(result, "obb", None)
    if obb is not None:
        corners = obb.xyxyxyxy.tolist()
        for cls, conf, pts in zip(obb.cls.tolist(), obb.conf.tolist(), corners):
            detections.append({
                "confidence": str(float(conf)),
                "label": _label(names, int(cls)),
                "points": [float(v) for xy in pts for v in xy],
                "type": "polygon",
            })
        return detections

    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return detections

    # セグメンテーションモデルの場合はインスタンスごとの輪郭が入る。
    # CVAT へは polygon として返すと、そのままポリゴンとして編集できる。
    masks = getattr(result, "masks", None)
    mask_xy = list(getattr(masks, "xy", []) or []) if masks is not None else []

    for idx, box in enumerate(boxes):
        cls_id = int(box.cls.item())
        confidence = float(box.conf.item())
        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())

        if idx < len(mask_xy) and len(mask_xy[idx]) >= 3:
            # CVAT の polygon は [x1, y1, x2, y2, ...] のフラットな配列
            points = [float(v) for xy in mask_xy[idx] for v in xy]
            shape_type = "polygon"
        else:
            points = [x1, y1, x2, y2]
            shape_type = "rectangle"

        detections.append(
            {
                "confidence": str(confidence),
                "label": _label(names, cls_id),
                "points": points,
                "type": shape_type,
            }
        )

    return detections
