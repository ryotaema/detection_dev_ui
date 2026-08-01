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
        detections = []
        if not results:
            return detections

        result = results[0]
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

            if isinstance(self.names, dict):
                label = self.names.get(cls_id, str(cls_id))
            else:
                label = self.names[cls_id]

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
                    "label": label,
                    "points": points,
                    "type": shape_type,
                }
            )

        return detections
