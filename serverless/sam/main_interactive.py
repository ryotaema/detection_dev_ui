# =============================================================================
# Nuclio 関数エントリポイント (SAM 3 / interactor)
#
# CVAT の「AI Tools → Interactors」から呼ばれる。
# 対象を囲んだボックスと、正/負の点を受け取り、その 1 個のマスクを返す。
#
# 入出力（pos_points / neg_points / obj_bbox → shapes[type=mask]）は CVAT 公式の
# interactor 関数の実装に合わせている。
#   https://github.com/cvat-ai/cvat  (Copyright (C) CVAT.ai Corporation / MIT License)
# =============================================================================
import base64
import io
import json

import numpy as np
from model_handler import InteractiveHandler
from PIL import Image


def init_context(context):
    context.logger.info("Init context...  0%")
    context.user_data.model = InteractiveHandler()
    context.logger.info("Init context...100%")


def handler(context, event):
    context.logger.info("Run SAM 3 interactive segmentation")
    data = event.body
    pos_points = data.get("pos_points") or []
    neg_points = data.get("neg_points") or []
    obj_bbox = data.get("obj_bbox", None)

    buf = io.BytesIO(base64.b64decode(data["image"]))
    image = Image.open(buf).convert("RGB")

    # ボックスも点も無ければ何も返せない。
    # 負の点だけが来た場合は、CVAT 公式の interactor に倣って
    # その外接矩形を対象範囲とみなす（点は無かったことにする）。
    if obj_bbox is None and not pos_points and neg_points:
        arr = np.array(neg_points, dtype=float)
        obj_bbox = [
            float(arr[:, 0].min()), float(arr[:, 1].min()),
            float(arr[:, 0].max()), float(arr[:, 1].max()),
        ]
        neg_points = []

    points = context.user_data.model.handle(image, pos_points, neg_points, obj_bbox)

    return context.Response(
        body=json.dumps(
            {
                "shapes": [
                    {
                        "points": points,
                        "group": 0,
                        "source": "semi-auto",
                        "attributes": [],
                        "occluded": False,
                        "rotation": 0,
                        "type": "mask",
                    }
                ]
            }
        ),
        headers={},
        content_type="application/json",
        status_code=200,
    )
