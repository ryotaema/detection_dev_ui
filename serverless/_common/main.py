# =============================================================================
# Nuclio 関数エントリポイント (CVAT detector)
# CVAT から base64 画像 + threshold を受け取り、検出結果 JSON を返す。
#
# インタフェース（init_context / handler の入出力）は CVAT 公式の
# serverless 関数の実装に合わせている。
#   https://github.com/cvat-ai/cvat  (Copyright (C) CVAT.ai Corporation / MIT License)
# =============================================================================
import base64
import io
import json

import yaml
from model_handler import ModelHandler
from PIL import Image


def init_context(context):
    context.logger.info("Init context...  0%")

    # function.yaml の annotations.spec からラベル定義を読む
    with open("/opt/nuclio/function.yaml", "rb") as function_file:
        functionconfig = yaml.safe_load(function_file)

    labels_spec = functionconfig["metadata"]["annotations"]["spec"]
    labels = {item["id"]: item["name"] for item in json.loads(labels_spec)}

    context.user_data.model = ModelHandler(labels)

    context.logger.info("Init context...100%")


def handler(context, event):
    context.logger.info("Run custom YOLO model")
    data = event.body
    buf = io.BytesIO(base64.b64decode(data["image"]))
    threshold = float(data.get("threshold", 0.5))
    image = Image.open(buf).convert("RGB")

    results = context.user_data.model.infer(image, threshold)

    return context.Response(
        body=json.dumps(results),
        headers={},
        content_type="application/json",
        status_code=200,
    )
