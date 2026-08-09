# =============================================================================
# Nuclio 関数エントリポイント (SAM 3 / detector)
#
# CVAT の「Actions → Automatic annotation」から呼ばれる。
# function.yaml の annotations.spec に並べたラベルを、環境変数 SAM_PROMPTS で
# テキストプロンプトに読み替えて SAM 3 に渡し、該当するインスタンスを全部返す。
#
# インタフェース（init_context / handler の入出力）は CVAT 公式の
# serverless 関数の実装に合わせている。
#   https://github.com/cvat-ai/cvat  (Copyright (C) CVAT.ai Corporation / MIT License)
# =============================================================================
import base64
import io
import json
import os

import yaml
from model_handler import ConceptHandler, parse_prompt_map
from PIL import Image


def init_context(context):
    context.logger.info("Init context...  0%")

    with open("/opt/nuclio/function.yaml", "rb") as function_file:
        functionconfig = yaml.safe_load(function_file)

    labels_spec = functionconfig["metadata"]["annotations"]["spec"]
    # ラベルの並び順がそのまま SAM 3 のクラス index になるので、
    # id ではなく spec に書かれた順序をそのまま使う
    labels = [item["name"] for item in json.loads(labels_spec)]

    pairs = parse_prompt_map(os.environ.get("SAM_PROMPTS", ""), labels)
    context.logger.info(f"SAM3 prompts: {pairs}")

    context.user_data.model = ConceptHandler(pairs)

    context.logger.info("Init context...100%")


def handler(context, event):
    context.logger.info("Run SAM 3 concept segmentation")
    data = event.body
    buf = io.BytesIO(base64.b64decode(data["image"]))
    threshold = float(data.get("threshold", 0.25))
    image = Image.open(buf).convert("RGB")

    results = context.user_data.model.infer(image, threshold)

    return context.Response(
        body=json.dumps(results),
        headers={},
        content_type="application/json",
        status_code=200,
    )
