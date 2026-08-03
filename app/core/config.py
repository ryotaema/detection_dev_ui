# =============================================================================
# パス・URL・共通定数
# =============================================================================
from __future__ import annotations

import os
from pathlib import Path



# ---------------------------------------------------------------------------
# 定数・環境変数
# ---------------------------------------------------------------------------
DATA_DIR       = Path(os.getenv("DATA_DIR",       "/workspace/data"))
MODELS_DIR     = Path(os.getenv("MODELS_DIR",     "/workspace/models"))
PREDICTIONS_DIR      = Path(os.getenv("PREDICTIONS_DIR","/workspace/predictions"))
PREDICTIONS_VIDEOS_DIR = PREDICTIONS_DIR / "videos"
SERVERLESS_DIR = Path(os.getenv("SERVERLESS_DIR", "/workspace/serverless"))
# 別リポジトリのツールを clone して置く場所。1 ディレクトリ = 1 拡張
EXTENSIONS_DIR = Path(os.getenv("EXTENSIONS_DIR", "/workspace/extensions"))
EXTENSION_MANIFEST = "extension.json"
CVAT_NETWORK   = os.getenv("CVAT_NETWORK", "")                          # Nuclio 関数を載せる網
NUCLIO_WEB     = os.getenv("NUCLIO_DASHBOARD", "http://localhost:8070") # ブラウザ表示用
CVAT_HOST      = os.getenv("CVAT_HOST",     "http://cvat-server:8080")  # コンテナ内通信用
CVAT_WEB       = os.getenv("CVAT_WEB_HOST", "http://localhost:8080")    # ブラウザ表示用
CVAT_USER      = os.getenv("CVAT_USERNAME","admin")
CVAT_PASS      = os.getenv("CVAT_PASSWORD","admin")
MLFLOW_URI     = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MLFLOW_WEB     = os.getenv("MLFLOW_WEB_HOST", "http://localhost:5000")
FIFTYONE_PORT  = int(os.getenv("FIFTYONE_PORT","5151"))

for d in [DATA_DIR, MODELS_DIR, PREDICTIONS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
# ---------------------------------------------------------------------------
# 来歴 (provenance) 管理
#
#   「このモデルはどのデータで学習されたか」を追跡できるようにする。
#   データを足しながら追加学習を重ねると、モデルとデータの対応が分からなくなり、
#   精度が落ちたときに原因を切り分けられなくなるため。
#
#   - データセット側: data/<name>/.provenance.json   （何から作ったか）
#   - モデル側      : models/<run>/.provenance.json  （何で学習したか）
#     モデル側にはデータセット側の内容をコピーして持たせる。
#     データセットが後で削除・変更されても、学習時点の情報を失わないようにするため。
# ---------------------------------------------------------------------------
PROVENANCE_FILE = ".provenance.json"


# ---------------------------------------------------------------------------
# データセット品質チェック
#
#   外部から持ち込んだデータや、複数人で分担したアノテーションほど
#   「画像とラベルの対応漏れ」「座標の壊れ」「クラス分布の偏り」が起きやすい。
#   学習を回す前に機械的に検査する。
# ---------------------------------------------------------------------------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# ---------------------------------------------------------------------------
# CVAT 自動アノテーション (Nuclio serverless) 連携
#
#   models/<run>/weights/best.pt
#     → serverless/custom/<slug>/{function.yaml, function-gpu.yaml, model.env} を自動生成
#     → serverless/deploy.sh を実行
#     → CVAT の「Actions → Automatic annotation」に出現
#
#   ラベル定義 (annotations.spec) はモデルのクラス名から生成するため、
#   「モデルのクラス名 = 関数のラベル名」が機械的に保証される。
#   （CVAT タスク側のラベル名との一致だけは利用者が担保する必要がある）
# ---------------------------------------------------------------------------
NUCTL_BIN = SERVERLESS_DIR / "bin" / "nuctl"


# ===========================================================================
# UI ヘルパー：ポップオーバー付きウィジェット
# ===========================================================================
_DOC_TRAIN = "https://docs.ultralytics.com/modes/train/#train-settings"
_DOC_AUG   = "https://docs.ultralytics.com/modes/train/#augmentation-settings-and-hyperparameters"



_MODEL_OPTS = [
    "yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x",
    "yolo11n-seg", "yolo11s-seg", "yolo11m-seg", "yolo11l-seg", "yolo11x-seg",
    "yolo11n-pose", "yolo11s-pose", "yolo11m-pose", "yolo11l-pose", "yolo11x-pose",
    "yolo11n-cls", "yolo11s-cls", "yolo11m-cls", "yolo11l-cls", "yolo11x-cls",
    "yolo11n-obb", "yolo11s-obb", "yolo11m-obb", "yolo11l-obb", "yolo11x-obb",
    "yolo26n", "yolo26s", "yolo26m", "yolo26l", "yolo26x",
    "その他",
]
