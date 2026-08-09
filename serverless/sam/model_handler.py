# =============================================================================
# SAM (Segment Anything Model) 用の推論ハンドラ — SAM 2 / SAM 3 共通
#
#   ConceptHandler     … テキスト（名詞句）で該当インスタンスを**全部**出す (PCS)
#                        → CVAT の detector（Actions → Automatic annotation）
#                        **SAM 3 のみ**。SAM 2 はテキストを受け取れない
#   InteractiveHandler … 点/ボックスで指した **1 個だけ**をマスク化 (PVS)
#                        → CVAT の interactor（AI Tools → Interactors）
#                        SAM 2 / SAM 3 のどちらでも動く
#
# 重みの入手経路が 2 つあることに注意:
#   SAM 2 … 非ゲート。Ultralytics が自動ダウンロードするのでビルド時に焼き込める
#   SAM 3 … ゲート付き（Meta の手動承認）。各自が取得したものを
#           models/.sam3/ からマウントして参照する（3.45GB あるため焼き込まない）
#
# mask_to_rle() は CVAT 公式の serverless 関数の実装をそのまま使っている。
#   https://github.com/cvat-ai/cvat  (Copyright (C) CVAT.ai Corporation / MIT License)
# =============================================================================
import hashlib
import json
import os

import numpy as np

# Ultralytics が設定ファイルを書き込む先。書込可能な /tmp に固定して権限エラーを防ぐ
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

SAM_VERSION  = os.environ.get("SAM_VERSION", "sam3").lower()      # sam2 / sam3
WEIGHTS_PATH = os.environ.get("SAM_WEIGHTS_PATH", "/opt/nuclio/sam3/sam3.pt")
# FP16。GPU メモリを減らせるが、環境によっては精度・安定性に影響するため既定はオフ
USE_HALF = os.environ.get("SAM_HALF", "0").lower() not in ("0", "", "false", "no")


def mask_to_rle(mask):
    """CVAT の mask シェイプ（RLE + 外接矩形）に変換する。

    CVAT.ai Corporation の実装 (MIT) をそのまま使用。
    """
    [height, width] = mask.shape
    pixels = (np.asarray(mask).reshape(-1) != 0).astype(np.uint8)
    if pixels.size == 0:
        return []

    changes = np.flatnonzero(pixels[1:] != pixels[:-1]) + 1
    rle = np.diff(np.concatenate(([0], changes, [pixels.size]))).tolist()
    if pixels[0] == 1:
        rle.insert(0, 0)

    rle.extend([0, 0, width - 1, height - 1])
    return rle


def to_bgr(image):
    """PIL 画像を Ultralytics が期待する BGR の ndarray にする。

    Predictor.preprocess は「OpenCV が返す BGR 順」を前提にしているので、
    RGB のまま渡すと色が入れ替わったまま推論されてしまう。
    """
    arr = np.asarray(image.convert("RGB"))
    return arr[:, :, ::-1].copy()


def flatten_bbox(obj_bbox):
    """CVAT から来る `obj_bbox` を [x1, y1, x2, y2] に揃える。

    **CVAT の UI は点のペアの配列 `[[x1, y1], [x2, y2]]` で送ってくる。**
    フラットな `[x1, y1, x2, y2]` を前提にすると `float()` に list を渡して
    落ち、CVAT 側には 500 としてしか見えない（実際に踏んだ）。
    どちらの形でも受け、点が何個来ても外接矩形にまとめる。
    """
    if not obj_bbox:
        return None

    flat = []
    for v in obj_bbox:
        if isinstance(v, (list, tuple)):
            flat.extend(float(x) for x in v)
        else:
            flat.append(float(v))

    if len(flat) < 4:
        return None

    xs, ys = flat[0::2], flat[1::2]
    return [min(xs), min(ys), max(xs), max(ys)]


def parse_prompt_map(raw, labels):
    """`SAM_PROMPTS` を [(CVAT ラベル名, テキストプロンプト), ...] にする。

    SAM 3 のテキストプロンプトは英語の短い名詞句を前提にしているので、
    CVAT 側のラベル名（日本語でも構わない）とは分けて持てるようにしている。
    未設定・壊れている場合はラベル名をそのままプロンプトとして使う。

    Args:
        raw: 環境変数の中身。`[{"label": "...", "prompt": "..."}, ...]`
        labels: function.yaml の annotations.spec から読んだラベル名の並び
    """
    mapping = {}
    if raw:
        try:
            for item in json.loads(raw):
                label = str(item.get("label", "")).strip()
                prompt = str(item.get("prompt", "")).strip()
                if label and prompt:
                    mapping[label] = prompt
        except Exception:
            mapping = {}
    return [(name, mapping.get(name, name)) for name in labels]


class _Base:
    """set_image の結果を使い回すための土台。

    CVAT は同じ画像に対して何度も呼んでくる（特に interactor は 1 クリック 1 回）。
    毎回エンコーダを通すと待ち時間がそのぶん伸びるので、画像が変わったときだけ
    set_image する。
    """

    def __init__(self):
        self._image_key = None

    def _set_image(self, image):
        bgr = to_bgr(image)
        key = hashlib.sha1(bgr.tobytes()).hexdigest()
        if key != self._image_key:
            self.predictor.set_image(bgr)
            self._image_key = key


class ConceptHandler(_Base):
    """テキストプロンプトで概念セグメンテーション (PCS) を行う。"""

    def __init__(self, pairs):
        # pairs: [(CVAT ラベル名, テキストプロンプト), ...]
        # 並び順がそのまま SAM 3 のクラス index になるので、順序を崩さないこと
        super().__init__()
        from ultralytics.models.sam import SAM3SemanticPredictor

        self.labels = [p[0] for p in pairs]
        self.prompts = [p[1] for p in pairs]

        overrides = {
            "model": WEIGHTS_PATH,
            "conf": 0.25,
            "save": False,
            "verbose": False,
        }
        if USE_HALF:
            overrides["half"] = True
        self.predictor = SAM3SemanticPredictor(overrides=overrides)
        # 重みの読み込みは 3.45GB 分かかる。最初のリクエストで待たせないよう
        # init_context の時点で済ませておく（function.yaml で readiness を延ばしてある）
        self.predictor.setup_model()

    def infer(self, image, threshold):
        if not self.prompts:
            return []

        self.predictor.args.conf = float(threshold)
        self._set_image(image)
        results = self.predictor(text=self.prompts)
        if not results:
            return []

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        masks = getattr(result, "masks", None)
        mask_xy = list(getattr(masks, "xy", []) or []) if masks is not None else []

        detections = []
        for idx, box in enumerate(boxes):
            cls_id = int(box.cls.item())
            confidence = float(box.conf.item())
            # クラス index は self.prompts の並びに対応する。
            # 同じ語を 2 度書かれても取り違えないよう、名前ではなく index で引く
            label = self.labels[cls_id] if 0 <= cls_id < len(self.labels) else str(cls_id)

            if idx < len(mask_xy) and len(mask_xy[idx]) >= 3:
                # CVAT の polygon は [x1, y1, x2, y2, ...] のフラットな配列
                points = [float(v) for xy in mask_xy[idx] for v in xy]
                shape_type = "polygon"
            else:
                points = [float(v) for v in box.xyxy[0].tolist()]
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


class InteractiveHandler(_Base):
    """点・ボックスで指した 1 個だけをマスク化する (PVS)。"""

    def __init__(self):
        super().__init__()
        # 点・ボックスの受け取り方は SAM 2 / SAM 3 で共通（SAM3Predictor は
        # SAM2Predictor の _prepare_prompts をそのまま継承している）ので、
        # 差し替えるのは Predictor のクラスだけでよい
        if SAM_VERSION == "sam2":
            from ultralytics.models.sam import SAM2Predictor as _Predictor
        else:
            from ultralytics.models.sam import SAM3Predictor as _Predictor

        overrides = {
            "model": WEIGHTS_PATH,
            "save": False,
            "verbose": False,
            # **conf を下げておくこと。** Predictor.postprocess は
            # `pred_scores > self.args.conf` でマスクを捨てる。既定の 0.25 のままだと、
            # 難しい対象をクリックしたときに**何も返らない**（利用者からは
            # 「反応しない」ようにしか見えない）。人が指した以上は何かを返し、
            # 採否は人に決めてもらう
            "conf": float(os.environ.get("SAM_INTERACTIVE_CONF", "0.05")),
        }
        if USE_HALF:
            overrides["half"] = True
        self.predictor = _Predictor(overrides=overrides)
        self.predictor.setup_model()

    def handle(self, image, pos_points, neg_points, obj_bbox):
        self._set_image(image)

        pos = [[float(p[0]), float(p[1])] for p in (pos_points or [])]
        neg = [[float(p[0]), float(p[1])] for p in (neg_points or [])]

        kwargs = {}
        if pos or neg:
            # **必ず (1, N, 2) の 3 次元で渡すこと。**
            # (N, 2) を渡すと Predictor._prepare_prompts が (N, 1, 2) に直し、
            # 「点の数だけ別々のオブジェクト」として扱われてしまう。
            # ここで欲しいのは「N 個の点で指した 1 個のオブジェクト」。
            kwargs["points"] = [pos + neg]
            kwargs["labels"] = [[1] * len(pos) + [0] * len(neg)]
        box = flatten_bbox(obj_bbox)
        if box:
            # bbox は _prepare_prompts が点列の先頭に連結してくれるので、
            # 点と併用しても 1 オブジェクトとして扱われる
            kwargs["bboxes"] = [box]

        if not kwargs:
            return []

        results = self.predictor(**kwargs)
        if not results:
            return []

        result = results[0]
        masks = getattr(result, "masks", None)
        data = getattr(masks, "data", None) if masks is not None else None
        if data is None or len(data) == 0:
            return []

        # 通常は 1 枚だけ返る（multimask_output=False）が、複数返ったときは
        # 先頭ではなくスコアが最も高いものを採る（並び順はスコア順とは限らない）
        idx = 0
        boxes = getattr(result, "boxes", None)
        conf = getattr(boxes, "conf", None) if boxes is not None else None
        if conf is not None and len(conf) == len(data) and len(conf) > 1:
            idx = int(conf.argmax().item())

        # マスクは postprocess で元画像サイズに戻っているので、
        # そのまま RLE にすれば CVAT の座標系と一致する
        mask = data[idx].cpu().numpy().astype(np.uint8)
        return mask_to_rle(mask)
