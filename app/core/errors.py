# =============================================================================
# エラーの解釈と対処の提示
#
#   例外メッセージをそのまま出しても「次に何をすればよいか」が分からない。
#   よく出るパターンを見分けて、具体的な対処を添える。
#   当てはまらないときは None を返し、呼び出し側は素のメッセージだけを出す。
# =============================================================================
from __future__ import annotations

from typing import Optional

# (判定に使う語の集合, タイトル, 対処) の一覧。
# 語はすべて小文字で持ち、メッセージ側も小文字化して部分一致で見る。
_PATTERNS: list[tuple[tuple[str, ...], str, str]] = [
    (
        ("cuda out of memory", "out of memory", "cublas_status_alloc_failed"),
        "GPU メモリが足りません",
        "- `batch` を半分に下げてください（例: 16 → 8）\n"
        "- それでも足りなければ `imgsz` を下げます（例: 1280 → 640）\n"
        "- 他に学習・推論が動いていないか確認してください"
        "（`docker compose ps` と `nvidia-smi`）\n"
        "- `amp` を有効にするとメモリ使用量が下がります",
    ),
    (
        ("no kernel image is available",),
        "GPU とビルドの組み合わせが合っていません",
        "この GPU（Blackwell / RTX 50 系）には CUDA 12.8 + cu128 が必要です。\n"
        "`app/Dockerfile` のベースイメージと PyTorch のインストール URL が"
        "セットで cu128 になっているか確認してください。",
    ),
    (
        ("dataset not found", "no such file or directory", "does not exist",
         "no labels found", "no images found"),
        "データセットのパスかファイルが見つかりません",
        "- `data.yaml` の `path` / `train` / `val` が正しいか確認してください\n"
        "- 「📁 データ管理」の品質チェックで、画像とラベルの対応漏れを検出できます\n"
        "- 他の PC から持ち込んだデータは `path` が別環境の絶対パスのままのことがあります",
    ),
    (
        ("nothing to resume",),
        "この学習は再開できません",
        "予定していたエポックを完了した学習は再開できません。\n"
        "続けて学習したい場合は、その `best.pt` を初期重みに指定して"
        "新しい学習を始めてください（Step3 のモデル名に "
        "`models/<run>/weights/best.pt` を入力）。",
    ),
    (
        ("size mismatch", "shape mismatch", "unexpected key", "missing key",
         "number of classes"),
        "モデルとデータのクラス数が合っていません",
        "- 追加学習のとき、初期重みのクラス数と `data.yaml` の `nc` が"
        "違うと起きます\n"
        "- クラス構成を変えた場合は、事前学習済みモデル"
        "（`yolo11s.pt` など）から学習し直してください",
    ),
    (
        ("connection", "connect", "timeout", "timed out", "max retries",
         "refused", "unreachable"),
        "サービスに接続できません",
        "- サイドバーの「サービス状態」で CVAT / MLflow の状態を確認してください\n"
        "- 停止している場合は `docker compose up -d` で起動します\n"
        "- CVAT は起動直後しばらく応答しないことがあります（1〜2分待つ）",
    ),
    (
        ("401", "403", "unauthorized", "forbidden", "authentication",
         "invalid credentials"),
        "CVAT の認証に失敗しました",
        "`.env` の `CVAT_USERNAME` / `CVAT_PASSWORD` が"
        "CVAT のアカウントと一致しているか確認してください。\n"
        "変更した場合は `docker compose up -d streamlit_app` で再読み込みが必要です。",
    ),
    (
        # アポストロフィの有無で取りこぼさないよう "get attribute" で拾う
        ("weights_only", "unsupported global", "unpickling", "_pickle",
         "get attribute", "modulenotfounderror", "ultralytics.nn"),
        "モデルファイルをこの環境で読み込めません",
        "学習元の ultralytics のバージョンが、この環境（8.4.48）と"
        "離れている可能性があります。\n"
        "- 学習元と同じバージョンで `.pt` を再保存してもらう\n"
        "- または学習元で ONNX などに書き出してもらう\n"
        "- 「📁 データ管理」からアップロードすると、読み込み検証の結果を確認できます",
    ),
    (
        ("permission denied", "read-only file system", "errno 13"),
        "ファイルの書き込み権限がありません",
        "コンテナ内は root で動くため、ホスト側で作ったディレクトリの"
        "所有者と食い違うことがあります。\n"
        "対象ディレクトリの権限を確認してください。",
    ),
    (
        ("no space left", "errno 28", "disk quota"),
        "ディスクの空き容量がありません",
        "- `predictions/` の結果をクリアする（データ管理タブ）\n"
        "- 不要なデータセット・モデルを削除する\n"
        "- `docker system prune` で未使用イメージを整理する",
    ),
    (
        ("expanders may not be nested",),
        "画面の組み立て方の問題です（不具合）",
        "`st.expander` を入れ子にすると発生します。"
        "`st.checkbox` + `st.container(border=True)` に置き換えてください。",
    ),
]


def explain_error(message: str) -> Optional[dict]:
    """エラーメッセージから、よくある原因と対処を返す。

    Returns: {"title": str, "hint": str} / 判別できなければ None
    """
    if not message:
        return None
    low = str(message).lower()
    for keys, title, hint in _PATTERNS:
        if any(k in low for k in keys):
            return {"title": title, "hint": hint}
    return None
