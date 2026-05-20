# detection_dev_ui — 初心者向け 画像検出パイプライン

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](LICENSE)

CVAT → YOLO → MLflow → FiftyOne を Docker Compose で統合した、  
**ターミナル操作不要・GUI完結**の物体検出 MLOps システムです。

> **初心者の方へ**: 機械学習・物体検出の経験が浅い方でも扱えるように設計しています。  
> セットアップ後はブラウザ操作のみで、アノテーション → 学習 → 推論・評価まで一連のワークフローを完結させることができます。  
> ターミナルを使うのはセットアップ時（Step 1〜7）だけです。  
> CVATの形式関連で詰まることが多かったので作成しました。  
> ぜひ、なれたらそれぞれの要素を自分で一通りできるように頑張ってみてください^^

---

## 動作確認済み環境

| 項目 | 要件 |
|---|---|
| OS | Ubuntu 22.04 LTS |
| Docker | 24.x 以上 |
| Docker Compose | v2.x 以上 |
| GPU | NVIDIA GPU（VRAM 8GB 以上推奨） |
| NVIDIAドライバ | 520 以上 |
| CUDA | 12.x 系 |
| nvidia-container-toolkit | 1.14 以上 |

> GPU非搭載環境でも動作しますが、YOLO学習はCPUのみとなり速度が大幅に低下します。  
> CPU動作時は `docker-compose.yml` の `streamlit_app` から `deploy.resources.reservations` ブロックを削除してください。

### ディスク容量の目安

| 対象 | 目安 |
|---|---|
| Docker イメージ（初回ダウンロード） | 30〜50 GB |
| アノテーション画像・データセット | 用途による |
| 学習済みモデル（1モデルあたり） | 10〜200 MB |

> 初回セットアップ時に大量の Docker イメージをダウンロードするため、  
> **余裕を持って 60 GB 以上の空き容量**を確保しておくことを推奨します。

---

## システム構成

```
detection_dev_ui/
├── docker-compose.yml       # 全コンテナ統括
├── .env                     # 認証情報（要作成・gitignore済み）
├── .gitignore
├── LICENSE                  # AGPL-3.0
├── nginx/
│   └── cvat.conf            # CVATリバースプロキシ
├── app/
│   ├── Dockerfile           # Streamlit統合UIコンテナ
│   ├── requirements.txt
│   └── main.py              # Streamlit アプリ本体
├── data/                    # CVATエクスポート先 / YOLO入力元（gitignore済み）
├── models/                  # YOLO学習済み重み（gitignore済み）
└── predictions/             # 推論結果JSON / エクスポート画像（gitignore済み）
    └── exports/             # 結果画像の書き出し先（PNG / JPEG）
```

### サービス一覧とポート

| サービス | コンテナ名 | URL | 役割 |
|---|---|---|---|
| CVAT UI | `cvat_proxy` | http://localhost:8080 | アノテーション |
| MLflow UI | `mlflow` | http://localhost:5000 | 実験管理 |
| Streamlit | `streamlit_app` | http://localhost:8501 | 統合UI（メイン） |
| FiftyOne | `streamlit_app` | http://localhost:5151 | 推論結果可視化 |

---

## セットアップ手順

### Step 1 — nvidia-container-toolkit のインストール

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Step 2 — docker グループへの追加

```bash
sudo usermod -aG docker $USER
newgrp docker

# 動作確認
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi
```

### Step 3 — リポジトリのクローン

```bash
git clone git@github.com:ryotaemi/detection_dev_ui.git
cd detection_dev_ui
```

### Step 4 — .env ファイルの作成

```bash
cat > .env << 'EOF'
COMPOSE_PROJECT_NAME=mlops_workspace
CVAT_USERNAME=admin
CVAT_PASSWORD=your_password_here
CVAT_DB_PASSWORD=your_db_password_here
CVAT_IAM_DB_PASSWORD=your_iam_db_password_here
NVIDIA_VISIBLE_DEVICES=all
EOF
```

> `.env` はGitの追跡対象外です。各パスワードは必ず強い値に変更してください。

### Step 5 — CVATデータベースの初期化

```bash
# DB系コンテナを先に起動（初回のみ）
docker compose up -d cvat_db cvat_redis cvat_iam_db
sleep 30

# マイグレーション実行（初回のみ）
docker compose run --rm cvat_server bash -c "python manage.py migrate"

# スーパーユーザー作成（初回のみ）
docker compose run --rm cvat_server bash -c \
  "python manage.py createsuperuser --username admin --email admin@local.com"
```

### Step 6 — 全サービス起動

```bash
docker compose up -d
sleep 30 && docker compose ps
```

> **初回起動時間の目安**: Docker イメージのダウンロードとビルドに **10〜30 分程度** かかります（回線速度・マシンスペックによって異なります）。  
> 2 回目以降はキャッシュが効くため数分で起動します。

### Step 7 — 疎通確認

```bash
curl -s http://localhost:8080/api/server/about | python3 -m json.tool | head -3  # CVAT
curl -s http://localhost:5000/health                                               # MLflow
curl -s -o /dev/null -w "%{http_code}" http://localhost:8501                       # Streamlit
```

---

## 日常的な起動・停止

```bash
# 起動
docker compose up -d

# 停止
docker compose down

# ログ確認
docker compose logs -f streamlit_app

# アップデート（リポジトリに更新があった場合）
git pull
docker compose build streamlit_app && docker compose up -d streamlit_app
# ※ docker-compose.yml や requirements.txt に変更があった場合は全サービス再起動
# docker compose down && docker compose up -d
```

---

## 運用フロー

```
① CVATでアノテーション
   http://localhost:8080 → admin / 設定したパスワード でログイン
   プロジェクト作成 → 画像アップロード → バウンディングボックス / ポリゴンを付ける

② Streamlit「📤 Step1: データ取込」タブ
   http://localhost:8501 を開く
   「タスク一覧を取得」→ 対象タスクを選択（複数可）
   →「エクスポート実行」→ ラベル・タスク種別（detect/segment）を設定
   →「データセット生成」→ data/ に YOLO 形式で展開される

③ Streamlit「🚀 Step2: モデル学習」タブ
   学習プリセットを選んで「▶ 適用」（または手動でパラメータ設定）
     - 組み込みプリセット: ノーマル / 速度優先 / バランス型 / 精度優先 / 小物体向け / ロボット視点
     - カスタムプリセット: 現在の設定に名前を付けて保存・編集・削除が可能
   「学習開始」→ バックグラウンドで実行
   → エポックごとに mAP50 / Loss のリアルタイムグラフとログが表示される
   → MLflow にメトリクスが自動記録される（http://localhost:5000）
   → 学習完了時にトースト通知＋バルーンアニメーション

④ Streamlit「🔭 Step3: 推論・評価」タブ
   学習済みモデルを選択 →「推論実行」
   → バウンディングボックス付き画像プレビューをグリッドで確認
   「📥 結果画像エクスポート」
   → 「すべて書き出す」または「選択して書き出す」（プレビューを見ながらページ単位で複数選択）
   → PNG / JPEG 形式で predictions/exports/ に保存
   「FiftyOneで可視化」→ http://localhost:5151 でブラウザ確認

⑤ Streamlit「📁 データ管理」タブ（任意）
   data/ のデータセット一覧・削除・統合
   models/ の学習済みモデルをカード形式で表示（mAP50・サイズ・学習日時付き）・削除・使用切替
   predictions/ の推論結果一括クリア
```

### データセット生成後の `data/` ディレクトリ構造

「データセット生成」を実行すると `data/` 以下に以下の構造でファイルが展開されます。

```
data/
└── {タスク名}_{日時}/
    ├── images/
    │   ├── train/      # 学習用画像
    │   └── val/        # 検証用画像
    ├── labels/
    │   ├── train/      # 学習用アノテーション（YOLO形式 .txt）
    │   └── val/        # 検証用アノテーション（YOLO形式 .txt）
    └── data.yaml       # クラス名・パス設定（学習時に自動参照）
```

---

## Dockerfile の CUDA バージョン設定

`app/Dockerfile` のベースイメージと PyTorch インストール URL は、  
**お使いの CUDA バージョンに合わせて変更**してください。

```dockerfile
# 現在の設定（CUDA 12.8 / RTX 50系対応）
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

RUN pip install --no-cache-dir torch torchvision torchaudio \
      --index-url https://download.pytorch.org/whl/cu128
```

| CUDA | `--index-url` の末尾 | ベースイメージタグ例 |
|---|---|---|
| 11.8 | `cu118` | `cuda:11.8.0-cudnn8-runtime-ubuntu22.04` |
| 12.1 | `cu121` | `cuda:12.1.1-cudnn8-runtime-ubuntu22.04` |
| 12.4 | `cu124` | `cuda:12.4.1-cudnn-runtime-ubuntu22.04` |
| 12.6 | `cu126` | `cuda:12.6.3-cudnn-runtime-ubuntu22.04` |
| 12.8 | `cu128` | `cuda:12.8.1-cudnn-runtime-ubuntu22.04` |

> RTX 50系（Blackwell / sm_120）は CUDA 12.8 以上が必要です。cu126 では `no kernel image` エラーが発生します。

利用可能なイメージ一覧: https://hub.docker.com/r/nvidia/cuda/tags  
PyTorch 対応ビルド一覧: https://pytorch.org/get-started/locally/

---

## トラブルシューティング

### `docker` コマンドで permission denied が出る

```bash
sudo usermod -aG docker $USER
newgrp docker
```

### コンテナ名が競合してエラーになる

```bash
docker ps --format "table {{.Names}}\t{{.Ports}}" | grep 8080
docker compose down && docker compose up -d
```

### FiftyOne が Streamlit 内に表示されない

ブラウザのセキュリティポリシー（iframe 制限）により、Streamlit 画面内に埋め込み表示されない場合があります。  
その場合は直接 http://localhost:5151 をブラウザで開いてください。

### GPU 非搭載環境で動かしたい（CPU のみ）

`docker-compose.yml` の `streamlit_app` から以下のブロックを削除してください：

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

---

## 備考

- `version: "3.8"` の警告（`the attribute version is obsolete`）は Docker Compose v2 系の仕様変更によるもので、動作には影響ありません。
- 本システムは完全ローカル動作のため、外部ネットワークへのデータ転送は発生しません。
- `data/`・`models/`・`predictions/` はホスト側の bind mount のため、コンテナを削除してもデータは保持されます。
- CVAT: `v2.64.0` / cvat-sdk: `2.64.0`（サーバーと SDK を同一バージョンに固定）
- ユーザー定義プリセットは `models/.user_presets.json` に保存されます。

---

## ライセンス

このプロジェクトは [GNU Affero General Public License v3.0](LICENSE) の下で公開されています。

使用しているライブラリのライセンス:

| ライブラリ | ライセンス |
|---|---|
| [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) | AGPL-3.0 |
| [CVAT](https://github.com/opencv/cvat) | MIT |
| [MLflow](https://github.com/mlflow/mlflow) | Apache 2.0 |
| [FiftyOne](https://github.com/voxel51/fiftyone) | Apache 2.0 |
| [Streamlit](https://github.com/streamlit/streamlit) | Apache 2.0 |
