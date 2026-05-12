# MLOps Pipeline — 完全ローカル完結型 画像検出パイプライン

CVAT → YOLO → ClearML → FiftyOne を Docker Compose で統合した、  
**ターミナル操作不要・GUI完結**の物体検出MLOpsシステムです。

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

---

## システム構成

```
mlops_workspace/
├── docker-compose.yml       # 全コンテナ統括
├── .env                     # 認証情報（要作成・gitignore済み）
├── .gitignore
├── nginx/
│   └── cvat.conf            # CVATリバースプロキシ
├── app/
│   ├── Dockerfile           # Streamlit統合UIコンテナ
│   ├── requirements.txt
│   └── main.py              # Streamlit アプリ本体
├── data/                    # CVATエクスポート先 / YOLO入力元
├── models/                  # YOLO学習済み重み
└── predictions/             # 推論結果JSON
```

### サービス一覧とポート

| サービス | コンテナ名 | URL | 役割 |
|---|---|---|---|
| CVAT UI | `cvat_proxy` | http://localhost:8080 | アノテーション |
| ClearML WebUI | `clearml_webserver` | http://localhost:8082 | 実験管理 |
| ClearML API | `clearml_apiserver` | http://localhost:8008 | 内部API |
| ClearML FileServer | `clearml_fileserver` | http://localhost:8081 | ファイル管理 |
| Streamlit | `streamlit_app` | http://localhost:8501 | 統合UI（メイン） |
| FiftyOne | `streamlit_app` | http://localhost:5151 | 推論結果可視化 |

---

## セットアップ手順

### Step 1 — nvidia-container-toolkit のインストール

```bash
# GPGキーとリポジトリ追加
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Dockerデーモンに設定を反映
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Step 2 — docker グループへの追加

```bash
sudo usermod -aG docker $USER
newgrp docker

# 動作確認（コンテナ内からGPUが見えればOK）
docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu22.04 nvidia-smi
```

### Step 3 — リポジトリのクローン

```bash
git clone git@github.com:ryotaema/mlops_workspace.git
cd mlops_workspace
```

### Step 4 — .env ファイルの作成

```bash
cat > .env << 'EOF'
CVAT_USERNAME=admin
CVAT_PASSWORD=your_password_here
CVAT_DB_PASSWORD=your_db_password_here
CVAT_IAM_DB_PASSWORD=your_iam_db_password_here
CLEARML_ACCESS_KEY=
CLEARML_SECRET_KEY=
NVIDIA_VISIBLE_DEVICES=all
EOF
```

> `.env` はGitの追跡対象外です。各パスワードは必ず強い値に変更してください。

### Step 5 — CVATデータベースの初期化

```bash
# DB系コンテナを先に起動（初回のみ）
docker compose up -d cvat_db cvat_redis cvat_iam_db clearml_mongo clearml_elastic clearml_redis
sleep 30

# マイグレーション実行（初回のみ）
docker compose run --rm cvat_server bash -c "python manage.py migrate"

# スーパーユーザー作成（初回のみ）
docker compose run --rm cvat_server bash -c \
  "python manage.py createsuperuser --username admin --email admin@local.com"
# パスワード入力を求められるので .env に設定したパスワードを入力
```

### Step 6 — 全サービス起動

```bash
docker compose up -d

# 状態確認（全コンテナが Up になればOK）
sleep 30 && docker compose ps
```

### Step 7 — 疎通確認

```bash
# CVAT
curl -s http://localhost:8080/api/server/about | python3 -m json.tool | head -3

# ClearML
curl -s http://localhost:8008/debug.ping

# Streamlit
curl -s -o /dev/null -w "%{http_code}" http://localhost:8501
```

---

## 日常的な起動・停止

```bash
cd mlops_workspace

# 起動
docker compose up -d

# 停止
docker compose down

# ログ確認
docker compose logs -f streamlit_app
docker compose logs -f cvat_server
```

---

## 運用フロー

```
① CVATでアノテーション
   http://localhost:8080 → admin / 設定したパスワード でログイン
   プロジェクト作成 → 画像アップロード → バウンディングボックスを付ける

② Streamlit タブ① でエクスポート
   http://localhost:8501 を開く
   「タスク一覧を取得」→ 対象タスクを選択 →「YOLOフォーマットでエクスポート」
   → data/ ディレクトリに自動展開される

③ Streamlit タブ② でYOLO学習
   モデルサイズ（n/s/m/l/x）・エポック数・バッチサイズを設定
   「学習開始」→ バックグラウンドで実行
   → エポックごとに mAP 等のメトリクスがログに表示される
   → ClearMLに自動でメトリクスが記録される（http://localhost:8082）

④ Streamlit タブ③ で推論・可視化
   学習済みモデルを選択 →「推論実行」
   → predictions/ にJSONで結果保存
   「FiftyOneで可視化」→ http://localhost:5151 でブラウザ確認
```

---

## Dockerfile の CUDA バージョン設定

`app/Dockerfile` のベースイメージと PyTorch インストール URL は、  
**お使いの CUDA バージョンに合わせて変更**してください。

```dockerfile
# ベースイメージの例（CUDA 12.6）
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

# PyTorch インストール（CUDA 12.6 対応ビルド）
RUN pip install --no-cache-dir torch torchvision torchaudio \
      --index-url https://download.pytorch.org/whl/cu126
```

お使いの CUDA バージョンに対応する設定値：

| CUDA | `--index-url` の末尾 | ベースイメージタグ例 |
|---|---|---|
| 11.8 | `cu118` | `cuda:11.8.0-cudnn8-runtime-ubuntu22.04` |
| 12.1 | `cu121` | `cuda:12.1.1-cudnn8-runtime-ubuntu22.04` |
| 12.4 | `cu124` | `cuda:12.4.1-cudnn-runtime-ubuntu22.04` |
| 12.6 | `cu126` | `cuda:12.6.3-cudnn-runtime-ubuntu22.04` |

利用可能なイメージ一覧: https://hub.docker.com/r/nvidia/cuda/tags  
PyTorch 対応ビルド一覧: https://pytorch.org/get-started/locally/

---

## トラブルシューティング

### `docker` コマンドで permission denied が出る

```bash
sudo usermod -aG docker $USER
newgrp docker   # ログアウト不要でグループを即時反映
```

### コンテナ名が競合してエラーになる

他の Docker プロジェクトが同じポートやコンテナ名を使っている場合、先に停止してください。

```bash
# 競合しているコンテナを確認
docker ps --format "table {{.Names}}\t{{.Ports}}" | grep 8080

# 該当プロジェクトを停止してから再起動
docker compose down && docker compose up -d
```

### Elasticsearch が起動しない（OOM エラー）

```bash
echo "vm.max_map_count=262144" | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
docker compose restart clearml_elastic
```

### GPU 非搭載環境で動かしたい（CPU のみ）

`docker-compose.yml` の `streamlit_app` から以下のブロックを削除してください：

```yaml
# 以下を削除
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

---

## ClearML API キーの取得（任意）

ClearML に認証ありでアクセスしたい場合：

1. http://localhost:8082 を開く
2. Settings → Workspace → **Create new credentials**
3. 表示された Access Key / Secret Key を `.env` に記入

```bash
# .env に追記
CLEARML_ACCESS_KEY=your_access_key
CLEARML_SECRET_KEY=your_secret_key

# Streamlit コンテナを再起動して反映
docker compose restart streamlit_app
```

---

## 備考

- `version: "3.8"` の警告（`the attribute version is obsolete`）は Docker Compose v2 系の仕様変更によるもので、動作には影響ありません。
- 本システムは完全ローカル動作のため、外部ネットワークへのデータ転送は発生しません。
- `data/`・`models/`・`predictions/` はホスト側の bind mount のため、コンテナを削除してもデータは保持されます。
- CVAT: `v2.64.0` / cvat-sdk: `2.64.0`（サーバーと SDK を同一バージョンに固定）
