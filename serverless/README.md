# CVAT 自動アノテーション (Nuclio serverless) — 自作 YOLO モデル

学習済みの自作 YOLO モデル (`models/<run>/weights/best.pt`) を CVAT の
**Automatic annotation** から使えるようにするための一式。

CVAT は [Nuclio](https://nuclio.io/) というサーバレス基盤経由でモデルを呼び出す。
各モデルを「Nuclio 関数」としてデプロイすると、CVAT のタスク/ジョブ画面の
**Actions → Automatic annotation** に選択肢として現れる。

```
CVAT UI (Actions → Automatic annotation)
   │  base64 画像 + threshold
   ▼
nuclio ダッシュボード (:8070)  ── cvat_net ──  関数コンテナ (best.pt 内蔵)
                                                   main.py:handler → 検出結果 JSON
```

## ディレクトリ構成

```
serverless/
├── deploy.sh              # 関数のビルド & デプロイ (nuctl 自動導入)
├── remove.sh              # 関数の削除
├── _common/              # 全関数で共有するハンドラ
│   ├── main.py           #   Nuclio エントリポイント
│   └── model_handler.py  #   Ultralytics 推論ラッパ
└── custom/                # 関数定義（各自のモデルごとに 1 ディレクトリ）
    └── <関数名>/
        ├── function.yaml       # CPU 用定義 (ラベル spec を含む)
        ├── function-gpu.yaml   # GPU 用定義 (cu128 / Blackwell 対応)
        └── model.env           # MODEL_WEIGHTS=<models/ からの相対パス>
```

`custom/` の中身は **リポジトリに含まれない**（`.gitignore` 済み）。
どのモデルを置くかは環境ごとに違うため、UI の
「🏷 Step1: アノテーション」タブから生成するか、下記「新しいモデルを追加する」の
手順で自分の環境に合わせて作る。

`best.pt` はリポジトリに含めない。`deploy.sh` が `model.env` の `MODEL_RUN` を読み、
`models/<MODEL_RUN>/weights/best.pt` をビルド時にコピーする。

## 使い方

### 1. Nuclio 基盤 + CVAT serverless 連携を起動

```bash
docker compose \
  -f docker-compose.yml \
  -f docker-compose.serverless.yml \
  up -d nuclio cvat_server cvat_worker_annotation
```

`docker-compose.serverless.yml` が行うこと:
- `nuclio` ダッシュボード (`quay.io/nuclio/dashboard:1.15.9`) を `cvat_net` に追加
- `cvat_server` / `cvat_worker_annotation` に `CVAT_SERVERLESS=1` を付与

### 2. 関数をデプロイ

```bash
# GPU (既定・下記「GPU で動かす」の前提を満たしていること)
./serverless/deploy.sh

# 特定の関数だけ
./serverless/deploy.sh <関数名>

# CPU (daemon 変更不要で確実に動く)
./serverless/deploy.sh --cpu
```

- 初回は `nuctl` (v1.15.9) を `serverless/bin/` に自動ダウンロードする
- 関数コンテナは `${COMPOSE_PROJECT_NAME}_cvat_net` に接続される
  （既定は `.env` の `COMPOSE_PROJECT_NAME` から解決。`CVAT_NETWORK` で上書き可）

### 3. CVAT で使う

1. CVAT (http://localhost:8080) でタスク/ジョブを開く
2. **Actions → Automatic annotation**
3. Model に自作モデル（`function.yaml` の `metadata.annotations.name` の表示名）を選択
4. モデルのラベルと CVAT タスクのラベルを対応付け → **Annotate**

Nuclio 側の状態は http://localhost:8070 でも確認できる。

### 4. 削除

```bash
./serverless/remove.sh              # 全関数削除
./serverless/remove.sh custom-<関数名>
```

## GPU で動かす（既定）

`deploy.sh` の既定は GPU。CVAT の GPU serverless は、Docker daemon の
**default-runtime を `nvidia`** にしていないと関数コンテナに GPU が渡らない。

`/etc/docker/daemon.json` を編集:

```json
{
    "default-runtime": "nvidia",
    "runtimes": {
        "nvidia": { "path": "nvidia-container-runtime", "args": [] }
    }
}
```

反映には Docker デーモンの再起動が必要（**稼働中の全コンテナが再起動する**点に注意）:

```bash
sudo systemctl restart docker
docker compose -f docker-compose.yml -f docker-compose.serverless.yml \
  up -d nuclio cvat_server cvat_worker_annotation
./serverless/deploy.sh --gpu
```

> RTX 50 系 (Blackwell / sm_120) は CUDA 12.8 + cu128 が必須。
> `function-gpu.yaml` は `nvidia/cuda:12.8.1` ベース + cu128 で構成済み。

## 新しいモデルを追加する

1. `serverless/custom/<name>/` を作成
2. `function.yaml` / `function-gpu.yaml` を用意
   - `metadata.name` は一意・小文字ハイフン
   - `annotations.spec` のラベル名は **モデルのクラス名 = CVAT タスクのラベル名** に合わせる
   - マルチクラスなら `spec` に全クラスを列挙（`id` は学習時のクラス index）
3. `model.env` に `MODEL_WEIGHTS=<models/ からの相対パス>` を記載
   （`MODEL_RUN=<run 名>` だけでも `models/<run>/weights/best.pt` として解決される）
4. `./serverless/deploy.sh <name>`

UI から行う場合は、この 1〜4 を「🏷 Step1: アノテーション」タブが自動で行う。
モデルのクラス名から `annotations.spec` を生成するため、ラベル名のずれも起きない。

## SAM 3 を使う（学習不要のゼロショット）

自作モデルがまだ無い段階でも、[SAM 3](https://github.com/facebookresearch/sam3) を
デプロイすればアノテーションの下書きを作れる。使い方は 2 通りで、
CVAT 側の出方が違うため関数を分けている。

| variant | CVAT での場所 | できること |
|---|---|---|
| `concept` | Actions → Automatic annotation | 英語の名詞句（例 `red fruit`）に当てはまるものを**全部**ポリゴンにする |
| `interactive` | AI Tools → Interactors | ボックスで囲む／点を打った **1 個だけ**をマスクにする |

### 1. 重みを置く

重みは配布物に含まれない（Meta の SAM License に従うため）。

1. https://huggingface.co/facebook/sam3 でアクセス承認を受ける
2. `sam3.pt`（約 3.45GB）をダウンロード
3. `models/.sam3/sam3.pt` に置く

### 2. デプロイ

「🏷 Step1: アノテーション」タブの「🧩 SAM 3 を使う」から生成〜デプロイまで行える。
CLI で行う場合は `serverless/custom/<name>/model.env` に以下を書き、
`function.yaml` / `function-gpu.yaml` を用意して `./serverless/deploy.sh <name>`:

```sh
MODEL_KIND=sam3               # これがあると deploy.sh が SAM 3 の経路を通る
SAM3_VARIANT=concept          # concept か interactive
MODEL_WEIGHTS=.sam3/sam3.pt   # models/ からの相対パス
```

### 仕組み上の注意

- **重みはイメージに焼き込まず、ホストの `models/.sam3/` をマウントする。**
  3.45GB をビルドのたびにコピーすると時間もディスクも食うため。
  `function.yaml` の `__SAM3_WEIGHTS_HOST_DIR__` は `deploy.sh` が
  デプロイ時にホストの実パスへ置換する（コンテナ内から実行された場合は
  自分のマウント表を引いて読み替える。`SAM3_WEIGHTS_HOST_DIR` で明示も可）
- **重みを差し替えたら関数の再起動が必要。** マウントなので再ビルドは要らないが、
  関数プロセスは起動時に読んだモデルを持ち続ける（UI は差分を見て再デプロイを促す）
- **CVAT のラベル名と SAM 3 に渡す語は別に持つ。** SAM 3 は英語の短い名詞句を
  前提にしているので、CVAT 側のラベル名（日本語でも可）とは分けて
  `SAM3_PROMPTS` 環境変数に対応表を持たせている
- **GPU メモリを数 GB 常時占有する。** `concept` と `interactive` を両方デプロイすると
  そのぶん増える。学習と同時に使うなら片方だけにするか、`SAM3_HALF=1`（FP16）にする
- 起動には重みの読み込みぶんの時間がかかるため、`readinessTimeoutSeconds` を延ばしてある

## トラブルシューティング

| 症状 | 対処 |
|---|---|
| CVAT に関数が出ない | `curl -s localhost:8070/api/functions` で関数を確認。`cvat_server` を再起動 |
| 関数が `error` 状態 | `docker logs nuclio-custom-...` でハンドラのログ確認 |
| GPU が使われない | daemon の `default-runtime=nvidia` になっているか確認 |
| ネットワーク不一致 | `CVAT_NETWORK=<実ネットワーク名> ./serverless/deploy.sh` で明示 |
| ラベルが投入されない | `function.yaml` の spec ラベル名と CVAT タスクのラベル名が一致しているか |
| SAM 3 が `SKIP` される | `models/.sam3/sam3.pt` があるか。無ければ HuggingFace から取得する |
| SAM 3 が起動しない | 重みのマウント元がホストの実パスか（`docker inspect nuclio-sam3-... --format '{{json .Mounts}}'`） |
| SAM 3 が何も検出しない | プロンプトが英語の短い名詞句になっているか。CVAT 側の threshold を下げてみる |
| SAM 3 のクリックが反応しない | マスクがスコアで捨てられている。`SAM3_INTERACTIVE_CONF`（既定 0.05）を下げる |
