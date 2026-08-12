# Docker 環境の準備

このシステムは Docker 上で動きます。**Docker が入っていない状態から始める人向け**の手引きです。

すでに `docker compose version` が動く人は、[4. NVIDIA GPU を使えるようにする](#4-nvidia-gpu-を使えるようにする)
だけ確認して、[README のセットアップ手順](../README.md#セットアップ手順)へ進んでください。

- 想定 OS は **Ubuntu 22.04 LTS**（24.04 でも同じ手順で入ります）
- Windows / macOS は [別の OS の場合](#別の-os-の場合)を参照
- 途中で失敗したら [うまくいかないとき](#うまくいかないとき)へ

---

## いま何が入っているか調べる

まずこれを実行してください。結果によって、どこから始めるかが決まります。

```bash
docker --version          # Docker Engine
docker compose version    # Compose v2（"docker-compose" ではなくスペース区切り）
nvidia-smi                # GPU ドライバ
```

| 結果 | 進むところ |
| --- | --- |
| `docker: command not found` | 1. から |
| `docker` はあるが `docker compose version` が失敗 | 1. から（古い Docker の可能性が高い） |
| どちらも動く | 4. から |
| `nvidia-smi` が動かない | 0. から（GPU を使う場合） |

---

## 0. NVIDIA ドライバ（GPU を使う場合のみ）

`nvidia-smi` が動けば飛ばしてください。GPU が無い PC でも動きますが、学習は非常に遅くなります。

```bash
ubuntu-drivers devices          # 推奨ドライバを確認
sudo ubuntu-drivers autoinstall # 推奨版を入れる
sudo reboot
```

再起動後、`nvidia-smi` で GPU 名とドライバのバージョンが表示されれば成功です。
**ドライバは 520 以上**が必要です（表右上の `Driver Version` で確認）。

---

## 1. 古い Docker を取り除く

Ubuntu の標準リポジトリに入っている `docker.io` などは、Compose v2 が付いてこないことがあります。
公式版と混ざると原因の切り分けが難しくなるので、先に消します。

```bash
for p in docker.io docker-doc docker-compose docker-compose-v2 podman-docker containerd runc; do
  sudo apt-get remove -y $p
done
```

> 「そんなパッケージは無い」と言われても問題ありません。入っていないだけです。
> **作成済みのイメージやコンテナは消えません**（`/var/lib/docker` は残ります）。

---

## 2. Docker の公式リポジトリを登録する

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl

sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
```

---

## 3. Docker 本体を入れる

```bash
sudo apt-get install -y \
  docker-ce docker-ce-cli containerd.io \
  docker-buildx-plugin docker-compose-plugin
```

確認します。

```bash
docker --version         # 例: Docker version 29.1.3
docker compose version   # 例: Docker Compose version v5.1.0
sudo docker run --rm hello-world
```

`hello-world` が「Hello from Docker!」を表示すれば成功です。

### sudo なしで使えるようにする

このリポジトリの手順は `sudo` を付けずに `docker` を実行します。

```bash
sudo usermod -aG docker $USER
newgrp docker      # いま開いているターミナルに反映（新しく開き直しても可）

docker run --rm hello-world   # sudo 無しで通ることを確認
```

> `newgrp docker` は**そのターミナルだけ**の反映です。
> 他のターミナルや GUI アプリからも使うなら、**一度ログアウトして入り直して**ください。

---

## 4. NVIDIA GPU を使えるようにする

GPU を使わない場合は飛ばして構いません（[GPU なしで動かす](#gpu-なしで動かす)を参照）。

> ### ⚠ 先に `nvidia-smi` が動くことを確認してください
>
> ```bash
> nvidia-smi        # GPU 名とドライバのバージョンが出ること
> ```
>
> **これが動かないまま 4-2 を実行しないでください。**
> ドライバが無いのに Docker の既定を `nvidia` にすると、
> `failed to initialize NVML: ERROR_LIBRARY_NOT_FOUND` で
> GPU を使うコンテナが起動できなくなります（元に戻す手順は
> [トラブルシューティング](#failed-to-initialize-nvml-error_library_not_found)）。
>
> 動かない場合は **[0. NVIDIA ドライバ](#0-nvidia-ドライバgpu-を使う場合のみ)** に戻るか、
> GPU を使わないなら 4 をまるごと飛ばしてください。

### 4-1. NVIDIA Container Toolkit を入れる

コンテナから GPU を見えるようにする部品です。

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
```

### 4-2. 既定のランタイムを nvidia にする

**この設定が要ります。** CVAT の自動アノテーション（Nuclio）は、
コンテナを起動するときに GPU の指定を渡せない作りになっているため、
**Docker 側の既定が `nvidia` でないと GPU で動きません**。

```bash
sudo nvidia-ctk runtime configure --runtime=docker --set-as-default
sudo systemctl restart docker
```

`/etc/docker/daemon.json` が次のようになります。

```json
{
    "default-runtime": "nvidia",
    "runtimes": {
        "nvidia": {
            "args": [],
            "path": "nvidia-container-runtime"
        }
    }
}
```

> **Docker を再起動すると、動いているコンテナが全部止まります。**
> 他の作業で Docker を使っている場合は、区切りの良いところで実行してください。

### 4-3. 確認する

```bash
docker info | grep -i "default runtime"      # Default Runtime: nvidia
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi
```

最後のコマンドで GPU の情報が表示されれば完了です。

---

## GPU なしで動かす

GPU が無くても動きます。ただし**学習は数十倍遅くなります**（アノテーションや推論結果の確認は問題なく行えます）。
学習は GPU のある PC で行い、できたモデルを「📁 データ管理」タブから取り込む使い方もできます。

**`docker-compose.cpu.yml` を重ねて起動してください。** ファイルを書き換えないので、
あとから `git pull` しても衝突しません。

```bash
docker compose -f docker-compose.yml -f docker-compose.cpu.yml up -d
```

毎回 2 つ指定するのが面倒なら、環境変数にしておけば `docker compose up -d` だけで済みます。

```bash
echo 'COMPOSE_FILE=docker-compose.yml:docker-compose.cpu.yml' >> .env
```

4 の手順（NVIDIA Container Toolkit）はまるごと不要です。

> Docker Compose が 2.24 より古い場合は `!reset` が使えません。
> その場合は `docker-compose.yml` の `streamlit_app` から
> `deploy:` のブロック（`driver: nvidia` を含む 6 行）を手で削除してください。

---

## CUDA のバージョンを変えるとき

既定は **CUDA 12.8 / cu128** です（RTX 50 系 = Blackwell 対応）。
お使いの GPU・ドライバに合わせて変える場合は、`app/Dockerfile` の
**ベースイメージと PyTorch のインストール先をセットで**変更してください。

```dockerfile
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

RUN pip install --no-cache-dir torch torchvision torchaudio \
      --index-url https://download.pytorch.org/whl/cu128
```

| CUDA | `--index-url` の末尾 | ベースイメージのタグ |
| --- | --- | --- |
| 11.8 | `cu118` | `cuda:11.8.0-cudnn8-runtime-ubuntu22.04` |
| 12.1 | `cu121` | `cuda:12.1.1-cudnn8-runtime-ubuntu22.04` |
| 12.4 | `cu124` | `cuda:12.4.1-cudnn-runtime-ubuntu22.04` |
| 12.6 | `cu126` | `cuda:12.6.3-cudnn-runtime-ubuntu22.04` |
| 12.8 | `cu128` | `cuda:12.8.1-cudnn-runtime-ubuntu22.04` |

> **RTX 50 系（Blackwell / sm_120）は CUDA 12.8 以上が必要です。**
> cu126 では `no kernel image is available for execution on the device` になります。

変更したら再ビルドが必要です。

```bash
docker compose build streamlit_app && docker compose up -d streamlit_app
```

- 利用できるイメージ: https://hub.docker.com/r/nvidia/cuda/tags
- PyTorch の対応ビルド: https://pytorch.org/get-started/locally/

---

## うまくいかないとき

### `permission denied while trying to connect to the Docker daemon socket`

`docker` グループへの追加が反映されていません。

```bash
groups | grep docker     # docker が出るか
```

出ない場合は `sudo usermod -aG docker $USER` を実行し、**ログアウトして入り直して**ください。

### `docker compose` が「is not a docker command」になる

Compose v2 が入っていません。1〜3 をやり直してください。
`docker-compose`（ハイフン）は古い v1 です。このリポジトリは **`docker compose`（スペース）** を使います。

### `Cannot connect to the Docker daemon`

Docker が起動していません。

```bash
sudo systemctl status docker      # 状態を見る
sudo systemctl enable --now docker  # 起動＋自動起動を有効に
```

### `could not select device driver "" with capabilities: [[gpu]]`

NVIDIA Container Toolkit が入っていないか、Docker の再起動がまだです。
4-1 → 4-2 をやり直してください。

### `failed to initialize NVML: ERROR_LIBRARY_NOT_FOUND`

```
Error response from daemon: failed to create task for container: ...
failed to initialize NVML: ERROR_LIBRARY_NOT_FOUND
```

**NVIDIA Container Toolkit は入っているのに、GPU ドライバ側が見つからない状態**です。
CVAT などは起動して、`streamlit_app` だけが落ちます（GPU を要求しているのがそれだけのため）。

まず、どちらの状況か確かめてください。

```bash
nvidia-smi                    # ① ドライバが動くか
lspci | grep -i nvidia        # ② そもそも GPU が載っているか
```

| 状況 | 対処 |
|---|---|
| ② に何も出ない（GPU 非搭載） | **A. GPU を使わない設定に戻す**（下記） |
| ② は出るが ① が動かない | **B. ドライバを入れる** → [0. NVIDIA ドライバ](#0-nvidia-ドライバgpu-を使う場合のみ) |
| ① が正常に表示される | **C. ツールキットの設定を作り直す**（下記） |

#### A. GPU を使わない設定に戻す

**ドライバが無いのに `default-runtime: nvidia` を設定してしまった場合**は、
先にそれを戻してください（`docker info | grep -i "default runtime"` で確認できます）。
そのままだと GPU を使うコンテナが起動できません。

```bash
sudo nvidia-ctk runtime configure --runtime=docker --set-as-default=false
sudo systemctl restart docker
```

このオプションが使えない版では `/etc/docker/daemon.json` を開き、
`"default-runtime": "nvidia",` の 1 行を消して `sudo systemctl restart docker` してください。

そのうえで [GPU なしで動かす](#gpu-なしで動かす)の方法で起動します。

#### C. ツールキットの設定を作り直す

ドライバ（`nvidia-smi`）が正常なのに出る場合は、設定を作り直して Docker を再起動します。

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

カーネルを更新した後にドライバのモジュールが読み込まれていないこともあります
（`nvidia-smi` が `couldn't communicate with the NVIDIA driver` を返す場合）。
その場合は**再起動**すると直ることが多いです。

**GPU を使う予定が無いなら、ドライバを入れる必要はありません。**
[GPU なしで動かす](#gpu-なしで動かす)の 1 行で起動できます。

### `nvidia-smi` はホストで動くのにコンテナで動かない

ドライバとツールキットのバージョンが噛み合っていない場合があります。

```bash
nvidia-smi                                   # ホスト側のドライバ
dpkg -l | grep nvidia-container-toolkit      # ツールキットのバージョン
```

ツールキットは **1.14 以上**が必要です。古ければ `sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit` で更新してください。

### ディスクが足りない

初回は Docker イメージのダウンロードで **30〜50 GB** 使います。

```bash
df -h /var/lib/docker     # 空き容量を確認
docker system df          # Docker が使っている量
```

不要なイメージの削除は `docker image prune -a` で行えますが、
**このリポジトリ以外で使っているイメージも消えます**。中身を確認してから実行してください。

---

## 別の OS の場合

### Windows

**WSL2 + Docker Desktop** を使います。

1. PowerShell（管理者）で `wsl --install` を実行 → 再起動
2. [Docker Desktop](https://www.docker.com/products/docker-desktop/) を入れる
3. Docker Desktop の設定 → *Resources* → *WSL Integration* で、使う WSL ディストリビューションを有効にする
4. GPU を使う場合は、**Windows 側**に NVIDIA ドライバを入れる（WSL 内には入れない）
5. 以降の作業は WSL のターミナル（Ubuntu）で行う

**リポジトリは WSL 側（`/home/<ユーザー名>/` 以下）に置いてください。**
Windows 側（`/mnt/c/...`）に置くとファイルの読み書きが極端に遅くなります。

> 自動アノテーション（Nuclio）の GPU 実行は、Docker Desktop では
> `default-runtime` を変更できないため利用できません。CPU での実行は可能です。

### macOS

[Docker Desktop](https://www.docker.com/products/docker-desktop/) を入れれば動きますが、
**NVIDIA GPU は使えません**（学習は CPU のみ）。
`docker-compose.yml` の GPU 指定を外す必要があります（[GPU なしで動かす](#gpu-なしで動かす)）。

Apple Silicon では一部のイメージが `linux/amd64` のエミュレーションになり、さらに遅くなります。
本格的に学習する用途には向きません。

---

## ここまで終わったら

```bash
docker --version
docker compose version
docker run --rm hello-world
docker info | grep -i "default runtime"    # GPU を使う場合は nvidia
```

これらが通れば準備完了です。[README のセットアップ手順](../README.md#セットアップ手順)へ進んでください。
