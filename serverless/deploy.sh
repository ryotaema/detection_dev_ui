#!/usr/bin/env bash
# =============================================================================
# 自作 YOLO モデルを CVAT の Nuclio serverless 関数としてデプロイする
# -----------------------------------------------------------------------------
#   GPU (既定):  ./serverless/deploy.sh
#   CPU:         ./serverless/deploy.sh --cpu
#   個別:        ./serverless/deploy.sh <関数名>
#
# 前提:
#   - nuclio ダッシュボードが起動していること
#       docker compose -f docker-compose.yml -f docker-compose.serverless.yml \
#         up -d nuclio cvat_server cvat_worker_annotation
#   - GPU 版は Docker daemon の default-runtime=nvidia が必要 (README 参照)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." >/dev/null 2>&1 && pwd)"
NUCTL_VERSION="1.15.9"        # nuclio ダッシュボードと同一バージョンに固定
BIN_DIR="$SCRIPT_DIR/bin"
BUILD_DIR="$SCRIPT_DIR/.build"
COMMON_DIR="$SCRIPT_DIR/_common"
CUSTOM_DIR="$SCRIPT_DIR/custom"
SAM3_DIR="$SCRIPT_DIR/sam3"

# --- 引数解析 -------------------------------------------------------------
# 既定は GPU。CPU で動かす場合は --cpu を付ける
# (GPU 版は Docker daemon の default-runtime=nvidia が必要。README 参照)
USE_GPU=1
TARGETS=()
for arg in "$@"; do
  case "$arg" in
    --gpu) USE_GPU=1 ;;
    --cpu) USE_GPU=0 ;;
    -*)    echo "不明なオプション: $arg" >&2; exit 1 ;;
    *)     TARGETS+=("$arg") ;;
  esac
done

# --- 接続先ネットワーク名 (COMPOSE_PROJECT_NAME + _cvat_net) -----------------
PROJECT_NAME="$(grep -E '^COMPOSE_PROJECT_NAME=' "$PROJECT_DIR/.env" 2>/dev/null | cut -d= -f2 || true)"
if [ -z "${PROJECT_NAME:-}" ]; then
  PROJECT_NAME="$(basename "$PROJECT_DIR" | tr '[:upper:]' '[:lower:]' | tr -cd 'a-z0-9_-')"
fi
CVAT_NETWORK="${CVAT_NETWORK:-${PROJECT_NAME}_cvat_net}"

if ! docker network inspect "$CVAT_NETWORK" >/dev/null 2>&1; then
  echo "ERROR: Docker ネットワーク '$CVAT_NETWORK' が見つかりません。" >&2
  echo "       CVAT スタックが起動しているか、CVAT_NETWORK 環境変数で上書きしてください。" >&2
  exit 1
fi
echo "==> 関数を接続するネットワーク: $CVAT_NETWORK"

# --- nuctl 自動導入 --------------------------------------------------------
NUCTL="$(command -v nuctl || true)"
if [ -z "$NUCTL" ]; then
  NUCTL="$BIN_DIR/nuctl"
  if [ ! -x "$NUCTL" ]; then
    echo "==> nuctl $NUCTL_VERSION を取得します..."
    mkdir -p "$BIN_DIR"
    url="https://github.com/nuclio/nuclio/releases/download/${NUCTL_VERSION}/nuctl-${NUCTL_VERSION}-linux-amd64"
    curl -fSL "$url" -o "$NUCTL"
    chmod +x "$NUCTL"
  fi
fi
echo "==> nuctl: $NUCTL ($("$NUCTL" version 2>/dev/null | head -1 || echo '?'))"

# --- nuclio プロジェクト作成 (存在すれば無視) --------------------------------
"$NUCTL" create project cvat --platform local 2>/dev/null || true

# --- コンテナ内のパス → ホストのパス ----------------------------------------
# SAM 3 の重み (3.45GB) はイメージに焼き込まず、ホストのディレクトリを
# 関数コンテナへマウントして使う。マウント元はホストの実パスでなければならないが、
# このスクリプトは streamlit_app コンテナの中から実行されることがあるため、
# そのままでは自分から見えているパス (/workspace/...) を渡してしまう。
# 自分自身のマウント表を docker inspect で引いて読み替える
# （app/core/hostpath.py と同じ考え方。決め打ちにすると別環境で外れる）。
resolve_host_dir() {
  local cpath="$1"
  if [ -n "${SAM3_WEIGHTS_HOST_DIR:-}" ]; then echo "$SAM3_WEIGHTS_HOST_DIR"; return 0; fi
  if [ ! -f /.dockerenv ]; then echo "$cpath"; return 0; fi

  python3 - "$cpath" <<'PYEOF'
import json, os, socket, subprocess, sys

cpath = os.path.normpath(sys.argv[1])
mounts = []
for ident in (socket.gethostname(), "streamlit_app"):
    try:
        r = subprocess.run(["docker", "inspect", ident, "--format", "{{json .Mounts}}"],
                           capture_output=True, text=True, timeout=10)
        if r.returncode == 0 and r.stdout.strip():
            for m in json.loads(r.stdout) or []:
                if m.get("Destination") and m.get("Source"):
                    mounts.append((os.path.normpath(m["Destination"]), m["Source"]))
            break
    except Exception:
        pass

# 長いマウント先から先に当てる (/workspace より /workspace/models を優先)
for dst, src in sorted(mounts, key=lambda x: -len(x[0])):
    if cpath == dst:
        print(src)
        sys.exit(0)
    if cpath.startswith(dst.rstrip("/") + "/"):
        print(os.path.join(src, os.path.relpath(cpath, dst)))
        sys.exit(0)
sys.exit(1)
PYEOF
}

# --- SAM 3 のデプロイ ------------------------------------------------------
# 自作モデルと違い、重みはイメージに焼き込まずホストからマウントする
# (sam3.pt は 3.45GB あるため)。ハンドラも専用のものを使う。
deploy_sam3() {
  local name="$1" fn_dir="$2" src_yaml="$3"
  local variant="${SAM3_VARIANT:-concept}"
  local handler="$SAM3_DIR/main_${variant}.py"

  if [ ! -f "$handler" ]; then
    echo "SKIP: $name (未知の SAM3_VARIANT: $variant / concept か interactive)"
    return 1
  fi

  local rel="${MODEL_WEIGHTS:-.sam3/sam3.pt}"
  local abs="$PROJECT_DIR/models/$rel"
  if [ ! -f "$abs" ]; then
    echo "SKIP: $name (重みがありません: models/$rel)"
    echo "      https://huggingface.co/facebook/sam3 でアクセス承認を受けたうえで"
    echo "      sam3.pt を models/$(dirname "$rel")/ に置いてください。"
    return 1
  fi

  local host_dir
  if ! host_dir="$(resolve_host_dir "$(dirname "$abs")")"; then
    echo "SKIP: $name (重みのホスト側パスを解決できませんでした)"
    echo "      SAM3_WEIGHTS_HOST_DIR=<ホストの絶対パス> を指定して再実行してください。"
    return 1
  fi

  local stage="$BUILD_DIR/$name"
  rm -rf "$stage"; mkdir -p "$stage"
  # マウント元はホストの実パスでなければならないので、ここで初めて確定させる
  sed "s|__SAM3_WEIGHTS_HOST_DIR__|$host_dir|g" "$src_yaml" > "$stage/function.yaml"
  cp "$handler" "$stage/main.py"
  cp "$SAM3_DIR/model_handler.py" "$stage/"

  echo ""
  echo "=========================================================="
  echo "  Deploy: $name  (SAM 3 / $variant)"
  echo "    weights: $host_dir/$(basename "$rel")  ← マウント"
  echo "=========================================================="
  "$NUCTL" deploy --project-name cvat \
    --path "$stage" \
    --file "$stage/function.yaml" \
    --platform local \
    --platform-config "{\"attributes\": {\"network\": \"$CVAT_NETWORK\"}}" || return 1

  # 重みはマウントなので焼き込みとは違い、差し替えても再デプロイは要らない。
  # ただし関数プロセスは起動時に読んだモデルを持ち続けるので、
  # 差し替えたら**再起動**が要る。それに気づけるよう記録は残す
  # (照合の取り方は deploy.sh / Python 側で必ず揃えること)。
  local _sha _size
  _sha="$(head -c 8388608 "$abs" | sha1sum | cut -d" " -f1)"
  _size="$(stat -c %s "$abs")"
  cat > "$fn_dir/.deployed.json" <<EOF
{
  "kind": "sam3",
  "variant": "$variant",
  "weights": "$rel",
  "mounted": true,
  "host_dir": "$host_dir",
  "sha1": "$_sha",
  "size": $_size,
  "gpu": $([ "$USE_GPU" = "1" ] && echo true || echo false),
  "deployed_at": "$(date -Iseconds)"
}
EOF
  chmod 0666 "$fn_dir/.deployed.json" 2>/dev/null || true
}

# --- デプロイ対象の決定 ----------------------------------------------------
if [ "${#TARGETS[@]}" -eq 0 ]; then
  while IFS= read -r d; do TARGETS+=("$(basename "$d")"); done \
    < <(find "$CUSTOM_DIR" -mindepth 1 -maxdepth 1 -type d | sort)
fi

if [ "$USE_GPU" -eq 1 ]; then
  echo "==> モード: GPU (function-gpu.yaml)"
else
  echo "==> モード: CPU (function.yaml)"
fi

# --- 各関数をステージング & デプロイ ---------------------------------------
for name in "${TARGETS[@]}"; do
  fn_dir="$CUSTOM_DIR/$name"
  [ -d "$fn_dir" ] || { echo "SKIP: $name (ディレクトリなし)"; continue; }

  if [ "$USE_GPU" -eq 1 ]; then
    src_yaml="$fn_dir/function-gpu.yaml"
  else
    src_yaml="$fn_dir/function.yaml"
  fi
  [ -f "$src_yaml" ] || { echo "SKIP: $name ($(basename "$src_yaml") なし)"; continue; }

  # model.env から使用モデルを読む。
  #   MODEL_WEIGHTS … models/ からの相対パス（best.pt 以外の名前でも指せる）
  #   MODEL_RUN     … 旧形式。models/<run>/weights/best.pt を指す
  # 取り込んだモデルはファイル名が best.pt とは限らないため、
  # どちらも無ければ weights/ の中の .pt を 1 つだけ拾う。
  # shellcheck disable=SC1091
  MODEL_RUN=""
  MODEL_WEIGHTS=""
  MODEL_KIND=""
  SAM3_VARIANT=""
  [ -f "$fn_dir/model.env" ] && source "$fn_dir/model.env"

  # SAM 3 は重みの扱いもハンドラも違うので、ここで分岐する
  if [ "${MODEL_KIND:-}" = "sam3" ]; then
    deploy_sam3 "$name" "$fn_dir" "$src_yaml" || true
    continue
  fi

  best_pt=""
  if [ -n "$MODEL_WEIGHTS" ] && [ -f "$PROJECT_DIR/models/$MODEL_WEIGHTS" ]; then
    best_pt="$PROJECT_DIR/models/$MODEL_WEIGHTS"
  elif [ -n "$MODEL_RUN" ] && [ -f "$PROJECT_DIR/models/${MODEL_RUN}/weights/best.pt" ]; then
    best_pt="$PROJECT_DIR/models/${MODEL_RUN}/weights/best.pt"
  elif [ -n "$MODEL_RUN" ]; then
    # weights/ に .pt がちょうど 1 つならそれを使う（取り込んだモデル向け）
    only=""
    count=0
    for f in "$PROJECT_DIR/models/${MODEL_RUN}/weights/"*.pt; do
      [ -f "$f" ] || continue
      only="$f"; count=$((count + 1))
    done
    [ "$count" = "1" ] && best_pt="$only"
  fi

  if [ -z "$best_pt" ]; then
    echo "SKIP: $name (重みが見つかりません: MODEL_WEIGHTS=${MODEL_WEIGHTS:-未設定} MODEL_RUN=${MODEL_RUN:-未設定})"
    continue
  fi

  stage="$BUILD_DIR/$name"
  rm -rf "$stage"; mkdir -p "$stage"
  # ハンドラは常に function.yaml という名前で配置 (main.py が /opt/nuclio/function.yaml を読むため)
  cp "$src_yaml" "$stage/function.yaml"
  cp "$COMMON_DIR/main.py" "$COMMON_DIR/model_handler.py" "$stage/"
  cp "$best_pt" "$stage/best.pt"

  echo ""
  echo "=========================================================="
  echo "  Deploy: $name  (weights=${best_pt#"$PROJECT_DIR/models/"})"
  echo "=========================================================="
  if "$NUCTL" deploy --project-name cvat \
    --path "$stage" \
    --file "$stage/function.yaml" \
    --platform local \
    --platform-config "{\"attributes\": {\"network\": \"$CVAT_NETWORK\"}}"; then
    # どの重みで焼いたかを残す。
    # 重みはビルド時にコンテナへ焼き込まれるので、models/ 側を更新しても
    # 再デプロイするまで反映されない。**気づけないのが一番の問題**なので、
    # ここに記録して UI が「更新されています」と出せるようにする。
    # 照合は先頭 8MB のハッシュとサイズ。**Python 側と同じ取り方にすること**
    # （全体 sha1 と先頭 8MB sha1 は当然一致せず、常に「更新あり」になる）
    _sha="$(head -c 8388608 "$best_pt" | sha1sum | cut -d" " -f1)"
    _size="$(stat -c %s "$best_pt")"
    cat > "$fn_dir/.deployed.json" <<EOF
{
  "weights": "${best_pt#"$PROJECT_DIR/models/"}",
  "sha1": "$_sha",
  "size": $_size,
  "gpu": $([ "$USE_GPU" = "1" ] && echo true || echo false),
  "deployed_at": "$(date -Iseconds)"
}
EOF
    # コンテナ内 root で実行されることがあるため、ホストからも読み書きできるように
    chmod 0666 "$fn_dir/.deployed.json" 2>/dev/null || true
  fi
done

echo ""
echo "==> デプロイ済み関数一覧:"
"$NUCTL" get function --platform local

cat <<'EOF'

------------------------------------------------------------------
完了。CVAT のタスク/ジョブ画面 → 「Actions → Automatic annotation」
に自作モデルが表示されます (反映に十数秒かかる場合あり)。
関数の状態は http://localhost:8070 (Nuclio ダッシュボード) でも確認できます。
------------------------------------------------------------------
EOF
