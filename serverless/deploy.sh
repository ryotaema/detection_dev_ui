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
  [ -f "$fn_dir/model.env" ] && source "$fn_dir/model.env"

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
