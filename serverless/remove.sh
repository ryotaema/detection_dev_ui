#!/usr/bin/env bash
# =============================================================================
# デプロイ済みの自作 YOLO 関数を削除する
#   全削除:  ./serverless/remove.sh
#   個別:    ./serverless/remove.sh custom-yolo11s-bellpepper
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
BIN_DIR="$SCRIPT_DIR/bin"
NUCTL="$(command -v nuctl || echo "$BIN_DIR/nuctl")"

if [ ! -x "$NUCTL" ] && ! command -v nuctl >/dev/null 2>&1; then
  echo "nuctl が見つかりません。先に serverless/deploy.sh を実行してください。" >&2
  exit 1
fi

if [ "$#" -gt 0 ]; then
  TARGETS=("$@")
else
  # custom/ 配下の function.yaml から metadata.name を収集
  TARGETS=()
  while IFS= read -r y; do
    n="$(grep -E '^\s*name:' "$y" | head -1 | sed -E 's/.*name:\s*//')"
    [ -n "$n" ] && TARGETS+=("$n")
  done < <(find "$SCRIPT_DIR/custom" -maxdepth 2 -name function.yaml | sort)
fi

for name in "${TARGETS[@]}"; do
  echo "==> delete function: $name"
  "$NUCTL" delete function "$name" --platform local 2>/dev/null \
    || echo "   (存在しないか既に削除済み)"
done

echo "==> 残存関数:"
"$NUCTL" get function --platform local || true
