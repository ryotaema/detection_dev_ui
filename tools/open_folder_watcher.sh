#!/usr/bin/env bash
# =============================================================================
# UI の「📂 フォルダを開く」を、OS のファイルアプリで開くための常駐スクリプト
#
#   Streamlit はコンテナの中で動いていて、画面も xdg-open も持たない。
#   そのためコンテナからは OS のファイルアプリを開けない。
#   このスクリプトを**ホスト側**で動かしておくと、
#   UI のボタンを押したときに実際にフォルダが開くようになる。
#
#   使い方（プロジェクトの直下で）:
#       ./tools/open_folder_watcher.sh
#
#   終わるときは Ctrl+C。
#   動かさなくても UI はホスト側のパスを表示するので、
#   コピーして自分で開くこともできる。
# =============================================================================
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REQ_DIR="$ROOT/predictions/.open_requests"
HEARTBEAT="$REQ_DIR/.watcher_alive"
INTERVAL="${OPEN_WATCHER_INTERVAL:-1}"

# 開くコマンドを決める（Linux / macOS / WSL）
if command -v xdg-open >/dev/null 2>&1; then
  OPENER="xdg-open"
elif command -v open >/dev/null 2>&1; then
  OPENER="open"                      # macOS
elif command -v explorer.exe >/dev/null 2>&1; then
  OPENER="explorer.exe"              # WSL
else
  echo "❌ フォルダを開くコマンドが見つかりません（xdg-open / open / explorer.exe）" >&2
  exit 1
fi

mkdir -p "$REQ_DIR"
echo "📂 フォルダを開く係を起動しました"
echo "   監視: $REQ_DIR"
echo "   使うコマンド: $OPENER"
echo "   終了するには Ctrl+C"

cleanup() {
  rm -f "$HEARTBEAT"
  echo ""
  echo "📂 終了しました"
  exit 0
}
trap cleanup INT TERM

while true; do
  # 生きていることを UI へ知らせる（UI はこのファイルの新しさを見る）
  touch "$HEARTBEAT" 2>/dev/null || true

  # 依頼を古い順に処理する
  shopt -s nullglob
  for req in "$REQ_DIR"/open_*.txt; do
    target="$(head -n 1 "$req" 2>/dev/null || true)"
    rm -f "$req"

    [ -z "$target" ] && continue
    if [ ! -e "$target" ]; then
      echo "  ⚠ ありません: $target"
      continue
    fi
    echo "  → 開きます: $target"
    "$OPENER" "$target" >/dev/null 2>&1 &
  done
  shopt -u nullglob

  sleep "$INTERVAL"
done
