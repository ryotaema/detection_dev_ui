#!/usr/bin/env bash
# =============================================================================
# 「📂 フォルダを開く」の常駐をログイン時に自動で始めるようにする
#
#   UI からフォルダを開くには、ホスト側で open_folder_watcher.sh が
#   動いている必要がある。毎回手で起動するのは面倒なので、
#   systemd のユーザーサービスとして登録する。
#
#       ./tools/install_folder_watcher.sh            登録して開始
#       ./tools/install_folder_watcher.sh --status   状態を見る
#       ./tools/install_folder_watcher.sh --uninstall 解除
#
#   systemd --user を使うのは、デスクトップ環境に依らず同じ手順で済むから。
#   ~/.config/autostart/*.desktop でも同じことはできるが、
#   状態確認とログの追い方が環境ごとに変わる。
# =============================================================================
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WATCHER="$ROOT/tools/open_folder_watcher.sh"
UNIT_NAME="detection-dev-ui-folder-watcher.service"
UNIT_DIR="$HOME/.config/systemd/user"
UNIT="$UNIT_DIR/$UNIT_NAME"

_check_systemd() {
  if ! command -v systemctl >/dev/null 2>&1; then
    echo "❌ systemctl がありません。" >&2
    echo "   代わりに、デスクトップの「自動起動するアプリ」に次を登録してください:" >&2
    echo "     $WATCHER" >&2
    exit 1
  fi
}

case "${1:-install}" in
  --status)
    _check_systemd
    systemctl --user status "$UNIT_NAME" --no-pager 2>&1 | head -20
    echo ""
    echo "ログを追う: journalctl --user -u $UNIT_NAME -f"
    exit 0
    ;;

  --uninstall)
    _check_systemd
    systemctl --user disable --now "$UNIT_NAME" 2>/dev/null
    rm -f "$UNIT"
    systemctl --user daemon-reload
    echo "✅ 自動起動を解除しました"
    echo "   必要なときは手で動かせます: $WATCHER"
    exit 0
    ;;

  install|"")
    ;;

  *)
    echo "使い方: $0 [--status|--uninstall]" >&2
    exit 1
    ;;
esac

_check_systemd

if [ ! -x "$WATCHER" ]; then
  echo "❌ $WATCHER が見つからないか、実行できません" >&2
  echo "   chmod +x tools/open_folder_watcher.sh を実行してください" >&2
  exit 1
fi

mkdir -p "$UNIT_DIR"
cat > "$UNIT" <<EOF
[Unit]
Description=detection_dev_ui: UI からの依頼でフォルダを開く
Documentation=file://$ROOT/docs/overview.md

[Service]
Type=simple
ExecStart=$WATCHER
Restart=on-failure
RestartSec=5
# ファイルアプリを開くのに必要（デスクトップのセッションから引き継ぐ）
PassEnvironment=DISPLAY XAUTHORITY DBUS_SESSION_BUS_ADDRESS

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
systemctl --user enable --now "$UNIT_NAME"

sleep 2
if systemctl --user is-active --quiet "$UNIT_NAME"; then
  echo "✅ 自動起動を登録しました（次回ログイン以降も動きます）"
  echo ""
  echo "   状態:   $0 --status"
  echo "   ログ:   journalctl --user -u $UNIT_NAME -f"
  echo "   解除:   $0 --uninstall"
  echo ""
  echo "   UI の「📂 開く」を押すとファイルアプリが開くようになります。"
else
  echo "⚠ 登録はしましたが、起動を確認できませんでした。" >&2
  echo "   $0 --status で状態を見てください。" >&2
  exit 1
fi

# ログアウト中もサービスを保つ（任意）。失敗しても致命的ではない
if command -v loginctl >/dev/null 2>&1; then
  loginctl enable-linger "$USER" >/dev/null 2>&1 || true
fi
