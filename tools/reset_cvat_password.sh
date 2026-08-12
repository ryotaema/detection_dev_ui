#!/usr/bin/env bash
# =============================================================================
# CVAT のログイン情報を `.env` に合わせる（作成もリセットも同じ操作でできる）
#
#     ./tools/reset_cvat_password.sh              .env の利用者に合わせる
#     ./tools/reset_cvat_password.sh <ユーザー名>  対象を指定する
#     ./tools/reset_cvat_password.sh -y           確認を省く
#
#   **CVAT のログイン用パスワードと `.env` の CVAT_PASSWORD は別管理**で、
#   前者は `createsuperuser` の対話入力で決まる。ここが食い違うと
#   「ブラウザからは入れるのに Streamlit から CVAT を操作できない」（逆もある）
#   という分かりにくい壊れ方をする。
#
#   `manage.py changepassword` は対話式で、しかも作成はできない。
#   ここでは **`.env` の値に揃える** ことだけを目的にして、
#   利用者がいなければ作り、いればパスワードを設定し直す。
#
#   パスワードは引数ではなく環境変数で渡す（`ps` で見えないようにするため）。
# =============================================================================
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

ASSUME_YES=0
TARGET_USER=""
for a in "$@"; do
  case "$a" in
    -y|--yes) ASSUME_YES=1 ;;
    -h|--help) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    -*) echo "不明なオプション: $a" >&2; exit 1 ;;
    *)  TARGET_USER="$a" ;;
  esac
done

die() { printf '\033[31m❌ %s\033[0m\n' "$1" >&2; exit 1; }

# --- 設定を読む（.env の展開結果を compose から取る）-------------------------
CONF="$(docker compose config --format json 2>/dev/null)" \
  || die "docker compose config を実行できません（このディレクトリで動かしてください）"

read -r ENV_USER ENV_PASS <<EOF
$(printf '%s' "$CONF" | python3 -c "
import json, sys
d = json.load(sys.stdin)
e = d['services']['streamlit_app']['environment']
print(e.get('CVAT_USERNAME', ''), e.get('CVAT_PASSWORD', ''))
" 2>/dev/null)
EOF

USER_NAME="${TARGET_USER:-$ENV_USER}"
[ -n "$USER_NAME" ] || die ".env の CVAT_USERNAME が読み取れません"
[ -n "$ENV_PASS" ]  || die ".env の CVAT_PASSWORD が空です。先に .env を設定してください"

docker compose ps --services --status running 2>/dev/null | grep -qx cvat_server \
  || die "cvat_server が動いていません（docker compose up -d cvat_server）"

# --- 確認 -------------------------------------------------------------------
printf '利用者 \033[1m%s\033[0m のパスワードを .env の CVAT_PASSWORD に合わせます。\n' "$USER_NAME"
printf 'その利用者がいなければ、管理者として新しく作成します。\n\n'
if [ "$ASSUME_YES" -ne 1 ]; then
  read -r -p "続けますか? [y/N] " ans
  case "$ans" in [yY]*) ;; *) echo "中止しました"; exit 0 ;; esac
fi

# --- 作成 or 更新 -----------------------------------------------------------
# `~` はコンテナ内で展開させる（bash -c で包まないとホスト側の HOME になる）
OUT="$(docker compose exec -T -e RU="$USER_NAME" -e RP="$ENV_PASS" cvat_server bash -c '~/manage.py shell -c "
import os
from django.contrib.auth import get_user_model
U = get_user_model()
n, p = os.environ[\"RU\"], os.environ[\"RP\"]
u, created = U.objects.get_or_create(username=n, defaults={\"email\": n + \"@local.com\"})
u.set_password(p)
u.is_active = True
u.is_staff = True
u.is_superuser = True
u.save()
print(\"created\" if created else \"updated\")
"' 2>&1)"

case "$OUT" in
  *created*) printf '\033[32m✅ 利用者 %s を作成しました\033[0m\n' "$USER_NAME" ;;
  *updated*) printf '\033[32m✅ 利用者 %s のパスワードを更新しました\033[0m\n' "$USER_NAME" ;;
  *) printf '\033[31m❌ 失敗しました\033[0m\n%s\n' "$OUT" >&2; exit 1 ;;
esac

# --- 実際にログインして確かめる ---------------------------------------------
BODY="$(RU="$USER_NAME" RP="$ENV_PASS" python3 -c "
import json, os
print(json.dumps({'username': os.environ['RU'], 'password': os.environ['RP']}))")"
CODE="$(curl -s -o /dev/null -m 10 -w '%{http_code}' -X POST http://localhost:8080/api/auth/login \
  -H 'Content-Type: application/json' -d "$BODY" 2>/dev/null)"

if [ "$CODE" = "200" ]; then
  printf '\033[32m✅ ログインを確認しました\033[0m\n\n'
  printf '   CVAT: http://localhost:8080  （%s / .env の CVAT_PASSWORD）\n' "$USER_NAME"
  printf '   UI  : http://localhost:8501\n\n'
  exit 0
fi

printf '\033[33m⚠ パスワードは設定しましたが、ログインの確認ができませんでした (HTTP %s)\033[0m\n' "$CODE"
printf '  CVAT が起動しきっていない可能性があります。少し待って ./tools/doctor.sh を実行してください。\n'
exit 1
