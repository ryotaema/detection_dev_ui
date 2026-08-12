#!/usr/bin/env bash
# =============================================================================
# 環境を調べて「詰まっている原因」と「対処」を出す
#
#     ./tools/doctor.sh
#
#   起動できないときに見えるのは症状だけで、原因はそこに書かれていない。
#   たとえば「CVAT が 502」の実際の原因は、DB のパスワード不一致・初期化漏れ・
#   起動途中のいずれでもありうるし、ブラウザからは区別できない。
#   ここでは**症状ではなく原因**を突き止めて、次にやることだけを出す。
#
#   セットアップの最後に一度流しておくと、後から詰まりにくい。
# =============================================================================
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

NG=0          # 致命的な問題の数
WARN=0        # 動くが直したほうがよいものの数
ACTIONS=()    # 最後にまとめて出す対処

ok()   { printf '  \033[32m✅\033[0m %s: %s\n' "$1" "${2-}"; }
warn() { printf '  \033[33m⚠\033[0m  %s: %s\n' "$1" "${2-}"; WARN=$((WARN+1)); }
bad()  { printf '  \033[31m❌\033[0m %s: %s\n' "$1" "${2-}"; NG=$((NG+1)); }
act()  { ACTIONS+=("$1"); }
head_() { printf '\n\033[1m%s\033[0m\n' "$1"; }

# compose の解決結果を 1 度だけ取る（.env の展開結果を含む正しい値）
COMPOSE_JSON="$(docker compose config --format json 2>/dev/null)"

# JSON から値を引く小道具（python3 は Ubuntu 標準。PyYAML には依存しない）
cfg() {
  [ -n "$COMPOSE_JSON" ] || return 1
  printf '%s' "$COMPOSE_JSON" | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
    for k in sys.argv[1:]:
        d = d[k]
    print(d)
except Exception:
    sys.exit(1)
" "$@" 2>/dev/null
}

printf '\033[1m🩺 detection_dev_ui  環境チェック\033[0m\n'
printf '   %s\n' "$ROOT"

# --- 1. Docker --------------------------------------------------------------
head_ "1. Docker"
if ! command -v docker >/dev/null 2>&1; then
  bad "docker" "見つかりません"
  act "Docker を入れる → docs/docker_setup.md"
elif ! docker info >/dev/null 2>&1; then
  bad "docker daemon" "接続できません（権限か、起動していない）"
  act "sudo systemctl start docker / usermod -aG docker \$USER 後に再ログイン"
else
  ok "docker" "$(docker --version | sed 's/Docker version //;s/,.*//')"
  if docker compose version >/dev/null 2>&1; then
    ok "docker compose" "$(docker compose version --short 2>/dev/null)"
  else
    bad "docker compose" "Compose v2 がありません"
    act "docker-compose-plugin を入れる → docs/docker_setup.md"
  fi
fi

# --- 2. .env ----------------------------------------------------------------
head_ "2. 設定ファイル (.env)"
if [ ! -f .env ]; then
  bad ".env" "ありません"
  act ".env を作る → README の Step 3"
else
  ok ".env" "あります"
  miss=()
  for k in CVAT_USERNAME CVAT_PASSWORD CVAT_DB_PASSWORD CVAT_IAM_DB_PASSWORD; do
    grep -qE "^${k}=" .env || miss+=("$k")
  done
  if [ ${#miss[@]} -gt 0 ]; then
    bad "必須のキー" "足りません: ${miss[*]}"
    act ".env に ${miss[*]} を足す → README の Step 3"
  else
    ok "必須のキー" "そろっています"
  fi

  # COMPOSE_FILE が 2 行あると後の行だけが効き、serverless などが黙って外れる
  n=$(grep -cE "^COMPOSE_FILE=" .env)
  if [ "$n" -gt 1 ]; then
    bad "COMPOSE_FILE" "${n} 行あります（後の行だけが効きます）"
    act ".env の COMPOSE_FILE を 1 行にまとめる（例: docker-compose.yml:docker-compose.serverless.yml:docker-compose.cpu.yml）"
  elif [ "$n" -eq 1 ]; then
    ok "COMPOSE_FILE" "$(grep -E '^COMPOSE_FILE=' .env | cut -d= -f2-)"
  else
    ok "COMPOSE_FILE" "未設定（docker-compose.yml のみ）"
  fi
fi

if [ -z "$COMPOSE_JSON" ]; then
  bad "compose の設定" "読み取れません（構文エラーか、ファイルが足りない）"
  act "docker compose config を実行してエラーを確認する"
fi

# --- 3. GPU -----------------------------------------------------------------
head_ "3. GPU"
DEFAULT_RUNTIME="$(docker info --format '{{.DefaultRuntime}}' 2>/dev/null)"
HAS_SMI=0; nvidia-smi >/dev/null 2>&1 && HAS_SMI=1
HAS_GPU=0; lspci 2>/dev/null | grep -qi nvidia && HAS_GPU=1

GPU_REQUESTED="$(cfg services streamlit_app environment NVIDIA_VISIBLE_DEVICES)"
[ -z "$GPU_REQUESTED" ] && GPU_REQUESTED="(未設定)"

if [ "$HAS_SMI" = "1" ]; then
  ok "ドライバ" "$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
  if [ "$DEFAULT_RUNTIME" = "nvidia" ]; then
    ok "default runtime" "nvidia（自動アノテーションを GPU で使えます）"
  else
    warn "default runtime" "$DEFAULT_RUNTIME（自動アノテーションは GPU で動きません）"
    act "GPU で自動アノテーションを使うなら: sudo nvidia-ctk runtime configure --runtime=docker --set-as-default && sudo systemctl restart docker"
  fi
else
  if [ "$HAS_GPU" = "1" ]; then
    warn "ドライバ" "GPU はありますが nvidia-smi が動きません"
    act "ドライバを入れる（sudo ubuntu-drivers autoinstall して再起動）→ docs/docker_setup.md の 0."
  else
    ok "ドライバ" "GPU なしの構成（学習は CPU になります）"
  fi
  # ドライバが無いのに GPU を要求していると NVML エラーで起動できない
  if [ "$GPU_REQUESTED" != "void" ] && [ "$GPU_REQUESTED" != "(未設定)" ]; then
    bad "GPU の要求" "ドライバが無いのに GPU を要求しています（NVIDIA_VISIBLE_DEVICES=$GPU_REQUESTED）"
    act "CPU 構成で起動する: docker compose -f docker-compose.yml -f docker-compose.cpu.yml up -d"
  else
    ok "GPU の要求" "していません（CPU 構成）"
  fi
  if [ "$DEFAULT_RUNTIME" = "nvidia" ]; then
    warn "default runtime" "nvidia のままです（ドライバが無いので GPU 付きコンテナは起動できません）"
    act "戻す: sudo nvidia-ctk runtime configure --runtime=docker --set-as-default=false && sudo systemctl restart docker"
  fi
fi

# --- 4. コンテナ -------------------------------------------------------------
head_ "4. コンテナ"
if [ -n "$COMPOSE_JSON" ]; then
  RUNNING="$(docker compose ps --services --status running 2>/dev/null)"
  stopped=()
  while IFS= read -r svc; do
    [ -n "$svc" ] || continue
    printf '%s\n' "$RUNNING" | grep -qx "$svc" || stopped+=("$svc")
  done < <(docker compose config --services 2>/dev/null)
  total="$(docker compose config --services 2>/dev/null | grep -c .)"
  up=$((total - ${#stopped[@]}))

  if [ "$up" -eq 0 ]; then
    bad "起動状況" "1 つも動いていません"
    act "docker compose up -d"
  elif [ ${#stopped[@]} -gt 0 ]; then
    bad "起動状況" "$up / $total（止まっているものがあります）"
    for s in "${stopped[@]}"; do
      st="$(docker compose ps -a --format '{{.Service}}\t{{.State}}' 2>/dev/null \
            | awk -F'\t' -v n="$s" '$1==n {print $2; exit}')"
      printf '       - %s (%s)\n' "$s" "${st:-未作成}"
    done
    act "止まっているものを見る: docker compose logs --tail 50 ${stopped[0]}"
  else
    ok "起動状況" "$up / $total（すべて動いています）"
  fi
fi

# --- 5. CVAT のデータベース（502 の主因）------------------------------------
head_ "5. CVAT のデータベース"
if docker compose ps --services --status running 2>/dev/null | grep -qx cvat_db; then
  DB_PW="$(cfg services cvat_db environment POSTGRES_PASSWORD)"
  if [ -z "$DB_PW" ]; then
    warn "パスワード" "設定を読み取れませんでした"
  elif docker compose exec -T -e PGPASSWORD="$DB_PW" cvat_db \
        psql -U root -d postgres -c 'select 1' >/dev/null 2>&1; then
    ok "パスワード" "いまの設定で接続できます"
  else
    # PostgreSQL はボリュームが空のときしか POSTGRES_PASSWORD を読まない。
    # .env を後から変えても DB 側は古いままなので、ここで食い違う
    bad "パスワード" ".env の値では DB に接続できません"
    act "【502 の主因】DB は最初に起動したときのパスワードを保持しています。"
    act "  ・まだアノテーションしていない → docker compose down -v して README の Step 4 からやり直す"
    act "  ・データを残したい → .env の CVAT_DB_PASSWORD を初回起動時の値（.env 無しで起動したなら cvat_secret）に戻す"
  fi
else
  warn "cvat_db" "動いていません"
  act "docker compose up -d cvat_db"
fi

# cvat_server のログに出ている既知の失敗を拾う
if docker compose ps --services 2>/dev/null | grep -qx cvat_server; then
  LOG="$(docker compose logs --tail 200 cvat_server 2>/dev/null)"
  if printf '%s' "$LOG" | grep -q "password authentication failed"; then
    bad "cvat_server のログ" "DB のパスワード認証に失敗しています"
  elif printf '%s' "$LOG" | grep -qE "does not exist|relation .* does not exist"; then
    bad "cvat_server のログ" "テーブルがありません（初期化が済んでいません）"
    act "docker compose run --rm cvat_server init を実行する"
  elif printf '%s' "$LOG" | grep -q "Application startup complete"; then
    ok "cvat_server" "起動しています"
  fi
fi

# --- 6. サービスの応答 -------------------------------------------------------
head_ "6. サービスの応答"
probe() {  # 表示名 URL 期待コード
  code="$(curl -s -o /dev/null -m 5 -w '%{http_code}' "$2" 2>/dev/null)"
  case "$code" in
    "$3")  ok "$1" "HTTP $code" ;;
    502|503) bad "$1" "HTTP $code（受け口はあるが中身が応答していない）" ;;
    000)   bad "$1" "つながりません（起動していない可能性）" ;;
    *)     warn "$1" "HTTP $code" ;;
  esac
}
probe "CVAT"      "http://localhost:8080/api/server/health/" 200
probe "Streamlit" "http://localhost:8501"                    200
probe "MLflow"    "http://localhost:5000/health"             200

# --- まとめ -----------------------------------------------------------------
head_ "結果"
if [ "$NG" -eq 0 ] && [ "$WARN" -eq 0 ]; then
  printf '  \033[32m問題は見つかりませんでした。\033[0m\n'
  printf '  UI: http://localhost:8501   CVAT: http://localhost:8080\n\n'
  exit 0
fi

printf '  問題 %d 件 / 注意 %d 件\n' "$NG" "$WARN"
if [ ${#ACTIONS[@]} -gt 0 ]; then
  printf '\n\033[1m  やること（上から順に）\033[0m\n'
  i=1
  for a in "${ACTIONS[@]}"; do
    case "$a" in
      "  "*) printf '     %s\n' "$a" ;;          # 補足行はぶら下げる
      *)     printf '  %d. %s\n' "$i" "$a"; i=$((i+1)) ;;
    esac
  done
fi
printf '\n  詳しい手順: README.md のトラブルシューティング / docs/docker_setup.md\n\n'
[ "$NG" -gt 0 ] && exit 1 || exit 0
