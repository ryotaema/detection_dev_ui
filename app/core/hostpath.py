# =============================================================================
# コンテナ内のパス ↔ ホストのパス、およびフォルダを開く依頼
#
#   この UI は Docker コンテナの中で動いていて、画面（X ディスプレイ）も
#   `xdg-open` も持たない。つまり**コンテナから OS のファイルアプリは開けない**。
#
#   そこで 2 段構えにする:
#     ① ホスト側のパスを表示する（コピーすればどこでも使える。常に動く）
#     ② `tools/open_folder_watcher.sh` をホストで動かしておくと、
#        ボタン 1 つで実際にファイルアプリが開く（任意）
#
#   ホストのパスは docker.sock 経由で自分自身のマウント表を引いて求める。
#   決め打ちにすると、別の場所に clone した人の環境で外れるため。
# =============================================================================
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import time
from pathlib import Path
from typing import Optional

from .config import PREDICTIONS_DIR

# ホストのウォッチャーとやり取りする場所。
# bind mount された predictions/ の下に置くことで、追加のマウントなしで共有できる。
REQUEST_DIR = PREDICTIONS_DIR / ".open_requests"
HEARTBEAT = REQUEST_DIR / ".watcher_alive"
HEARTBEAT_TIMEOUT = 30          # 秒。これより古ければ動いていないとみなす

_MOUNT_CACHE: Optional[list[tuple[str, str]]] = None


def _docker_available() -> bool:
    return bool(shutil.which("docker")) and Path("/var/run/docker.sock").exists()


def _load_mounts() -> list[tuple[str, str]]:
    """自分自身のマウント表を (コンテナ内, ホスト) の並びで返す。

    長いパスから先に見るよう並べ替える（/workspace より
    /workspace/data を優先して当てるため）。
    """
    global _MOUNT_CACHE
    if _MOUNT_CACHE is not None:
        return _MOUNT_CACHE

    mounts: list[tuple[str, str]] = []
    if _docker_available():
        for ident in (socket.gethostname(), "streamlit_app"):
            try:
                r = subprocess.run(
                    ["docker", "inspect", ident, "--format", "{{json .Mounts}}"],
                    capture_output=True, text=True, timeout=10)
                if r.returncode != 0 or not r.stdout.strip():
                    continue
                for m in json.loads(r.stdout):
                    dst, src = m.get("Destination"), m.get("Source")
                    if dst and src:
                        mounts.append((dst, src))
                break
            except Exception:
                continue

    mounts.sort(key=lambda x: -len(x[0]))
    _MOUNT_CACHE = mounts
    return mounts


def host_path_available() -> bool:
    return bool(_load_mounts())


def to_host_path(container_path) -> Optional[str]:
    """コンテナ内のパスをホストのパスに読み替える。

    対応が分からなければ None（呼び出し側で「分からない」と伝える）。
    """
    p = str(Path(container_path))
    for dst, src in _load_mounts():
        if p == dst:
            return src
        if p.startswith(dst.rstrip("/") + "/"):
            return src.rstrip("/") + p[len(dst.rstrip("/")):]
    return None


# ---------------------------------------------------------------------------
# ホスト側のウォッチャーへの依頼
# ---------------------------------------------------------------------------
def watcher_running() -> bool:
    """ホストのウォッチャーが動いているか（心拍ファイルの新しさで見る）"""
    try:
        return (HEARTBEAT.exists()
                and (time.time() - HEARTBEAT.stat().st_mtime) < HEARTBEAT_TIMEOUT)
    except Exception:
        return False


def request_open(container_path) -> dict:
    """「このフォルダを開いてほしい」という依頼をホスト側へ置く。

    ウォッチャーが動いていなければ、依頼は置かずに理由を返す
    （溜め込んでも意味がなく、あとから一斉に開いても困るため）。
    """
    host = to_host_path(container_path)
    if host is None:
        return {"ok": False, "host_path": None,
                "error": "ホスト側のパスが分かりませんでした"}

    if not watcher_running():
        return {"ok": False, "host_path": host,
                "error": "ホスト側のウォッチャーが動いていません"}

    try:
        REQUEST_DIR.mkdir(parents=True, exist_ok=True)
        req = REQUEST_DIR / f"open_{int(time.time() * 1000)}.txt"
        req.write_text(host + "\n", encoding="utf-8")
        try:
            os.chmod(req, 0o666)      # ホスト側（別ユーザー）が消せるように
        except Exception:
            pass
        return {"ok": True, "host_path": host, "error": ""}
    except Exception as e:
        return {"ok": False, "host_path": host, "error": str(e)}


def open_command(container_path) -> str:
    """手で実行する場合のコマンド（ウォッチャーが無いとき用）"""
    host = to_host_path(container_path)
    if host is None:
        return ""
    return f'xdg-open "{host}"'
