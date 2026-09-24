# =============================================================================
# 学習の子プロセスの入口
#
#   python -m core.train_runner <config.json>
#
#   core.training._train_worker() が起動する。直接動かすものではない。
#   標準出力に「ログ行」と「イベント行（EVENT_PREFIX + JSON）」を出す。
# =============================================================================
from __future__ import annotations

import json
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    from core.training import EVENT_PREFIX, run_training_job

    cfg = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    stop_file = Path(cfg["stop_file"])

    def emit(kind: str, **kw) -> None:
        print(EVENT_PREFIX + json.dumps({"type": kind, **kw}, ensure_ascii=False),
              flush=True)

    run_training_job(cfg, emit, stop_file.exists)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
