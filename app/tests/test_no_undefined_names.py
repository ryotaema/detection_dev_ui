# =============================================================================
# 「import し忘れ」を機械的に見つける
#
#   main.py の分割で、モジュールをまたいで参照していた名前が
#   import されないまま残り、実行して初めて NameError になる事故が起きた
#   （_MODEL_OPTS / time / PREDICTIONS_DIR / _train_state の 4 件）。
#
#   質が悪いのは**分岐の奥でしか踏まない**こと。
#   学習は分割後ずっと動かない状態だったのに、誰も気づかなかった。
#   実行しないと分からない性質なので、構文木で静的に見張る。
# =============================================================================
from __future__ import annotations

import ast
import builtins
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent

# モジュールに暗黙で存在するもの
_DUNDERS = {"__file__", "__name__", "__doc__", "__package__",
            "__spec__", "__loader__", "__builtins__", "__path__"}


def _star_names() -> set[str]:
    """`from core import *` で入ってくる名前"""
    import core

    names = {n for n in dir(core) if not n.startswith("_")}
    names |= set(getattr(core, "__all__", []))
    return names


def _bound_names(tree: ast.AST, star: set[str]) -> set[str]:
    """そのファイルで定義・束縛される名前をすべて集める。

    スコープは見ない（見ないほうが安全側に倒れる＝誤検出しない）。
    狙いは「どこにも無い名前」を見つけることなので、これで足りる。
    """
    bound: set[str] = set(dir(builtins)) | _DUNDERS
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and any(a.name == "*" for a in node.names):
            bound |= star
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                bound.add((a.asname or a.name).split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            bound.add(node.id)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            bound |= set(node.names)
    return bound


def _undefined(path: Path, star: set[str]) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    bound = _bound_names(tree, star)
    out = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
                and node.id not in bound):
            out.append(f"{path.relative_to(APP_DIR)}:{node.lineno}  {node.id}")
    return out


def test_どのファイルにも未定義の名前が残っていない():
    star = _star_names()
    found: list[str] = []
    for f in sorted(APP_DIR.rglob("*.py")):
        if "tests" in f.parts:
            continue
        found += _undefined(f, star)

    assert not found, (
        "import されていない名前があります（実行時に NameError になります）:\n  "
        + "\n  ".join(found))
