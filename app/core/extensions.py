# =============================================================================
# 拡張機能（別リポジトリのツールの取り込み）
#
#   extensions/<名前>/ に git clone すると、そのツール専用のタブが増える。
#   このリポジトリ本体を膨らませずに、道具を足していけるようにするための仕組み。
#
#   設計上の約束:
#     - **探索の段階では拡張のコードを一切実行しない**。
#       マニフェスト(JSON)を読むだけ。import も subprocess も走らせない。
#     - 実行は利用者がボタンを押したときだけ。何を実行するかは事前に画面に出す。
#     - 拡張が壊れていても本体は落とさない（読めなければその拡張だけ無効にする）。
#
#   取り込み方は 3 通り:
#     streamlit … render() を持つモジュール。タブの中にそのまま描画できる（いちばん良い）
#     command   … CLI として実行し、標準出力を見せる
#     desktop   … Tkinter などの GUI。コンテナには画面が無いので**ホストで動かす**
#                 手順とコマンドを提示するだけに留める
# =============================================================================
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from .config import EXTENSION_MANIFEST, EXTENSIONS_DIR

# 同梱の既定マニフェスト。clone しただけで動くように、
# こちらが知っているリポジトリのぶんは用意しておく（相手のリポジトリを汚さない）。
PRESET_DIR = Path(__file__).resolve().parent.parent / "ext_presets"

KINDS = ("streamlit", "command", "desktop")

# 置換できるプレースホルダ。利用者に見せる説明も兼ねる
PLACEHOLDERS = {
    "{data_dir}":        "データセットの置き場 (data/)",
    "{models_dir}":      "モデルの置き場 (models/)",
    "{predictions_dir}": "推論結果の置き場 (predictions/)",
    "{ext_dir}":         "この拡張のディレクトリ",
}


# 拡張ではないディレクトリ（作業用に紛れ込みがちなもの）
IGNORED_DIRS = {"__pycache__", "node_modules", "venv", ".venv", "site-packages"}


def _safe_name(name: str) -> bool:
    """ディレクトリ名として素直なものだけ受け付ける"""
    if name.startswith(".") or name in IGNORED_DIRS:
        return False
    return bool(re.fullmatch(r"[A-Za-z0-9._-]+", name))


# マニフェストを探す場所（先に見つかったものを使う）。
#   連携用のファイルが増えても散らからないよう、ディレクトリにまとめる形を標準にする。
#   単体ファイルを直下に置く形も、小さい拡張のために残しておく。
MANIFEST_LOCATIONS = (
    f"extension/{EXTENSION_MANIFEST}",     # 標準。連携用の資材をここにまとめる
    f".dev_ui/{EXTENSION_MANIFEST}",        # 隠しディレクトリにしたい場合
    EXTENSION_MANIFEST,                     # 直下に 1 ファイルだけ置く簡易形
)


def _read_json(path: Path) -> Optional[dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _preset_manifest(dir_name: str) -> Optional[dict]:
    p = PRESET_DIR / f"{dir_name}.json"
    return _read_json(p) if p.exists() else None


def _infer_manifest(ext_dir: Path) -> dict:
    """マニフェストが無いときに、ファイル構成から当たりを付ける。

    当てずっぽうなので、あくまで「何もしないよりまし」の位置づけ。
    正しく出したいならマニフェストを置いてもらう。
    """
    actions: list[dict] = []

    # Streamlit として取り込めそうなもの
    for cand in ("extension.py", "streamlit_app.py", "st_app.py"):
        if (ext_dir / cand).exists():
            actions.append({
                "label": f"{cand} を描画",
                "kind": "streamlit",
                "module": cand[:-3],
                "function": "render",
            })
            break

    # デスクトップ GUI らしきもの
    for gui in sorted(ext_dir.glob("*gui*.py")) + sorted(ext_dir.glob("*_tool.py")):
        actions.append({
            "label": f"{gui.name} を起動（ホストで実行）",
            "kind": "desktop",
            "command": ["python3", gui.name],
        })

    # CLI らしきもの
    for script in sorted((ext_dir / "scripts").glob("*.py")) if (ext_dir / "scripts").is_dir() else []:
        actions.append({
            "label": f"scripts/{script.name}",
            "kind": "command",
            "command": ["python3", f"scripts/{script.name}", "--help"],
            "note": "引数はマニフェストに書くと入力欄が出ます",
        })

    return {
        "name": ext_dir.name,
        "icon": "🧩",
        "description": "（マニフェストが無いため構成から推測しています）",
        "actions": actions,
        "inferred": True,
    }


def _normalize(manifest: dict, ext_dir: Path) -> dict:
    """マニフェストを整える。壊れた項目は落とし、理由を warnings に積む。"""
    warnings: list[str] = []
    actions: list[dict] = []

    for i, raw in enumerate(manifest.get("actions") or []):
        if not isinstance(raw, dict):
            warnings.append(f"actions[{i}] が辞書ではありません")
            continue
        kind = raw.get("kind")
        if kind not in KINDS:
            warnings.append(f"actions[{i}]: 未対応の kind です: {kind}")
            continue
        label = str(raw.get("label") or f"操作 {i + 1}")

        if kind == "streamlit":
            if not raw.get("module"):
                warnings.append(f"{label}: module が指定されていません")
                continue
            actions.append({
                "label": label, "kind": kind,
                "module": str(raw["module"]),
                "function": str(raw.get("function") or "render"),
                "note": str(raw.get("note") or ""),
            })
        else:
            cmd = raw.get("command")
            if not isinstance(cmd, list) or not cmd or not all(isinstance(c, str) for c in cmd):
                warnings.append(f"{label}: command は文字列のリストで指定してください")
                continue
            actions.append({
                "label": label, "kind": kind,
                "command": [str(c) for c in cmd],
                "inputs": [str(x) for x in (raw.get("inputs") or [])],
                "note": str(raw.get("note") or ""),
            })

    return {
        "name": str(manifest.get("name") or ext_dir.name),
        "icon": str(manifest.get("icon") or "🧩"),
        "description": str(manifest.get("description") or ""),
        "url": str(manifest.get("url") or ""),
        "requirements": [str(r) for r in (manifest.get("requirements") or [])],
        "actions": actions,
        "warnings": warnings,
        "inferred": bool(manifest.get("inferred")),
    }


def missing_requirements(names: list[str]) -> list[str]:
    """import できない依存を返す（実際に import して確かめる）"""
    import importlib.util
    missing = []
    for n in names:
        mod = n.split("==")[0].split(">=")[0].strip()
        # パッケージ名と import 名が違うよく使うもの
        mod = {"opencv-python": "cv2", "Pillow": "PIL", "pyyaml": "yaml",
               "PyYAML": "yaml", "scikit-learn": "sklearn"}.get(mod, mod)
        if not mod:
            continue
        try:
            if importlib.util.find_spec(mod) is None:
                missing.append(n)
        except Exception:
            missing.append(n)
    return missing


def git_revision(ext_dir: Path) -> str:
    """clone したものがどの版か分かるように短いハッシュを返す（失敗しても空文字）"""
    if not (Path(ext_dir) / ".git").exists() or not shutil.which("git"):
        return ""
    try:
        r = subprocess.run(
            ["git", "-C", str(ext_dir), "describe", "--always", "--dirty"],
            capture_output=True, text=True, timeout=5)
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        return ""


def discover_extensions() -> list[dict]:
    """extensions/ を走査して拡張の一覧を返す。**拡張のコードは実行しない。**"""
    root = Path(EXTENSIONS_DIR)
    if not root.exists():
        return []

    found: list[dict] = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if not _safe_name(d.name):
            continue

        # ① 拡張リポジトリ自身が持つマニフェスト（これを標準にしたい）。
        #    ツールの引数やエントリポイントは向こうで変わるので、
        #    その定義は実装している側に置くのが本筋。
        manifest, source, base = None, "", d
        for rel in MANIFEST_LOCATIONS:
            manifest = _read_json(d / rel)
            if manifest is not None:
                source = f"リポジトリ内の {rel}"
                base = (d / rel).parent
                break

        if manifest is None:
            manifest = _preset_manifest(d.name)             # ② 変更できないリポジトリ用
            source = "同梱の既定マニフェスト"
        if manifest is None:
            manifest = _infer_manifest(d)                   # ③ 構成から推測
            source = "ファイル構成からの推測"

        ext = _normalize(manifest, d)
        ext.update({
            "dir": str(d),                # 実行時の作業ディレクトリ（リポジトリの根）
            "base_dir": str(base),        # module を解決する起点（マニフェストのある場所）
            "dir_name": d.name,
            "manifest_source": source,
            "has_own_manifest": base != d or source.startswith("リポジトリ内"),
            "revision": git_revision(d),
            "missing": missing_requirements(ext["requirements"]),
        })
        found.append(ext)
    return found


def scaffold_manifest(ext_dir: Path, manifest: dict) -> dict:
    """推測した内容をもとに、拡張リポジトリ側へ雛形を書き出す。

    これを向こうのリポジトリにコミットしてもらえば、
    以降は clone するだけで正しいタブが出るようになる。
    """
    target = Path(ext_dir) / "extension" / EXTENSION_MANIFEST
    if target.exists():
        return {"ok": False, "path": str(target),
                "error": f"すでにあります: extension/{EXTENSION_MANIFEST}"}

    body = {
        "name": manifest.get("name") or Path(ext_dir).name,
        "icon": manifest.get("icon") or "🧩",
        "description": manifest.get("description") or "",
        "url": manifest.get("url") or "",
        "requirements": manifest.get("requirements") or [],
        "actions": [
            {k: v for k, v in a.items() if v not in ("", [], None)}
            for a in (manifest.get("actions") or [])
        ],
    }
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(body, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        # コンテナは root で動くので、そのままだとホスト側で編集・削除できない。
        # 書き出したものは向こうのリポジトリでコミットしてもらう前提なので、
        # 誰でも書き換えられるようにしておく。
        try:
            os.chmod(target.parent, 0o777)
            os.chmod(target, 0o666)
        except Exception:
            pass
    except Exception as e:
        return {"ok": False, "path": str(target), "error": str(e)}
    return {"ok": True, "path": str(target), "error": ""}


def resolve_command(command: list[str], ext_dir: Path, values: Optional[dict] = None) -> list[str]:
    """コマンド内のプレースホルダを実際のパスに置き換える"""
    from .config import DATA_DIR, MODELS_DIR, PREDICTIONS_DIR

    table = {
        "{data_dir}": str(DATA_DIR),
        "{models_dir}": str(MODELS_DIR),
        "{predictions_dir}": str(PREDICTIONS_DIR),
        "{ext_dir}": str(ext_dir),
    }
    table.update({f"{{{k}}}": str(v) for k, v in (values or {}).items()})

    out = []
    for part in command:
        for key, val in table.items():
            part = part.replace(key, val)
        out.append(part)
    return out


def run_extension_command(
    command: list[str],
    ext_dir: Path,
    values: Optional[dict] = None,
    timeout: int = 900,
) -> dict:
    """拡張の CLI を実行して結果を返す。

    作業ディレクトリは拡張のディレクトリ。
    相手のスクリプトが `dsm/` などを相対で import できるように PYTHONPATH も通す。
    """
    resolved = resolve_command(command, Path(ext_dir), values)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ext_dir)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))

    try:
        r = subprocess.run(
            resolved, cwd=str(ext_dir), env=env,
            capture_output=True, text=True, timeout=timeout)
        return {
            "ok": r.returncode == 0,
            "returncode": r.returncode,
            "stdout": r.stdout,
            "stderr": r.stderr,
            "command": resolved,
            "error": "" if r.returncode == 0 else f"終了コード {r.returncode}",
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "returncode": -1, "stdout": "", "stderr": "",
                "command": resolved, "error": f"{timeout} 秒で打ち切りました"}
    except FileNotFoundError as e:
        return {"ok": False, "returncode": -1, "stdout": "", "stderr": "",
                "command": resolved,
                "error": f"コマンドが見つかりません: {e}"}
    except Exception as e:
        return {"ok": False, "returncode": -1, "stdout": "", "stderr": "",
                "command": resolved, "error": str(e)}


def load_streamlit_action(ext_dir: Path, module: str, function: str):
    """拡張の描画関数を読み込む。**ここで初めて拡張のコードが動く。**

    戻り値: (関数, エラー文字列)。読めなければ (None, 理由)
    """
    import importlib.util
    import sys

    path = Path(ext_dir) / f"{module.replace('.', '/')}.py"
    if not path.exists():
        return None, f"{path.name} が見つかりません"

    # 拡張どうしで名前がぶつからないように、モジュール名を拡張ごとに分ける
    mod_name = f"_ext_{Path(ext_dir).name}_{module.replace('.', '_')}"
    try:
        # 相対 import が使えるように拡張のディレクトリを一時的に通す
        added = str(ext_dir) not in sys.path
        if added:
            sys.path.insert(0, str(ext_dir))
        try:
            spec = importlib.util.spec_from_file_location(mod_name, path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[mod_name] = mod
            spec.loader.exec_module(mod)
        finally:
            if added:
                sys.path.remove(str(ext_dir))
    except Exception as e:
        return None, f"{module} の読み込みに失敗しました: {type(e).__name__}: {e}"

    fn = getattr(mod, function, None)
    if not callable(fn):
        return None, f"{module}.{function}() が見つかりません"
    return fn, ""
