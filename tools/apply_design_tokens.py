#!/usr/bin/env python3
"""Claude Design で調整した配色を `app/ui/theme.py` に書き戻す。

`tools/build_design_system.py` が生成した `tokens/colors.html` を読み、
`.t0`〜`.tN` の CSS 変数から `PRESET_THEMES` を組み直す。

    # Claude Design 側の colors.html を落としてきてから
    python3 tools/apply_design_tokens.py path/to/colors.html            # 差分を見るだけ
    python3 tools/apply_design_tokens.py path/to/colors.html --write    # 実際に書き換える

既定は dry-run。`--write` を付けたときだけ theme.py を書き換える。

読むのは `<style>` 内の `.tN { --var: value; ... }` だけで、
どのパネルがどのテーマかは `data-theme-name` 属性で判定する。
プレビューの見出しテキストが編集されていても影響を受けない。
"""
from __future__ import annotations

import argparse
import ast
import difflib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
THEME_PY = ROOT / "app" / "ui" / "theme.py"

# build_design_system.py の VAR_MAP と同じ対応（CSS 変数名 → テーマ辞書のキー）
VAR_TO_KEY = {
    "--bg-app": "bg_app", "--bg-sidebar": "bg_sidebar",
    "--bg-card": "bg_card", "--bg-card-inner": "bg_card_inner",
    "--bg-log": "bg_log", "--border": "border",
    "--border-accent": "border_accent", "--text-primary": "text_primary",
    "--text-secondary": "text_secondary", "--text-muted": "text_muted",
    "--accent": "accent", "--accent-dark": "accent_dark",
    "--success": "success", "--success-bg": "success_bg",
    "--success-border": "success_border", "--warning": "warning",
    "--warning-bg": "warning_bg", "--warning-border": "warning_border",
    "--error": "error", "--error-bg": "error_bg",
    "--error-border": "error_border", "--btn-bg": "btn_bg",
    "--btn-hover": "btn_hover", "--chip-bg": "chip_bg",
    "--chip-border": "chip_border", "--chip-text": "chip_text",
}

# theme.py に書き出すときの並び（1 行にまとめる単位）
WRITE_LAYOUT = [
    ["bg_app", "bg_sidebar", "bg_card"],
    ["bg_card_inner", "bg_log"],
    ["border", "border_accent"],
    ["text_primary", "text_secondary", "text_muted"],
    ["accent", "accent_dark"],
    ["success", "success_bg", "success_border"],
    ["warning", "warning_bg", "warning_border"],
    ["error", "error_bg", "error_border"],
    ["btn_bg", "btn_hover"],
    ["chip_bg", "chip_border", "chip_text"],
]

HEX_RE = re.compile(r"^#[0-9a-fA-F]{6}$")


class ParseError(Exception):
    pass


# ---------------------------------------------------------------------------
def parse_colors_html(text: str) -> dict[str, dict[str, str]]:
    """colors.html から {テーマ名: {キー: 色}} を取り出す。"""
    # パネルの index → テーマ名
    names: dict[int, str] = {}
    for m in re.finditer(
        r'data-theme-index="(\d+)"\s+data-theme-name="([^"]*)"', text
    ):
        names[int(m.group(1))] = _unescape(m.group(2))
    if not names:
        raise ParseError(
            "data-theme-name 属性が見つかりません。\n"
            "tools/build_design_system.py で生成した colors.html を渡してください"
        )

    # .tN { ... } の中身
    themes: dict[str, dict[str, str]] = {}
    for idx, name in sorted(names.items()):
        m = re.search(r"\.t%d\s*\{([^}]*)\}" % idx, text)
        if not m:
            raise ParseError(f".t{idx} の CSS ブロックが見つかりません（{name}）")

        colors: dict[str, str] = {}
        for var, value in re.findall(r"(--[a-z-]+)\s*:\s*([^;]+);", m.group(1)):
            key = VAR_TO_KEY.get(var)
            if key is None:
                continue                      # 知らない変数は黙って無視する
            value = value.strip()
            if not HEX_RE.match(value):
                raise ParseError(
                    f"{name} の {var} が 6 桁の 16 進表記ではありません: {value!r}\n"
                    "（rgb() や名前付きの色には対応していません）"
                )
            colors[key] = value.lower()

        missing = set(VAR_TO_KEY.values()) - set(colors)
        if missing:
            raise ParseError(
                f"{name} に足りない変数があります: {', '.join(sorted(missing))}"
            )
        themes[name] = colors
    return themes


def _unescape(s: str) -> str:
    return (s.replace("&quot;", '"').replace("&#x27;", "'")
             .replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&"))


# ---------------------------------------------------------------------------
def find_preset_themes(src: str) -> tuple[int, int, dict]:
    """theme.py の PRESET_THEMES の行範囲（0 始まり・終端含む）と現在値を返す。"""
    tree = ast.parse(src)
    for node in tree.body:
        targets = getattr(node, "targets", []) or [getattr(node, "target", None)]
        for t in targets:
            if isinstance(t, ast.Name) and t.id == "PRESET_THEMES":
                return node.lineno - 1, node.end_lineno - 1, ast.literal_eval(node.value)
    raise ParseError("PRESET_THEMES が theme.py に見つかりません")


def render_preset_themes(themes: dict[str, dict[str, str]]) -> list[str]:
    """PRESET_THEMES の定義を組み立てる（既存の書き方に合わせる）。"""
    out = ["PRESET_THEMES: dict[str, dict] = {"]
    for name, colors in themes.items():
        out.append(f'    "{name}": {{')
        for group in WRITE_LAYOUT:
            out.append("        " + " ".join(f'"{k}": "{colors[k]}",' for k in group))
        out.append("    },")
    out.append("}")
    return out


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("colors_html", type=Path,
                    help="Claude Design から取得した tokens/colors.html")
    ap.add_argument("--write", action="store_true",
                    help="実際に theme.py を書き換える（既定は差分の表示のみ）")
    args = ap.parse_args()

    if not args.colors_html.exists():
        print(f"❌ ファイルがありません: {args.colors_html}", file=sys.stderr)
        return 1

    try:
        new_themes = parse_colors_html(args.colors_html.read_text(encoding="utf-8"))
    except ParseError as e:
        print(f"❌ colors.html を読めません:\n{e}", file=sys.stderr)
        return 1

    src = THEME_PY.read_text(encoding="utf-8")
    try:
        start, end, cur_themes = find_preset_themes(src)
    except ParseError as e:
        print(f"❌ {e}", file=sys.stderr)
        return 1

    # テーマ名の増減は事故の元なので拒否する。
    # 名前を変えたい・増やしたいときは theme.py を直接編集してから生成し直す。
    if set(new_themes) != set(cur_themes):
        print("❌ テーマの顔ぶれが theme.py と一致しません。", file=sys.stderr)
        print(f"   theme.py    : {', '.join(cur_themes)}", file=sys.stderr)
        print(f"   colors.html : {', '.join(new_themes)}", file=sys.stderr)
        print("   テーマの追加・改名は theme.py 側で行ってから生成し直してください。",
              file=sys.stderr)
        return 1

    # theme.py の並び順を保つ（colors.html 側の順序には従わない）
    ordered = {name: new_themes[name] for name in cur_themes}

    changed = [
        (name, key, cur_themes[name][key], ordered[name][key])
        for name in ordered for key in ordered[name]
        if cur_themes[name][key].lower() != ordered[name][key]
    ]
    if not changed:
        print("変更はありません。")
        return 0

    print(f"■ 変更される色: {len(changed)} 件")
    for name, key, old, new in changed:
        print(f"  {name:<16} {key:<14} {old} → {new}")

    lines = src.split("\n")
    new_src = "\n".join(lines[:start] + render_preset_themes(ordered) + lines[end + 1:])

    print("\n■ theme.py の差分")
    for d in difflib.unified_diff(src.split("\n"), new_src.split("\n"),
                                  "theme.py (現在)", "theme.py (変更後)", lineterm="", n=1):
        print("  " + d)

    if not args.write:
        print("\n(dry-run です。実際に書き換えるには --write を付けてください)")
        return 0

    # 書き換えた結果が壊れていないことを確かめてから保存する
    try:
        _, _, roundtrip = find_preset_themes(new_src)
    except Exception as e:
        print(f"❌ 生成結果が壊れています。書き換えを中止しました: {e}", file=sys.stderr)
        return 1
    if roundtrip != ordered:
        print("❌ 生成結果が意図した内容と一致しません。書き換えを中止しました。",
              file=sys.stderr)
        return 1

    THEME_PY.write_text(new_src, encoding="utf-8")
    print(f"\n✅ {THEME_PY.relative_to(ROOT)} を更新しました")
    print("   反映: docker compose restart streamlit_app")
    print("   戻す: git checkout app/ui/theme.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
