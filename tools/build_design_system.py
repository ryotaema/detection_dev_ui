#!/usr/bin/env python3
"""Claude Design に上げるコンポーネントプレビューを生成する。

`app/ui/theme.py` の配色と `app/main.py` のカスタム CSS を読み取り、
各コンポーネントを 4 テーマ横断で並べた HTML を書き出す。

手で書き写すと必ず実装とずれるので、必ずここから生成すること。
生成先の `design_system/` は .gitignore 済み（このスクリプトだけを追跡する）。

    python3 tools/build_design_system.py
"""
from __future__ import annotations

import ast
import html
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
THEME_PY = ROOT / "app" / "ui" / "theme.py"
MAIN_PY = ROOT / "app" / "main.py"
OUT_DIR = ROOT / "design_system"

# CSS 変数名 ⇔ テーマ辞書のキー（theme.py の _VAR_MAP と同じ対応）
VAR_MAP = [
    ("--bg-app", "bg_app"), ("--bg-sidebar", "bg_sidebar"),
    ("--bg-card", "bg_card"), ("--bg-card-inner", "bg_card_inner"),
    ("--bg-log", "bg_log"), ("--border", "border"),
    ("--border-accent", "border_accent"), ("--text-primary", "text_primary"),
    ("--text-secondary", "text_secondary"), ("--text-muted", "text_muted"),
    ("--accent", "accent"), ("--accent-dark", "accent_dark"),
    ("--success", "success"), ("--success-bg", "success_bg"),
    ("--success-border", "success_border"), ("--warning", "warning"),
    ("--warning-bg", "warning_bg"), ("--warning-border", "warning_border"),
    ("--error", "error"), ("--error-bg", "error_bg"),
    ("--error-border", "error_border"), ("--btn-bg", "btn_bg"),
    ("--btn-hover", "btn_hover"), ("--chip-bg", "chip_bg"),
    ("--chip-border", "chip_border"), ("--chip-text", "chip_text"),
]

# トークンをまとまりごとに見せるための分類
TOKEN_GROUPS = [
    ("背景",       ["--bg-app", "--bg-sidebar", "--bg-card", "--bg-card-inner", "--bg-log"]),
    ("枠線",       ["--border", "--border-accent"]),
    ("文字",       ["--text-primary", "--text-secondary", "--text-muted"]),
    ("アクセント", ["--accent", "--accent-dark"]),
    ("状態",       ["--success", "--success-bg", "--success-border",
                    "--warning", "--warning-bg", "--warning-border",
                    "--error", "--error-bg", "--error-border"]),
    ("ボタン",     ["--btn-bg", "--btn-hover"]),
    ("チップ",     ["--chip-bg", "--chip-border", "--chip-text"]),
]


# ---------------------------------------------------------------------------
# 情報源の読み取り
# ---------------------------------------------------------------------------
def load_themes(include_user: bool = True) -> dict[str, dict]:
    """theme.py から PRESET_THEMES をリテラルとして取り出す。

    theme.py は streamlit を import するのでそのままでは読み込めない。
    AST から辞書リテラルだけを拾う。

    include_user のとき、UI のカラーピッカーで保存された
    `models/.user_themes.json` のテーマも後ろに足す。
    テーマを増やしたいときはコードを触らずここから増やすのが正規のルート
    （プリセットの追加・改名は theme.py を直接編集する）。
    """
    tree = ast.parse(THEME_PY.read_text(encoding="utf-8"))
    presets = None
    for node in tree.body:
        targets = getattr(node, "targets", []) or [getattr(node, "target", None)]
        for t in targets:
            if isinstance(t, ast.Name) and t.id == "PRESET_THEMES":
                presets = ast.literal_eval(node.value)
    if presets is None:
        raise SystemExit("PRESET_THEMES が theme.py に見つかりません")

    if include_user:
        user_path = ROOT / "models" / ".user_themes.json"
        if user_path.exists():
            try:
                user = json.loads(user_path.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"  ⚠ ユーザーテーマを読めませんでした（無視します）: {e}")
                user = {}
            for name, colors in user.items():
                # 変数が欠けているものはプレビューを壊すので飛ばす
                missing = {k for _, k in VAR_MAP} - set(colors)
                if missing:
                    print(f"  ⚠ ユーザーテーマ「{name}」は変数が不足のため除外: "
                          f"{', '.join(sorted(missing))}")
                    continue
                presets[f"👤 {name}"] = colors
    return presets


def load_app_css() -> str:
    """main.py 先頭の <style> ブロックを取り出す。"""
    src = MAIN_PY.read_text(encoding="utf-8")
    m = re.search(r"<style>(.*?)</style>", src, re.S)
    if not m:
        raise SystemExit("main.py に <style> ブロックが見つかりません")
    css = m.group(1)
    # 外部フォントの取得はプレビューでは行わない（ネットワークを使わせない）
    css = re.sub(r"@import url\([^)]*\);", "", css)
    return css.strip()


# ---------------------------------------------------------------------------
# HTML 生成
# ---------------------------------------------------------------------------
def theme_class_css(themes: dict[str, dict]) -> str:
    """テーマごとに CSS 変数を持つクラスを作る。

    :root ではなくクラスに載せることで、1 ページに 4 テーマを同時に置ける。
    """
    out = []
    for i, t in enumerate(themes.values()):
        decls = " ".join(f"{var}: {t[key]};" for var, key in VAR_MAP)
        out.append(f".t{i} {{ {decls} }}")
    return "\n".join(out)


PAGE_CSS = """
* { box-sizing: border-box; }
body {
  margin: 0;
  padding: 20px;
  background: #6b7280;
  font-family: 'IBM Plex Sans', system-ui, -apple-system, sans-serif;
}
.ds-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 14px;
}
.ds-panel {
  background: var(--bg-app);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 16px;
  overflow-x: auto;
}
.ds-panel-title {
  font-size: .7rem;
  letter-spacing: .06em;
  text-transform: uppercase;
  color: var(--text-muted);
  margin-bottom: 12px;
  font-family: 'JetBrains Mono', ui-monospace, monospace;
}
.ds-note {
  color: #f3f4f6;
  font-size: .85rem;
  margin: 0 0 14px;
  line-height: 1.6;
}
.ds-note code {
  background: rgba(0,0,0,.35);
  padding: 1px 6px;
  border-radius: 4px;
  font-family: 'JetBrains Mono', ui-monospace, monospace;
}
/* トークン一覧 */
.tok-group { margin-bottom: 14px; }
.tok-group h4 {
  margin: 0 0 6px;
  font-size: .74rem;
  color: var(--text-secondary);
  font-weight: 600;
}
.tok-row {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 3px 0;
  font-size: .72rem;
  font-family: 'JetBrains Mono', ui-monospace, monospace;
}
.tok-chip {
  width: 22px; height: 22px;
  border-radius: 5px;
  border: 1px solid rgba(128,128,128,.5);
  flex: none;
}
.tok-name  { color: var(--text-secondary); flex: 1; }
.tok-value { color: var(--text-muted); }
"""


def page(card: str, title: str, note: str, themes: dict[str, dict],
         app_css: str, body_for: callable) -> str:
    """1 コンポーネントを 4 テーマ分並べたページを組み立てる。"""
    panels = []
    for i, name in enumerate(themes):
        # data-theme-name は書き戻し（tools/apply_design_tokens.py）が
        # 「どのパネルがどのテーマか」を機械的に判定するための目印。
        # 見出しテキストは編集される可能性があるので、属性側を正とする。
        panels.append(
            f'<div class="ds-panel t{i}" data-theme-index="{i}" '
            f'data-theme-name="{html.escape(name, quote=True)}">'
            f'<div class="ds-panel-title">{html.escape(name)}</div>'
            f'{body_for(i, name)}'
            f'</div>'
        )
    return f"""{card}
<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
{app_css}
{PAGE_CSS}
{theme_class_css(themes)}
</style>
</head>
<body>
<p class="ds-note">{note}</p>
<div class="ds-grid">
{''.join(panels)}
</div>
</body>
</html>
"""


def card(group: str, name: str, subtitle: str, width: int = 900, height: int = 520) -> str:
    return (f'<!-- @dsCard group="{group}" name="{name}" '
            f'subtitle="{subtitle}" width="{width}" height="{height}" -->')


# --- 各コンポーネントの中身 -------------------------------------------------
def body_tokens(themes: dict[str, dict]):
    def inner(i: int, name: str) -> str:
        t = themes[name]
        by_var = dict(VAR_MAP)
        groups = []
        for label, vars_ in TOKEN_GROUPS:
            rows = []
            for v in vars_:
                val = t[by_var[v]]
                rows.append(
                    f'<div class="tok-row">'
                    f'<span class="tok-chip" style="background:{val}"></span>'
                    f'<span class="tok-name">{v}</span>'
                    f'<span class="tok-value">{val}</span>'
                    f'</div>'
                )
            groups.append(f'<div class="tok-group"><h4>{label}</h4>{"".join(rows)}</div>')
        return "".join(groups)
    return inner


def body_section_head(i, name):
    return ('<div class="section-head"><h3>🔭 推論して結果を見る</h3></div>'
            '<div class="section-head"><h3>📊 mAP を同じ条件で測って比べる</h3></div>')


def body_badges(i, name):
    return ('<div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center;">'
            '<span class="badge-ok">ONLINE</span>'
            '<span class="badge-warn">RUNNING</span>'
            '<span class="badge-err">OFFLINE</span>'
            '</div>')


def body_pipeline(i, name):
    steps = [
        ("complete", "📝", "STEP 1", "アノテーション", "✅"),
        ("complete", "📁", "STEP 2", "データ取込", "✅"),
        ("active",   "🚀", "STEP 3", "モデル学習", "⏳"),
        ("pending",  "🔭", "STEP 4", "推論・評価", "○"),
    ]
    parts = []
    for j, (cls, icon, label, nm, mark) in enumerate(steps):
        parts.append(
            f'<div class="pf-step {cls}">'
            f'<div class="pf-label">{label}</div>'
            f'<div class="pf-name">{icon} {nm}</div>'
            f'<div class="pf-icon">{mark}</div>'
            f'</div>'
        )
        if j < len(steps) - 1:
            parts.append('<div class="pf-arrow">→</div>')
    return '<div class="pipeline-flow">' + "".join(parts) + '</div>'


def body_step_banner(i, name):
    return ('<div class="step-banner">'
            '<div class="sb-title">🚀 STEP 3: モデル学習</div>'
            '<div class="sb-prev">← 前のステップ: ✅ data.yaml が 3 件あります</div>'
            '<div class="sb-desc">→ ここでやること: モデルサイズ・学習パラメータを設定して学習開始</div>'
            '</div>')


def body_metric_grid(i, name):
    items = [("モデル", "yolo11s.pt"), ("エポック数", "100"), ("バッチ", "Auto"),
             ("imgsz", "640"), ("patience", "50"), ("optimizer", "auto"),
             ("lr0", "0.01"), ("AMP", "ON")]
    cells = "".join(
        f'<div class="mg-item"><div class="mg-label">{l}</div>'
        f'<div class="mg-value">{v}</div></div>' for l, v in items
    )
    return f'<div class="metric-grid">{cells}</div>'


def body_topic_card(i, name):
    return ('<div class="topic-card">'
            '<div class="tc-icon">🧭</div>'
            '<div class="tc-title">タスクの選び方</div>'
            '<div class="tc-body">物体の位置を四角で囲みたいなら detect、'
            '形を正確に取りたいなら segment を選びます。</div>'
            '<div class="tc-sub">対応タスク</div>'
            '<span class="tc-chip">detect</span>'
            '<span class="tc-chip">segment</span>'
            '<span class="tc-chip">obb</span>'
            '<span class="tc-chip">classify</span>'
            '<span class="tc-chip">pose</span>'
            '</div>')


def body_link_card(i, name):
    return ('<div class="link-card">'
            '<div class="lc-title"><a href="#">Ultralytics ドキュメント</a></div>'
            '<div class="lc-desc">学習パラメータの一次情報</div>'
            '</div>'
            '<div class="link-card">'
            '<div class="lc-title"><a href="#">CVAT ショートカット</a></div>'
            '<div class="lc-desc">アノテーション作業を速くする</div>'
            '</div>')


def body_sidebar_stat(i, name):
    return ('<div style="background:var(--bg-sidebar);padding:12px 14px;border-radius:8px;'
            'border:1px solid var(--border);">'
            '<div class="sidebar-stat"><span class="ss-label">📂 データセット</span>'
            '<span class="ss-value">3</span></div>'
            '<div class="sidebar-stat"><span class="ss-label">🤖 学習済みモデル</span>'
            '<span class="ss-value">7</span></div>'
            '<div class="sidebar-stat"><span class="ss-label">📋 推論結果</span>'
            '<span class="ss-value">128</span></div>'
            '</div>')


def body_log_area(i, name):
    log = ("Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances\n"
           "  1/100     4.21G     1.842      2.104      1.317         18\n"
           "  2/100     4.21G     1.655      1.702      1.245         22\n"
           "                 Class     Images  Box(P      R      mAP50\n"
           "                   all        149      0.612  0.548     0.571")
    return f'<div class="log-area">{html.escape(log)}</div>'


def body_buttons(i, name):
    return ('<div style="display:flex;gap:10px;flex-wrap:wrap;">'
            '<div class="stButton"><button>▶ 推論実行</button></div>'
            '<div class="stButton"><button>📊 評価する</button></div>'
            '<div class="stButton"><button>🗑 削除</button></div>'
            '</div>'
            '<p style="color:var(--text-muted);font-size:.72rem;margin:10px 0 0;">'
            'ホバーで --btn-hover に変化します</p>')


# ---------------------------------------------------------------------------
def main() -> None:
    themes = load_themes()
    app_css = load_app_css()

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    (OUT_DIR / "tokens").mkdir(parents=True)
    (OUT_DIR / "components").mkdir(parents=True)

    specs = [
        ("tokens/colors.html", "Colors", "カラートークン",
         "26 変数 × 4 テーマ",
         "配色トークンの一覧です。定義は <code>app/ui/theme.py</code> の "
         "<code>PRESET_THEMES</code>。タブ側では必ず <code>var(--...)</code> で参照します。",
         body_tokens(themes), 1000, 900),

        ("components/section-head.html", "Components", "セクション見出し",
         "タブ内の区切り",
         "タブ内のセクションの始まりを示す帯。<code>st.markdown</code> は呼び出しごとに"
         "独立した DOM になるため、開きタグと閉じタグを分けて中身を囲むことはできません。"
         "1 回の呼び出しで完結させています。",
         body_section_head, 900, 360),

        ("components/badges.html", "Components", "バッジ",
         "ok / warn / err",
         "サービス状態や実行状態を示す小さなラベル。",
         body_badges, 900, 260),

        ("components/pipeline-flow.html", "Components", "パイプラインフロー",
         "complete / active / pending",
         "ヘッダーに常時出る STEP1→4 の進行表示。<code>active</code> は枠線が明滅します。",
         body_pipeline, 1000, 400),

        ("components/step-banner.html", "Components", "ステップバナー",
         "前の状態 + このタブでやること",
         "各タブの先頭に置く案内。前のステップの状況と、ここで何をするかを 1 か所で示します。",
         body_step_banner, 1000, 360),

        ("components/metric-grid.html", "Components", "指標の並び",
         "幅に応じて折り返す",
         "表示専用の指標。<code>st.columns()</code> は狭い画面でも等分のままなので、"
         "flex-wrap で折り返すこちらを使います。",
         body_metric_grid, 1000, 420),

        ("components/topic-card.html", "Components", "トピックカード",
         "アイコン + 本文 + チップ",
         "トピックスタブの解説カードとタグ表示。",
         body_topic_card, 1000, 520),

        ("components/link-card.html", "Components", "リンクカード",
         "外部ドキュメントへの導線",
         "公式ドキュメントなどへのリンク。",
         body_link_card, 900, 340),

        ("components/sidebar-stat.html", "Components", "サイドバー統計",
         "ラベル + 数値",
         "サイドバー最上部の「現在の状態」。",
         body_sidebar_stat, 900, 320),

        ("components/log-area.html", "Components", "ログ表示",
         "等幅・スクロール",
         "学習ログの表示領域。実際の学習中ログは <code>components.html()</code> の iframe で"
         "描画しており、iframe は CSS 変数を継承しないため色を直接埋め込んでいます。",
         body_log_area, 1000, 420),

        ("components/buttons.html", "Components", "ボタン",
         "既定 / ホバー",
         "<code>.stButton > button</code> への上書き。Streamlit の DOM 構造に依存するため、"
         "バージョンを上げたときは確認が必要です。",
         body_buttons, 900, 300),
    ]

    for path, group, name, subtitle, note, body, w, h in specs:
        out = OUT_DIR / path
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            page(card(group, name, subtitle, w, h), name, note, themes, app_css, body),
            encoding="utf-8",
        )
        print(f"  {path}")

    print(f"\n✅ {len(specs)} ファイルを {OUT_DIR} に生成しました"
          f"（テーマ {len(themes)} 種 / CSS {len(app_css)} 文字）")


if __name__ == "__main__":
    main()
