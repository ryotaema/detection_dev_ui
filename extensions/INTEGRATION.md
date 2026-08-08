# detection_dev_ui 連携仕様

このファイルは、**別のリポジトリで作った道具を detection_dev_ui のタブとして
使えるようにするための取り決め**です。

対応させたいリポジトリにこのファイルをコピーして、
そこで作業すれば対応できるように書いてあります。
detection_dev_ui 側のコードを読む必要はありません。

```bash
# 対応させたいリポジトリの中で
cp /path/to/detection_dev_ui/extensions/INTEGRATION.md .
```

---

## 0. 前提（先に確認してください）

- **detection_dev_ui に依存しません。** 追加するのは JSON 1 つと、必要なら
  `streamlit` だけを使う小さなファイル 1 つです。
  このリポジトリのコードを import することはありません。
- **単体で動くことを壊しません。** 既存の使い方はそのまま残します。
  足すだけで、消したり書き換えたりはしません。
- 依存の向きは常に `detection_dev_ui → こちら` の一方向です。

## 1. やること（3行）

1. リポジトリの直下に `extension/` を作る
2. `extension/extension.json` を書く
3. （任意）画面まで統合したいなら `extension/app.py` に `render()` を書く

これだけです。detection_dev_ui 側の `extensions/` に clone されると、
自動で読み取られてタブになります。

## 2. 置き場所

```
<このリポジトリ>/
├── extension/
│   ├── extension.json      ← 必須
│   └── app.py              ← 任意（画面まで統合する場合）
├── scripts/                ← 既存のものはそのまま
└── ...
```

`.dev_ui/extension.json` や、直下に `extension.json` を 1 つ置く形でも読まれますが、
**`extension/` にまとめる形を標準**とします（連携用の資材が増えても散らからないため）。

## 3. `extension.json`

```jsonc
{
  "name": "アノテーション整理",              // タブに出る名前
  "icon": "🗂",                              // タブのアイコン（絵文字 1 つ）
  "description": "データセットの統合・分割・形式変換",
  "url": "https://github.com/...",           // 任意
  "requirements": ["pyyaml", "opencv-python"],
  "actions": [ /* 下記 */ ]
}
```

| 項目 | 必須 | 内容 |
|---|---|---|
| `name` | — | タブ名。省略するとディレクトリ名 |
| `icon` | — | 絵文字。省略すると `🧩` |
| `description` | — | タブの上に出る 1 行説明 |
| `url` | — | リポジトリの URL |
| `requirements` | — | 必要なパッケージ名。足りなければ画面で知らせる |
| `actions` | ○ | できることの一覧。1 つ以上 |

### `actions` の共通項目

| 項目 | 必須 | 内容 |
|---|---|---|
| `label` | ○ | 画面に出る操作名 |
| `kind` | ○ | `streamlit` / `command` / `desktop` のいずれか |
| `note` | — | 補足説明。使い方のヒントを書く |

## 4. 3 つの `kind`

### `command` — CLI として実行する

標準出力が画面に出ます。既存のスクリプトをそのまま活かせます。

```jsonc
{
  "label": "🔍 アノテーションを検証する",
  "kind": "command",
  "command": ["python3", "scripts/anno_validator.py",
              "--labels_dir", "{labels_dir}",
              "--images_dir", "{images_dir}"],
  "inputs": ["labels_dir", "images_dir"],
  "note": "labels_dir には {data_dir}/dataset_11/labels/train のように入れます"
}
```

- `command` … **文字列のリスト**で書きます。`"python3 a.py"` のような 1 本の文字列は不可
- `inputs` … ここに書いた名前が画面の入力欄になり、`{名前}` に入ります
- 作業ディレクトリはリポジトリの直下。`PYTHONPATH` にも直下が入るので、
  自前のパッケージ（`dsm/` など）を import できます
- 既定の打ち切り時間は 900 秒

> **注意**: 中で `tkinter` を import しているスクリプトは `command` にできません。
> 実行環境に画面がないため失敗します（下記 6 を参照）。`desktop` にしてください。

### `desktop` — 画面つき GUI

Tkinter などのデスクトップ GUI 用です。
**実行はされず、利用者に「ホストで実行する手順」が表示されます。**

```jsonc
{
  "label": "🖥 統合 GUI を起動する（ホストで実行）",
  "kind": "desktop",
  "command": ["python3", "dataset_manager_gui.py"],
  "note": "Tkinter の画面なのでホストで実行します"
}
```

### `streamlit` — タブの中に直接描く（いちばん良い形）

detection_dev_ui の画面に溶け込みます。**ロジックと画面が分かれている道具ほど、
この形にしやすい**です。

```jsonc
{
  "label": "データセットを分割する",
  "kind": "streamlit",
  "module": "app",           // extension/app.py を指す（.py は書かない）
  "function": "render"       // 省略時は "render"
}
```

`module` は **`extension.json` のある場所**を起点に解決します。
`extension/extension.json` なら `"app"` → `extension/app.py` です。

## 5. プレースホルダ

`command` の中に書くと、実行時に実際のパスへ置き換わります。

| 書き方 | 中身 |
|---|---|
| `{data_dir}` | データセットの置き場 |
| `{models_dir}` | 学習済みモデルの置き場 |
| `{predictions_dir}` | 推論結果の置き場 |
| `{ext_dir}` | このリポジトリのパス |
| `{自分で決めた名前}` | `inputs` に書いた名前。画面の入力欄の値が入る |

## 6. 動く環境の制約（ここを外すと動きません）

| 制約 | 意味すること |
|---|---|
| **画面（X ディスプレイ）が無い** | `tkinter` / `cv2.imshow` は使えません。GUI は `desktop` にします |
| コンテナの中で動く | パスはコンテナ内のものです。プレースホルダを使ってください |
| Python 3.11 | これより新しい文法は使えません |
| 利用できるもの | `numpy` / `opencv-python` / `Pillow` / `pyyaml` / `pandas` / `streamlit` / `ultralytics` などは既にあります |
| それ以外の依存 | `requirements` に書いてください。画面で不足を知らせます |

`data/` `models/` `predictions/` はホストのディレクトリをそのまま見ているので、
ホストで動かしても同じファイルを扱えます。

## 7. `render()` の約束（`kind: streamlit` の場合）

```python
# extension/app.py
import streamlit as st


def render() -> None:
    """detection_dev_ui のタブの中に描かれる。"""
    st.markdown("#### データセットを分割する")
    ratio = st.slider("val の割合", 0.05, 0.5, 0.2, key="ext_split_ratio")
    if st.button("実行", key="ext_split_run"):
        from dsm import ops              # このリポジトリのロジックをそのまま使う
        ops.split(...)
        st.success("完了しました")
```

守ること:

- **引数なし・戻り値なし**の関数にします
- `st.set_page_config()` は**呼ばないでください**（本体が既に呼んでいます）
- `st.stop()` は呼ばないでください（他のタブが描かれなくなります）
- **`st.expander` を入れ子にしない**でください（例外になります）
- ウィジェットには `key` を付けてください。他のタブと衝突しないよう、
  `ext_` など固有の接頭辞を付けると安全です
- 時間のかかる処理はボタンを押したときだけ実行してください。
  `render()` は画面が再描画されるたびに毎回呼ばれます
- import は `render()` の中に書くと、読み込みが軽くなり、
  依存が足りないときも本体を巻き込みません

### 自前のパッケージを import する場合

リポジトリ直下のパッケージ（`dsm/` など）は、
**`extension/` の中からでもそのまま import できます**。

```python
# extension/app.py
def render():
    from dsm import ops        # リポジトリ直下の dsm/ が見える
```

`render()` を呼ぶあいだ、リポジトリの直下と `extension/` の両方が
import の探索先に入るようになっています（呼び終わると外れます）。
相対 import（`from ..dsm import ops`）は使えないので、
上のように絶対名で書いてください。

例外を投げても本体は落ちません（そのタブにエラーが出るだけ）が、
利用者が読める形で伝えるほうが親切です。

## 8. 受け付けられない書き方

検証で弾かれる（その操作だけ無効になり、画面に理由が出る）ものです。

| 書き方 | なぜだめか |
|---|---|
| `"kind": "cli"` など未定義の値 | `streamlit` / `command` / `desktop` のみ |
| `"command": "python3 a.py"` | 文字列のリストで書いてください |
| `"command": []` | 空は不可 |
| `kind: streamlit` で `module` 無し | どのファイルか決まりません |

`extension.json` 自体が壊れた JSON の場合は、ファイル構成からの推測に切り替わります
（意図しないタブになるので、書いたら必ず動作を確認してください）。

## 9. 対応させる手順

1. このリポジトリが何をする道具かを 1 行で書けるようにする → `description`
2. 使える操作を洗い出す
   - スクリプトの引数を `--help` などで確認する
   - **中で `tkinter` を import していないか確認する**（していれば `desktop`）
3. `extension/extension.json` を書く
4. 画面まで統合するなら `extension/app.py` に `render()` を書く
   - 既存の GUI コードは触らず、ロジックだけを呼ぶ形にする
5. 確認する（下記 10）
6. コミットする

### 迷ったときの指針

- 引数が多くて複雑 → まず `command` で `--help` を出す形にし、
  固まってから `inputs` 付きにする
- 対話的な操作（ドラッグ・キー入力）が必要 → `desktop`
- 処理が GUI から独立している → `streamlit` にする価値が高い

## 10. 確認方法

detection_dev_ui 側で確認します。

```bash
cd /path/to/detection_dev_ui/extensions
git clone <このリポジトリ>          # すでにあれば git pull
```

ブラウザを再読み込みすると、タブが増えます。タブの上部で次を確認してください。

- 設定の出所が **`リポジトリ内の extension/extension.json`** になっている
  （`同梱の既定マニフェスト` や `ファイル構成からの推測` になっていたら読まれていません）
- 操作の一覧が意図どおり
- ⚠ の警告が出ていない（出ていれば書き方の誤りです）

JSON だけ先に確かめたい場合:

```bash
python3 -c "import json; json.load(open('extension/extension.json')); print('OK')"
```

## 11. 完全な例

### 例 A: CLI が中心の道具

```json
{
  "name": "アノテーション整理",
  "icon": "🗂",
  "description": "データセットの統合・分割・形式変換・品質検証",
  "url": "https://example.com/your-org/anno-tools",
  "requirements": ["pyyaml"],
  "actions": [
    {
      "label": "🔍 アノテーションを検証する",
      "kind": "command",
      "command": ["python3", "scripts/anno_validator.py",
                  "--labels_dir", "{labels_dir}",
                  "--images_dir", "{images_dir}"],
      "inputs": ["labels_dir", "images_dir"],
      "note": "labels_dir の例: {data_dir}/dataset_11/labels/train"
    },
    {
      "label": "🖥 統合 GUI を起動する（ホストで実行）",
      "kind": "desktop",
      "command": ["python3", "dataset_manager_gui.py"],
      "note": "Tkinter の画面なのでホストで実行します"
    }
  ]
}
```

### 例 B: GUI しか無い道具

```json
{
  "name": "モザイク処理",
  "icon": "🟦",
  "description": "アノテーション漏れや写り込んだ物体をモザイクで隠す",
  "requirements": ["opencv-python", "numpy", "Pillow"],
  "actions": [
    {
      "label": "モザイクツールを起動する",
      "kind": "desktop",
      "command": ["python3", "mosaic_tool.py"],
      "note": "起動後のダイアログで data/ 配下の画像フォルダを選びます。上書き前に _backup_original/ へ自動バックアップされます"
    }
  ]
}
```

### 例 C: 画面まで統合する

```json
{
  "name": "データ整理",
  "icon": "🗂",
  "actions": [
    {"label": "分割する", "kind": "streamlit", "module": "app"},
    {"label": "変換する", "kind": "streamlit", "module": "convert",
     "function": "render_convert"}
  ]
}
```

```python
# extension/app.py
import streamlit as st


def render() -> None:
    st.markdown("#### train / val に分ける")
    ratio = st.slider("val の割合", 0.05, 0.5, 0.2, key="ext_ratio")
    src = st.text_input("対象ディレクトリ", key="ext_src")
    if st.button("分割する", key="ext_split", type="primary"):
        if not src:
            st.warning("対象ディレクトリを入力してください")
            return
        try:
            from dsm import ops
            result = ops.split(src, val_ratio=ratio)
        except Exception as e:
            st.error(f"分割に失敗しました: {e}")
            return
        st.success(f"完了: {result}")
```

---

## 補足: 移行の近道

まだ `extension/` が無いリポジトリを detection_dev_ui に clone すると、
ファイル構成から推測したタブが出ます。
そのタブの **「📄 雛形を書き出す」** を押すと `extension/extension.json` が作られるので、
中身を直してコミットするのが手早いです。
