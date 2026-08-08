# extensions — 別リポジトリの道具を持ち込む場所

ここに `git clone` すると、Streamlit UI にその道具専用のタブが増えます。
本体（`app/`）を大きくせずに機能を足していくための仕組みです。

> **対応させる側の手順は [INTEGRATION.md](INTEGRATION.md) にまとまっています。**
> 取り込みたいリポジトリにあのファイルをコピーして、そこで作業すれば対応できます
> （detection_dev_ui 側のコードを読む必要はありません）。
>
> ```bash
> cp extensions/INTEGRATION.md /path/to/対応させたいリポジトリ/
> ```

```bash
cd extensions
git clone <拡張リポジトリの URL>
# ブラウザを再読み込みするとタブが増えます
```

clone したものは git 管理外です（`.gitignore` 済み）。
この README と、`app/ext_presets/` にある既定の設定ファイルだけを追跡しています。

## 3 つの取り込み方

| kind | 何をするか | 向いているもの |
|---|---|---|
| `streamlit` | 拡張の `render()` をタブの中で直接描く | **いちばん良い形**。この UI に溶け込む |
| `command` | CLI として実行し、標準出力を画面に出す | 既存のスクリプト群 |
| `desktop` | ホストで動かす手順とコマンドを表示する | Tkinter などの画面つき GUI |

### なぜ `desktop` は「表示するだけ」なのか

この UI は Docker コンテナの中で動いていて、**画面（X ディスプレイ）を持ちません**。
Tkinter のウィンドウをブラウザに出すことはできないため、
ホスト側で実行してもらう案内に留めています。

`data/` `models/` `predictions/` はホストのディレクトリをそのままマウントしているので、
ホストで動かしても同じファイルを触れます。

## 設定は「拡張リポジトリ側」に置く

マニフェストが書いているのは**そのツールの引数やエントリポイント**です。
それは向こうのリポジトリで変わるので、**定義も向こうに置く**のが原則です。
本体側に置くと、向こうを更新したときに黙って壊れ、
気づくのは誰かがボタンを押したときになります。

拡張リポジトリの中に、連携用のディレクトリを 1 つ作ってください。

```
<拡張リポジトリ>/
├── extension/
│   ├── extension.json      ← マニフェスト
│   └── app.py              ← Streamlit 用の render()（任意）
├── scripts/
└── ...
```

探す順番は次のとおりです。

| 順 | 場所 | 用途 |
|---|---|---|
| ① | `extension/extension.json` | **標準**。連携用の資材をまとめる |
| ① | `.dev_ui/extension.json` | 隠しディレクトリにしたい場合 |
| ① | `extension.json`（直下） | 小さい拡張向けの簡易形 |
| ② | 本体の `app/ext_presets/<名前>.json` | **自分で変更できないリポジトリ**用の逃げ道 |
| ③ | ファイル構成からの推測 | 最後の手段 |

②③ で表示されているときは、タブに「📄 雛形を書き出す」ボタンが出ます。
押すと今の表示内容をもとに `extension/extension.json` が作られるので、
中身を直して**拡張リポジトリ側でコミット**してください。
以降は clone するだけで正しいタブが出ます。

### 書き方

```jsonc
{
  "name": "アノテーション整理",          // タブに出る名前
  "icon": "🗂",                          // タブのアイコン
  "description": "データセットの統合・分割・形式変換",
  "url": "https://github.com/...",       // 任意
  "requirements": ["pyyaml"],            // 足りなければ画面で知らせる
  "actions": [
    {
      "label": "データセットを分割する",
      "kind": "command",
      "command": ["python3", "scripts/split_dataset.py",
                  "--src", "{data_dir}/{dataset}", "--out", "{out}"],
      "inputs": ["dataset", "out"],      // 画面に入力欄が出る
      "note": "補足説明（任意）"
    },
    {
      "label": "GUI を起動する",
      "kind": "desktop",
      "command": ["python3", "dataset_manager_gui.py"]
    }
  ]
}
```

### プレースホルダ

`command` の中に書くと、実行時に実際のパスへ置き換わります。

| 書き方 | 中身 |
|---|---|
| `{data_dir}` | データセットの置き場 |
| `{models_dir}` | モデルの置き場 |
| `{predictions_dir}` | 推論結果の置き場 |
| `{ext_dir}` | この拡張のディレクトリ |
| `{任意の名前}` | `inputs` に書いた名前は入力欄の値に置き換わる |

## Streamlit に溶け込ませる（推奨）

拡張の側に、`st.*` を呼ぶ関数を 1 つ用意するだけです。
相手のリポジトリに 1 ファイル足すだけで済み、本体のコードは触りません。

```python
# extensions/<name>/extension.py
import streamlit as st

def render():
    st.markdown("### データセットを分割する")
    ratio = st.slider("val の割合", 0.05, 0.5, 0.2)
    if st.button("実行"):
        from dsm import ops          # 元のリポジトリのロジックをそのまま使う
        ops.split(...)
        st.success("完了しました")
```

```jsonc
{ "actions": [{ "label": "分割", "kind": "streamlit", "module": "extension" }] }
```

**ロジックと画面が分かれている拡張ほど、この形にしやすい**です。
`anno_dataset_tools` の `dsm/ops.py` のように処理が独立していれば、
Tkinter の画面を捨てて `render()` を書き直すだけで、この UI の中で完結します。

## 注意

- 拡張のコードは、**そのタブを開いたとき**（`streamlit`）または
  **ボタンを押したとき**（`command`）に実行されます。
  一覧を作るだけの段階では何も実行しません。
- 拡張の中で例外が出ても本体は落ちません。そのタブにエラーが出るだけです。
- 拡張が必要とするパッケージは、コンテナ内で使うなら
  `app/requirements.txt` に足して `docker compose build streamlit_app` してください。
  ホストで動かすぶんには不要です。
