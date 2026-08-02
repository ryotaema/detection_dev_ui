# =============================================================================
# トピックス
# =============================================================================
from __future__ import annotations

import io
import json
import os
import shutil
import time
import zipfile
from datetime import datetime
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from core import *  # noqa: F401,F403
from core import (  # noqa: F401
    _box_iou, _collect_prediction_items, _deploy_worker, _DOC_AUG, _DOC_TRAIN,
    _draw_predictions, _eval_worker, _find_image_dirs, _get_deploy_shared,
    _get_eval_shared, _get_train_shared, _iou, _MODEL_OPTS, _nuctl,
    _StdoutCapture, _train_worker, _yolo_txt_to_xyxy,
)




def render_topics() -> None:
    st.markdown('<div class="section-head"><h3>📚 ガイド</h3></div>', unsafe_allow_html=True)
    st.markdown(
        "<p style='color:var(--text-secondary);font-size:.85rem;'>物体検出 MLOps の概念・操作ガイドです。GitHub の詳細ドキュメントを参照してください。</p>",
        unsafe_allow_html=True,
    )

    _tp1, _tp2, _tp3, _tp4, _tp5 = st.tabs([
        "🧭 タスクの選び方", "📐 指標の読み方", "🩺 うまくいかないとき",
        "🗂 データセットの作り方", "🛣 今後の方針",
    ])

    # ── タスク種別の選び方 ────────────────────────────────────────────
    with _tp1:
        st.markdown("#### どのタスク種別を選ぶか")
        st.markdown(
            "| やりたいこと | タスク種別 | CVAT で付けるもの | 出力 |\n"
            "|---|---|---|---|\n"
            "| 物体の位置を四角で囲む | `detect` | 矩形 (box) | 位置とクラス |\n"
            "| 物体の形を正確に取る | `segment` | ポリゴン | 輪郭マスク |\n"
            "| 傾いた物体を囲む | `obb` | 回転付き矩形 / 4点ポリゴン | 回転した四角 |\n"
            "| 画像全体を仕分ける | `classify` | タグ | 画像ごとのクラス |\n"
            "| 関節・特徴点を取る | `pose` | ポイント | キーポイント座標 |\n"
        )
        st.info(
            "**迷ったら `detect` から。** アノテーションが最も速く、"
            "必要な情報が足りないと分かってから `segment` に移っても、"
            "矩形は自動でポリゴンに変換できます（逆はできません）。"
        )

        st.markdown("#### モデルサイズの選び方")
        st.markdown(
            "`n` → `s` → `m` → `l` → `x` の順に大きく、精度が上がり、遅く重くなります。"
        )
        st.markdown(
            "- **まず `n` か `s` で一周する** … データやラベルの問題は小さいモデルでも分かります\n"
            "- **精度が頭打ちなら大きくする** … ただし伸びは小さいことが多く、"
            "データを増やす方が効くケースが大半です\n"
            "- **実機に載せるなら速度も測る** … 「🔭 Step4」の評価で推論時間(ms)が出ます\n"
        )
        st.caption(
            "参考: 同一データセットでの実測では、大きいモデルが必ず勝つとは限りません。"
            "mAP50 がほぼ同じでも推論時間が2倍以上違うことがあるので、"
            "評価タブで両方を比べてから決めてください。"
        )

        st.markdown("#### 画像サイズ (imgsz)")
        st.markdown(
            "- 推論時間はおおむね imgsz の**2乗に比例**します（640→1280 で約4倍）\n"
            "- 小さく写る対象が多いなら上げる価値があります\n"
            "- 学習と推論で同じ値を使うのが基本です\n"
        )

    # ── 指標の読み方 ──────────────────────────────────────────────
    with _tp2:
        st.markdown("#### 検出・セグメンテーションの指標")
        st.markdown(
            "- **Precision（適合率）** … 検出したもののうち正しかった割合。"
            "低い = **誤検出が多い**\n"
            "- **Recall（再現率）** … 実際にあるもののうち見つけられた割合。"
            "低い = **見逃しが多い**\n"
            "- **IoU** … 予測と正解の重なり具合。1.0 で完全一致\n"
            "- **mAP50** … IoU 0.5 以上を正解とみなした精度。"
            "「だいたい合っている」かを見る\n"
            "- **mAP50-95** … IoU 0.5〜0.95 で平均した精度。"
            "**位置の正確さまで含めた実力**。実用ではこちらが効く\n"
            "- **top1 / top5 accuracy** … 画像分類の正答率\n"
        )
        st.info(
            "**mAP50 が高いのに mAP50-95 が低い**場合、「物体は見つけられているが"
            "枠の位置が甘い」状態です。アノテーションの枠が雑になっていないか、"
            "imgsz が小さすぎないかを疑ってください。"
        )

        st.markdown("#### Precision と Recall はトレードオフ")
        st.markdown(
            "推論時の `conf`（信頼度しきい値）を上げると Precision が上がり Recall が下がります。"
            "下げるとその逆です。用途で決めてください。"
        )
        st.markdown(
            "- **見逃したくない**（検査・安全）… conf を下げて Recall を優先\n"
            "- **誤検出を出したくない**（自動処理）… conf を上げて Precision を優先\n"
            "- **自動アノテーションの下書き** … 少し低めが便利（消す方が描くより速い）\n"
        )
        st.caption(
            "なお mAP を測るときの conf は 0.001 が正しい値です（全信頼度域の"
            "PR 曲線から計算するため）。実運用のしきい値とは別物です。"
        )

    # ── トラブルシューティング ────────────────────────────────────
    with _tp3:
        st.markdown("#### mAP が上がらない")
        st.markdown(
            "**まずデータを疑ってください。** モデルやパラメータより効きます。\n\n"
            "1. 「📁 データ管理」の **品質チェック**を実行する"
            "（幅0の枠、画像とラベルの対応漏れ、クラス分布の偏りが出ます）\n"
            "2. 「🔭 Step4」の **正解ラベルとの差分分析**で FN が多い画像を見る"
            "— アノテーション漏れが見つかることが多いです\n"
            "3. 学習枚数が足りているか（目安: 1クラスあたり最低 100〜200 枚、"
            "実用なら 1000 枚以上）\n"
            "4. train と val で撮影条件が違いすぎないか\n"
        )

        st.markdown("#### 過学習している（train は良いのに val が悪い）")
        st.markdown(
            "- データを増やす / データ拡張を強める（mosaic, mixup, hsv 系）\n"
            "- モデルを小さくする\n"
            "- `patience` を設定して早期終了させる\n"
            "- エポックを減らす\n"
        )
        st.caption("学習曲線で val の loss が下げ止まって上がり始めたら過学習のサインです。")

        st.markdown("#### 特定のクラスだけ精度が低い")
        st.markdown(
            "- **クラス別 AP** を評価タブで確認（どのクラスが悪いか特定する）\n"
            "- そのクラスの枚数が少なければ追加する（クラス分布の偏りは品質チェックで検出できます）\n"
            "- 似たクラスと混同しているなら、混同行列を確認してクラス定義自体を見直す\n"
        )

        st.markdown("#### 学習が途中で止まってしまった / 止めたい")
        st.markdown(
            "- 学習中の **⏹ 学習を停止** でエポック末に安全に止められます\n"
            "- 止めた学習は **⏯ 中断した学習を再開する** から `last.pt` の続きから再開できます\n"
            "- GPU メモリ不足で落ちる場合は `batch` か `imgsz` を下げてください\n"
        )

        st.markdown("#### 他の PC で学習した .pt が読み込めない")
        st.markdown(
            "学習元の ultralytics のバージョンがこの環境（8.4.48）と離れていると起きます。"
            "「📁 データ管理」からアップロードすると読み込み検証まで行うので、"
            "エラー内容を確認してください。"
        )

    # ── データセットの作り方 ──────────────────────────────────────
    with _tp4:
        st.markdown("#### 枚数の目安")
        st.markdown(
            "| 段階 | 枚数の目安 | 何が分かるか |\n"
            "|---|---|---|\n"
            "| お試し | 50〜100 枚 | パイプラインが通るか |\n"
            "| 最低限 | 1クラス 100〜200 枚 | 実用になるかの当たり |\n"
            "| 実用 | 1クラス 1000 枚以上 | 安定した精度 |\n"
        )

        st.markdown("#### アノテーションの質")
        st.markdown(
            "- **枠は対象にぴったり合わせる** … 甘い枠は mAP50-95 を直接下げます\n"
            "- **基準を統一する** … 隠れている部分を含めるか、どこまでを1つと数えるか。"
            "複数人で作業するなら特に重要です\n"
            "- **迷う対象のルールを決めておく** … 後から直すコストは大きいです\n"
        )
        st.info(
            "**自動アノテーションを活用してください。** 一度モデルを作れば、"
            "「🏷 Step1」から CVAT にデプロイして下書きを自動生成できます。"
            "ゼロから描くより、間違いを直す方が圧倒的に速いです。"
        )

        st.markdown("#### 学習を回す順序")
        st.markdown(
            "1. 少ないデータ・小さいモデル・少ないエポックで**一周させる**\n"
            "2. 品質チェックと差分分析で**データの問題を潰す**\n"
            "3. データを追加する（自動アノテーションで効率化）\n"
            "4. モデルサイズ・エポック・パラメータを調整する\n"
        )
        st.caption("1〜3 を回すのが最も効きます。4 は最後で構いません。")

        st.markdown("#### 途中からデータを足したいとき")
        st.markdown(
            "- 「📁 データ管理」の各データセットから**画像を追加**できます\n"
            "- 複数のデータセットを**統合**することもできます\n"
            "- 既存モデルを初期重みにして**追加学習**できます"
            "（Step3 のモデル名に `models/<run>/weights/best.pt` を指定）\n"
        )

    # ── 今後の方針 ────────────────────────────────────────────────
    with _tp5:
        st.markdown("#### このリポジトリの目的")
        st.markdown(
            "画像系の学習モデルを作るために必要な作業を、"
            "**1つの環境で完結**させることを目指しています。"
            "アノテーション・データ整備・学習・評価・モデル管理を"
            "同じ UI から扱えるようにしています。"
        )
        st.markdown("#### 設計の方針")
        st.markdown(
            "- **どの段階からでもデータを入れられる** … CVAT 経由でも、ZIP でも、"
            "画像単体でも、他 PC で作った `.pt` でも受け入れる\n"
            "- **持ち出せる** … データセットもモデルも ZIP で書き出せる\n"
            "- **壊れたデータを検出して直せる** … 品質チェックと自動修正\n"
            "- **判断材料を UI 内に出す** … 同一条件での mAP 比較、推論速度、"
            "正解ラベルとの差分\n"
        )
        st.markdown("#### 実装予定・検討中")
        st.markdown(
            "- ハイパーパラメータ探索 / k-fold 交差検証\n"
            "- train/val の再分割、クラス名の編集・統合\n"
            "- 学習に使ったデータの来歴を記録する仕組み\n"
            "- MLflow の実験比較を UI 内に埋め込む\n"
            "- `app/main.py` の分割（機能追加を続けやすくするため）\n"
        )
        st.markdown(
            "**[→ docs/overview.md をGitHubで開く]"
            "(https://github.com/ryotaema/detection_dev_ui/blob/main/docs/overview.md)**"
        )
        st.caption(
            "実装済みの機能・コード構成・設計方針・実装上の落とし穴をまとめています。"
            "新しく参加する人はまずこれを読んでください。"
        )
        st.caption(
            "※ 今後の実装予定と既知の不具合は、この環境の `docs/roadmap.md` にあります"
            "（開発方針のため Git 管理外。`SPEC.md` / `CLAUDE.md` と同じ扱い）。"
        )

    st.markdown("---")

    # ── ガイドへのリンク ──────────────────────────────────────────
    st.markdown("""
    <div class="step-banner" style="margin-bottom:20px;">
      <div class="sb-title">📖 物体検出 MLOps 学習ガイド</div>
      <div class="sb-desc">アノテーションのコツ・学習パラメータの意味・データ拡張・モデルサイズの選び方などを解説しています。</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        "**[→ docs/guide.md をGitHubで開く](https://github.com/ryotaema/detection_dev_ui/blob/main/docs/guide.md)**",
    )
    st.caption("アノテーションのコツ / mAP・IoU・過学習の解説 / 学習パラメータ / データ拡張 / モデルサイズ選択基準 を掲載しています。")

    st.markdown("---")

    # ── 公式ドキュメント ──────────────────────────────────────────
    st.markdown("#### 公式ドキュメント")
    link_col1, link_col2 = st.columns(2)
    with link_col1:
        st.markdown("""
    <div class="link-card">
      <div class="lc-title">📝 <a href="https://docs.cvat.ai/" target="_blank">CVAT 公式ドキュメント</a></div>
      <div class="lc-desc">アノテーション操作・プロジェクト管理・エクスポート形式の詳細</div>
    </div>
    <div class="link-card">
      <div class="lc-title">🚀 <a href="https://docs.ultralytics.com/" target="_blank">Ultralytics YOLO 公式ドキュメント</a></div>
      <div class="lc-desc">モデルの使い方・各学習パラメータの意味・モデルサイズ一覧</div>
    </div>
    <div class="link-card">
      <div class="lc-title">📊 <a href="https://mlflow.org/docs/latest/index.html" target="_blank">MLflow 公式ドキュメント</a></div>
      <div class="lc-desc">実験管理・モデルレジストリ・比較ビューの使い方</div>
    </div>
    """, unsafe_allow_html=True)
    with link_col2:
        st.markdown("""
    <div class="link-card">
      <div class="lc-title">🔭 <a href="https://docs.voxel51.com/" target="_blank">FiftyOne 公式ドキュメント</a></div>
      <div class="lc-desc">データセット探索・推論結果可視化・フィルタリングの使い方</div>
    </div>
    <div class="link-card">
      <div class="lc-title">📺 <a href="https://docs.streamlit.io/" target="_blank">Streamlit 公式ドキュメント</a></div>
      <div class="lc-desc">このUIで使用しているフレームワーク。ウィジェット・レイアウトの仕様</div>
    </div>
    """, unsafe_allow_html=True)


    # ---------------------------------------------------------------------------
    # フッター
    # ---------------------------------------------------------------------------
    st.markdown("""
    <div style="border-top:1px solid var(--border); margin-top:40px; padding-top:12px;
            text-align:center; color:var(--text-muted); font-size:.75rem; font-family:'JetBrains Mono',monospace;">
    detection_dev_ui v1.0 · CVAT · YOLO · MLflow · FiftyOne · Streamlit
    </div>
    """, unsafe_allow_html=True)