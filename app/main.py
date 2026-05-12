# =============================================================================
# MLOps 統合UI - main.py
# Streamlit + CVAT API + YOLO学習 + ClearML + FiftyOne
# =============================================================================
from __future__ import annotations

import json
import os
import subprocess
import threading
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Optional
from cvat_sdk.core.client import Client
import zipfile

import streamlit as st

# ---------------------------------------------------------------------------
# 定数・環境変数
# ---------------------------------------------------------------------------
DATA_DIR       = Path(os.getenv("DATA_DIR",       "/workspace/data"))
MODELS_DIR     = Path(os.getenv("MODELS_DIR",     "/workspace/models"))
PREDICTIONS_DIR= Path(os.getenv("PREDICTIONS_DIR","/workspace/predictions"))
CVAT_HOST      = os.getenv("CVAT_HOST",  "http://cvat-server:8080")
CVAT_USER      = os.getenv("CVAT_USERNAME","admin")
CVAT_PASS      = os.getenv("CVAT_PASSWORD","admin")
CLEARML_API    = os.getenv("CLEARML_API_HOST","http://clearml_apiserver:8008")
CLEARML_WEB    = os.getenv("CLEARML_WEB_HOST","http://localhost:8082")  # Fix: 8080→8082
FIFTYONE_PORT  = int(os.getenv("FIFTYONE_PORT","5151"))

for d in [DATA_DIR, MODELS_DIR, PREDICTIONS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Streamlit ページ設定
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="MLOps Pipeline",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# カスタム CSS (ダークテーマ / インダストリアル)
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}
code, pre, .stCode { font-family: 'JetBrains Mono', monospace; }

.stApp { background: #0d0f14; }

/* サイドバー */
[data-testid="stSidebar"] {
    background: #12151c;
    border-right: 1px solid #1e2330;
}

/* カード */
.pipeline-card {
    background: #161b26;
    border: 1px solid #1e2330;
    border-radius: 8px;
    padding: 20px;
    margin: 10px 0;
}
.pipeline-card h3 { color: #7ecff4; margin-top: 0; }

/* ステータスバッジ */
.badge-ok   { background:#1a3a2a; color:#4caf7d; border:1px solid #2d6b47;
                padding:2px 10px; border-radius:4px; font-size:.78rem; }
.badge-warn { background:#3a2a10; color:#f0a830; border:1px solid #7a5520;
                padding:2px 10px; border-radius:4px; font-size:.78rem; }
.badge-err  { background:#3a1a1a; color:#f06060; border:1px solid #7a3030;
                padding:2px 10px; border-radius:4px; font-size:.78rem; }

/* ログエリア */
.log-area {
    background: #0a0c10;
    border: 1px solid #1e2330;
    border-radius: 6px;
    padding: 12px;
    font-family: 'JetBrains Mono', monospace;
    font-size: .8rem;
    color: #8fb8d0;
    max-height: 280px;
    overflow-y: auto;
}

/* プログレスバー */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #2d7dd2, #7ecff4);
}

/* ボタン */
.stButton > button {
    background: #1a2540;
    color: #7ecff4;
    border: 1px solid #2d4a80;
    border-radius: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    letter-spacing: .05em;
    transition: all .2s;
}
.stButton > button:hover {
    background: #2d4a80;
    border-color: #7ecff4;
    color: #fff;
}
</style>
""", unsafe_allow_html=True)

# ===========================================================================
# セッションステート初期化
# ===========================================================================
defaults = {
    "training_log": [],
    "training_running": False,
    "training_progress": 0,
    "fiftyone_session": None,
    "fiftyone_port": None,
    "last_model_path": None,
    "cvat_tasks": [],
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ===========================================================================
# ヘルパー関数群
# ===========================================================================

# ---------------------------------------------------------------------------
# CVAT API クライアント
# ---------------------------------------------------------------------------
def get_cvat_client():
    """cvat-sdk の CvatClient を返す（接続失敗時は None）"""
    try:
        from cvat_sdk import make_client
        client = make_client(
            host=CVAT_HOST,
            credentials=(CVAT_USER, CVAT_PASS),
        )
        return client
    except Exception as e:
        st.error(f"CVAT接続エラー: {e}")
        return None


def fetch_cvat_tasks() -> list[dict]:
    """CVATのタスク一覧を取得する"""
    client = get_cvat_client()
    if not client:
        return []
    try:
        tasks = client.tasks.list()
        return [{"id": t.id, "name": t.name, "size": t.size,
                "status": t.status} for t in tasks]
    except Exception as e:
        st.error(f"タスク取得エラー: {e}")
        return []


def export_cvat_task_yolo(task_id: int, out_dir: Path) -> Optional[Path]:
    """
    指定タスクを YOLO 1.1 フォーマットでエクスポートし、
    out_dir に解凍したパスを返す。
    """
    zip_path = out_dir / "dataset.zip"
    
    try:
        # SDKでCVATにログイン
        with Client(url=CVAT_HOST) as client:
            client.login((CVAT_USER, CVAT_PASS))
            
            # タスクを取得
            task = client.tasks.retrieve(task_id)
            
            # データセットのZIPダウンロード（自動で進捗を待機してくれます）
            task.export_dataset(
                format_name="Ultralytics YOLO Segmentation 1.0",
                filename=str(zip_path),
                include_images=True
            )
            
        # ダウンロードしたZIPファイルを指定のディレクトリに解凍
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(out_dir)
            
        # 解凍が終わったら元のZIPファイルは削除してスッキリさせる
        zip_path.unlink()
        
        return out_dir
    
    except Exception as e:
        import streamlit as st
        st.error(f"CVATからのエクスポート中にエラーが発生しました: {e}")
        return None


# ---------------------------------------------------------------------------
# ClearML 設定
# ---------------------------------------------------------------------------
def init_clearml(project_name: str, task_name: str):
    """
    ClearML Task を初期化して返す。
    環境変数でサーバーURLを上書き済みであることが前提。
    """
    try:
        from clearml import Task
        os.environ["CLEARML_API_HOST"]   = CLEARML_API
        os.environ["CLEARML_WEB_HOST"]   = CLEARML_WEB
        task = Task.init(
            project_name=project_name,
            task_name=task_name,
            reuse_last_task_id=False,
        )
        return task
    except Exception as e:
        st.warning(f"ClearML 初期化エラー（実験追跡なし）: {e}")
        return None


# ---------------------------------------------------------------------------
# YOLO 学習ワーカー (別スレッドで実行)
# ---------------------------------------------------------------------------
def _train_worker(
    data_yaml: str,
    model_size: str,
    epochs: int,
    batch_size: int,
    project_name: str,
    run_name: str,
):
    """バックグラウンドスレッドで YOLO 学習を実行する"""
    log = st.session_state.training_log

    # ClearML タスク開始
    clearml_task = init_clearml(project_name, run_name)
    if clearml_task:
        clearml_task.connect({
            "model_size": model_size,
            "epochs": epochs,
            "batch_size": batch_size,
            "data_yaml": data_yaml,
        })
        log.append(f"[ClearML] タスク開始: {clearml_task.id}")

    try:
        from ultralytics import YOLO

        model_name = f"yolo11{model_size}.pt"   # 例: yolo11n.pt, yolo11s.pt …
        log.append(f"[YOLO] モデルロード: {model_name}")

        model = YOLO(model_name)

        # --- 学習実行 ---
        log.append("[YOLO] 学習開始...")
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            project=str(MODELS_DIR),
            name=run_name,
            exist_ok=True,
            # ClearML 統合は自動検出 (clearml が init されていれば有効)
        )

        best_model = Path(results.save_dir) / "weights" / "best.pt"
        st.session_state.last_model_path = str(best_model)
        log.append(f"[YOLO] 学習完了: {best_model}")
        st.session_state.training_progress = 100

        if clearml_task:
            clearml_task.upload_artifact("best_model", str(best_model))
            clearml_task.close()

    except Exception as e:
        log.append(f"[ERROR] {e}")

    finally:
        st.session_state.training_running = False


# ---------------------------------------------------------------------------
# FiftyOne セッション管理
# ---------------------------------------------------------------------------
def launch_fiftyone(dataset_name: str, predictions_dir: Path) -> Optional[int]:
    """
    FiftyOne データセットを作成し、Appを起動してポート番号を返す。
    既存のセッションがあれば再利用。

    Fix: remote=True → remote=False, address="0.0.0.0"
        コンテナ内で 0.0.0.0:5151 でListenさせてホストブラウザからアクセス可能にする。
    """
    try:
        import fiftyone as fo

        # 既存データセットをリセット
        if fo.dataset_exists(dataset_name):
            fo.delete_dataset(dataset_name)

        dataset = fo.Dataset(name=dataset_name)

        # predictions_dir の JSON ファイルを読み込んでサンプル追加
        json_files = list(predictions_dir.glob("*.json"))
        if not json_files:
            st.warning("predictions/ に結果JSONがありません。先に推論を実行してください。")
            return None

        samples = []
        for jf in json_files:
            with open(jf) as f:
                pred = json.load(f)

            img_path = pred.get("image_path", "")
            detections = []
            for box in pred.get("boxes", []):
                detections.append(
                    fo.Detection(
                        label=box["label"],
                        bounding_box=box["bbox_xywhn"],  # [x, y, w, h] 正規化済
                        confidence=box.get("confidence", 1.0),
                    )
                )
            sample = fo.Sample(filepath=img_path)
            sample["predictions"] = fo.Detections(detections=detections)
            samples.append(sample)

        dataset.add_samples(samples)

        # 既存セッションを閉じる
        if st.session_state.fiftyone_session:
            try:
                st.session_state.fiftyone_session.close()
            except Exception:
                pass

        # Fix: remote=False, address="0.0.0.0" でコンテナ外から直接アクセス可能に
        session = fo.launch_app(
            dataset,
            port=FIFTYONE_PORT,
            address="0.0.0.0",
            remote=False,
        )
        st.session_state.fiftyone_session = session
        st.session_state.fiftyone_port = FIFTYONE_PORT
        return FIFTYONE_PORT

    except Exception as e:
        st.error(f"FiftyOne エラー: {e}")
        return None


# ---------------------------------------------------------------------------
# YOLO 推論
# ---------------------------------------------------------------------------
def run_inference(
    model_path: str,
    image_dir: Path,
    out_dir: Path,
    conf_threshold: float = 0.25,
) -> list[Path]:
    """指定モデルで画像フォルダを推論し、結果JSONを out_dir に保存して返す"""
    try:
        from ultralytics import YOLO

        model = YOLO(model_path)
        results_list = model.predict(
            source=str(image_dir),
            conf=conf_threshold,
            save=False,
        )
        saved_jsons = []
        for res in results_list:
            img_path = res.path
            boxes = []
            if res.boxes:
                for box in res.boxes:
                    xyxy   = box.xyxy[0].tolist()
                    xywhn  = box.xywhn[0].tolist()
                    cls_id = int(box.cls[0])
                    conf   = float(box.conf[0])
                    label  = res.names[cls_id]
                    boxes.append({
                        "label": label,
                        "confidence": round(conf, 4),
                        "bbox_xyxy": [round(v, 2) for v in xyxy],
                        "bbox_xywhn": [round(v, 6) for v in xywhn],
                    })

            out_json = out_dir / (Path(img_path).stem + ".json")
            with open(out_json, "w") as f:
                json.dump({"image_path": img_path, "boxes": boxes}, f, indent=2, ensure_ascii=False)
            saved_jsons.append(out_json)

        return saved_jsons
    except Exception as e:
        st.error(f"推論エラー: {e}")
        return []


# ===========================================================================
# UI レイアウト
# ===========================================================================

# --- ヘッダー ---
st.markdown("""
<div style="border-bottom:1px solid #1e2330; padding-bottom:16px; margin-bottom:24px;">
  <h1 style="color:#7ecff4; font-family:'JetBrains Mono',monospace; font-size:1.6rem; margin:0;">
    🔬 MLOps Pipeline
  </h1>
  <p style="color:#4a6080; font-size:.85rem; margin:4px 0 0;">
    CVAT → YOLO → ClearML → FiftyOne 統合ダッシュボード
  </p>
</div>
""", unsafe_allow_html=True)

# --- サイドバー: サービス接続状態 ---
with st.sidebar:
    st.markdown("### 🖥 サービス状態")

    def check_service(url: str, name: str):
        import requests
        try:
            r = requests.get(url, timeout=3)
            ok = r.status_code < 500
        except Exception:
            ok = False
        badge = "badge-ok" if ok else "badge-err"
        status = "ONLINE" if ok else "OFFLINE"
        st.markdown(
            f'<div style="margin:4px 0">{name} '
            f'<span class="{badge}">{status}</span></div>',
            unsafe_allow_html=True,
        )

    check_service(f"{CVAT_HOST}/api/server/about", "CVAT")
    check_service(f"{CLEARML_API}/debug.ping", "ClearML API")
    check_service(f"{CLEARML_WEB}", "ClearML WebUI")

    st.markdown("---")
    st.markdown("#### 📁 ディレクトリ")
    st.code(f"data/        {DATA_DIR}\nmodels/      {MODELS_DIR}\npredictions/ {PREDICTIONS_DIR}", language="text")

    st.markdown("---")
    st.markdown("#### 🔗 クイックリンク")
    st.markdown(f"[📝 CVAT UI]({CVAT_HOST})", unsafe_allow_html=False)
    st.markdown(f"[📊 ClearML UI]({CLEARML_WEB})", unsafe_allow_html=False)
    if st.session_state.fiftyone_port:
        fo_url = f"http://localhost:{st.session_state.fiftyone_port}"
        st.markdown(f"[🔭 FiftyOne App]({fo_url})", unsafe_allow_html=False)

# ---------------------------------------------------------------------------
# タブ構成
# ---------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "① CVAT エクスポート",
    "② YOLO 学習",
    "③ 推論 & 可視化",
])

# ===========================================================================
# タブ1: CVAT エクスポート
# ===========================================================================
with tab1:
    st.markdown('<div class="pipeline-card"><h3>📤 CVATタスクエクスポート</h3>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🔄 タスク一覧を取得", use_container_width=True):
            with st.spinner("CVATからタスクを取得中…"):
                st.session_state.cvat_tasks = fetch_cvat_tasks()

    tasks = st.session_state.cvat_tasks
    if not tasks:
        st.info("「タスク一覧を取得」ボタンを押してCVATに接続してください。")
    else:
        task_options = {f"[{t['id']}] {t['name']} ({t['size']} items)": t["id"] for t in tasks}
        selected_label = st.selectbox("エクスポートするタスクを選択", list(task_options.keys()))
        selected_id = task_options[selected_label]

        export_dir_name = st.text_input(
            "エクスポート先サブディレクトリ名",
            value=f"dataset_{selected_id}_{datetime.now():%Y%m%d}",
        )

        if st.button("⬇️  YOLOフォーマットでエクスポート実行", type="primary", use_container_width=True):
            out_dir = DATA_DIR / export_dir_name
            out_dir.mkdir(parents=True, exist_ok=True)
            with st.spinner("エクスポート中…"):
                result_path = export_cvat_task_yolo(selected_id, out_dir)
            if result_path:
                st.success(f"✅ エクスポート完了\n保存先: `{result_path}`")
                # data.yaml の場所を確認・提示
                yaml_candidates = list(result_path.glob("*.yaml")) + list(result_path.glob("**/*.yaml"))
                if yaml_candidates:
                    st.info(f"🗂 data.yaml を検出: `{yaml_candidates[0]}`\nタブ②でこのパスを使用してください。")

    st.markdown('</div>', unsafe_allow_html=True)

    # マニュアルエクスポート補足
    with st.expander("💡 CVATで手動エクスポートする場合の手順"):
        st.markdown("""
1. CVAT UI (`http://localhost:8080`) を開く
2. 対象タスク → **Actions** → **Export task dataset**
3. フォーマット: **YOLO 1.1** を選択
4. ダウンロードされた ZIP を `./data/` に配置して解凍
5. 解凍後のフォルダ内 `data.yaml` のパスをタブ②に入力
        """)


# ===========================================================================
# タブ2: YOLO 学習
# ===========================================================================
with tab2:
    st.markdown('<div class="pipeline-card"><h3>🚀 YOLO 学習設定</h3>', unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        model_size = st.selectbox(
            "モデルサイズ",
            ["n", "s", "m", "l", "x"],
            index=1,
            help="n=Nano, s=Small, m=Medium, l=Large, x=XLarge",
        )
    with col_b:
        epochs = st.slider("エポック数", min_value=1, max_value=300, value=50, step=1)
    with col_c:
        batch_size = st.select_slider(
            "バッチサイズ",
            options=[4, 8, 16, 32, 64, 128],
            value=16,
        )

    data_yaml_path = st.text_input(
        "data.yaml パス (コンテナ内)",
        value=str(DATA_DIR / "dataset/data.yaml"),
        help="YOLOフォーマットの data.yaml ファイルへの絶対パスを指定してください。",
    )

    col_p, col_q = st.columns(2)
    with col_p:
        clearml_project = st.text_input("ClearML プロジェクト名", value="YOLO-Detection")
    with col_q:
        run_name = st.text_input(
            "ラン名",
            value=f"yolo11{model_size}_ep{epochs}_{datetime.now():%H%M}",
        )

    conf_threshold = st.slider("推論の確信度しきい値", 0.05, 0.95, 0.25, step=0.05)

    st.markdown("---")

    # --- 学習ボタン ---
    btn_col1, btn_col2 = st.columns([2, 1])
    with btn_col1:
        start_btn = st.button(
            "▶ 学習開始",
            type="primary",
            disabled=st.session_state.training_running,
            use_container_width=True,
        )
    with btn_col2:
        if st.session_state.training_running:
            st.markdown('<span class="badge-warn">RUNNING</span>', unsafe_allow_html=True)
        elif st.session_state.training_progress == 100:
            st.markdown('<span class="badge-ok">COMPLETED</span>', unsafe_allow_html=True)

    if start_btn:
        yaml_p = Path(data_yaml_path)
        if not yaml_p.exists():
            st.error(f"data.yaml が見つかりません: {yaml_p}")
        else:
            st.session_state.training_log = []
            st.session_state.training_progress = 0
            st.session_state.training_running = True

            t = threading.Thread(
                target=_train_worker,
                args=(data_yaml_path, model_size, epochs, batch_size, clearml_project, run_name),
                daemon=True,
            )
            t.start()
            st.rerun()

    # --- 進捗表示 ---
    if st.session_state.training_running or st.session_state.training_progress > 0:
        prog = st.session_state.training_progress
        st.progress(prog / 100, text=f"進捗: {prog}%")

        log_html = "<br>".join(st.session_state.training_log[-30:])
        st.markdown(
            f'<div class="log-area">{log_html}</div>',
            unsafe_allow_html=True,
        )

        if st.session_state.training_running:
            time.sleep(2)
            st.rerun()

    # --- 完了後: モデル選択 ---
    if st.session_state.last_model_path:
        st.success(f"✅ 最新モデル: `{st.session_state.last_model_path}`")

    st.markdown('</div>', unsafe_allow_html=True)

    # --- 既存モデル選択 ---
    with st.expander("📦 既存の学習済みモデルを選択"):
        existing_models = list(MODELS_DIR.rglob("*.pt"))
        if existing_models:
            model_labels = [str(p.relative_to(MODELS_DIR)) for p in existing_models]
            sel_model = st.selectbox("モデルファイル", model_labels)
            if st.button("このモデルを使用"):
                st.session_state.last_model_path = str(MODELS_DIR / sel_model)
                st.success(f"モデルを設定: {st.session_state.last_model_path}")
        else:
            st.info("models/ ディレクトリに .pt ファイルが見つかりません。")


# ===========================================================================
# タブ3: 推論 & 可視化
# ===========================================================================
with tab3:
    st.markdown('<div class="pipeline-card"><h3>🔭 推論 & FiftyOne 可視化</h3>', unsafe_allow_html=True)

    # --- モデル確認 ---
    current_model = st.session_state.last_model_path or ""
    model_display = current_model if current_model else "（未設定）"
    st.info(f"使用モデル: `{model_display}`\n→ タブ②で学習または既存モデルを選択してください。")

    # --- 推論対象ディレクトリ ---
    test_image_dir = st.text_input(
        "テスト画像ディレクトリ (コンテナ内)",
        value=str(DATA_DIR / "test/images"),
    )
    inf_conf = st.slider("確信度しきい値", 0.05, 0.95, 0.25, step=0.05, key="inf_conf")

    col_run, col_vis = st.columns(2)

    # --- 推論実行ボタン ---
    with col_run:
        if st.button("▶ 推論実行", type="primary", use_container_width=True,
                    disabled=not current_model):
            img_dir = Path(test_image_dir)
            if not img_dir.exists():
                st.error(f"画像ディレクトリが存在しません: {img_dir}")
            else:
                with st.spinner("推論中…"):
                    saved = run_inference(
                        current_model,
                        img_dir,
                        PREDICTIONS_DIR,
                        conf_threshold=inf_conf,
                    )
                if saved:
                    st.success(f"✅ 推論完了: {len(saved)} 件のJSONを保存")
                    with st.expander("保存されたJSON (先頭1件)"):
                        with open(saved[0]) as f:
                            st.json(json.load(f))

    # --- FiftyOne 起動ボタン ---
    with col_vis:
        fo_dataset_name = st.text_input("FiftyOneデータセット名", value="yolo_predictions")
        if st.button("🔭 FiftyOne で可視化", use_container_width=True):
            with st.spinner("FiftyOne App を起動中…"):
                port = launch_fiftyone(fo_dataset_name, PREDICTIONS_DIR)
            if port:
                fo_url = f"http://localhost:{port}"
                st.success(f"FiftyOne App が起動しました: {fo_url}")
                st.session_state.fiftyone_port = port

    # --- FiftyOne iframe 埋め込み ---
    if st.session_state.fiftyone_port:
        fo_url = f"http://localhost:{st.session_state.fiftyone_port}"
        st.markdown(f"""
<div style="margin-top:16px;">
    <p style="color:#4a6080; font-size:.85rem;">
        FiftyOne App が別ポートで起動中。同一ホストの場合は以下から直接アクセスできます。
    </p>
    <a href="{fo_url}" target="_blank" style="color:#7ecff4; font-family:'JetBrains Mono',monospace;">
        🔗 FiftyOne App を開く → {fo_url}
    </a>
</div>
<iframe src="{fo_url}" width="100%" height="600px"
    style="border:1px solid #1e2330; border-radius:8px; margin-top:12px;"
    allow="fullscreen">
</iframe>
""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # --- 推論結果 JSON ブラウザ ---
    with st.expander("📋 predictions/ の結果ファイル一覧"):
        json_files = sorted(PREDICTIONS_DIR.glob("*.json"))
        if json_files:
            for jf in json_files[:20]:  # 最大20件表示
                with st.container():
                    c1, c2 = st.columns([4, 1])
                    with c1:
                        st.text(jf.name)
                    with c2:
                        if st.button("👁", key=f"view_{jf.name}"):
                            with open(jf) as f:
                                st.json(json.load(f))
        else:
            st.info("predictions/ にJSONファイルがありません。先に推論を実行してください。")

# ---------------------------------------------------------------------------
# フッター
# ---------------------------------------------------------------------------
st.markdown("""
<div style="border-top:1px solid #1e2330; margin-top:40px; padding-top:12px;
            text-align:center; color:#2a3a50; font-size:.75rem; font-family:'JetBrains Mono',monospace;">
    MLOps Pipeline v1.0 · CVAT · YOLO · ClearML · FiftyOne · Streamlit
</div>
""", unsafe_allow_html=True)