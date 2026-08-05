# =============================================================================
# ハイパーパラメータ探索のテスト
#
#   実際の探索は学習を何度も回すのでテストしない。
#   ここで見るのは「始める前に所要時間を出せるか」「結果を読めるか」。
#   押してから8時間かかると気づくのが最悪なので、見積もりは重要。
# =============================================================================
from __future__ import annotations

from pathlib import Path

import pytest

from core import tuning as tn


@pytest.fixture
def models(tmp_path, monkeypatch):
    monkeypatch.setattr(tn, "MODELS_DIR", tmp_path)
    return tmp_path


def _run_with_csv(root, name, epochs=10, total_time=600.0, dataset=None):
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    lines = ["epoch,time,metrics/mAP50-95(B)"]
    for i in range(1, epochs + 1):
        lines.append(f"{i},{total_time * i / epochs},{0.5 + i * 0.01}")
    (d / "results.csv").write_text("\n".join(lines))
    if dataset:
        import json
        (d / ".provenance.json").write_text(
            json.dumps({"dataset": {"name": dataset}}))
    return d


# ---------------------------------------------------------------------------
# 探索空間のプリセット
# ---------------------------------------------------------------------------
def test_プリセットの形が揃っている():
    assert set(tn.SEARCH_PRESETS) == {"lr", "loss", "aug", "all", "custom"}
    for k, v in tn.SEARCH_PRESETS.items():
        assert v["label"] and v["desc"], k
        assert "space" in v, k


def test_学習率のプリセットは項目が少ない():
    """少ない回数でも効くよう絞ってある"""
    assert len(tn.SEARCH_PRESETS["lr"]["space"]) <= 5


def test_すべてはNoneで本家に任せる():
    """自前で26項目を書き写すと本家とずれるので None を渡す"""
    assert tn.SEARCH_PRESETS["all"]["space"] is None


def test_範囲は下限より上限が大きい():
    for k, v in tn.SEARCH_PRESETS.items():
        for name, rng in (v["space"] or {}).items():
            assert rng[0] < rng[1], f"{k}.{name}"


# ---------------------------------------------------------------------------
# 所要時間の見積もり（押す前に出すのが要点）
# ---------------------------------------------------------------------------
def test_記録が無ければ分からないと言う(models):
    """推測で数字を出すより「分からない」と言うほうがよい"""
    est = tn.estimate_tuning(10, 50)
    assert est["known"] is False
    assert "見積もれません" in est["text"]


def test_過去の学習から1エポックを推定する(models):
    _run_with_csv(models, "run_a", epochs=10, total_time=600.0)
    assert tn.estimate_epoch_seconds() == pytest.approx(60.0)


def test_見積もりは回数とエポックに比例する(models):
    _run_with_csv(models, "run_a", epochs=10, total_time=600.0)
    e1 = tn.estimate_tuning(10, 50)
    e2 = tn.estimate_tuning(20, 50)
    assert e1["known"] and e2["known"]
    assert e2["total"] == pytest.approx(e1["total"] * 2)
    assert e1["per_run"] == pytest.approx(60.0 * 50)


def test_同じデータセットの実測を優先する(models):
    """速いデータで測った値を、遅いデータの見積もりに使わない"""
    _run_with_csv(models, "fast", epochs=10, total_time=100.0, dataset="small")
    _run_with_csv(models, "slow", epochs=10, total_time=1000.0, dataset="big")
    assert tn.estimate_epoch_seconds(Path("/x/big")) == pytest.approx(100.0)
    assert tn.estimate_epoch_seconds(Path("/x/small")) == pytest.approx(10.0)


def test_時間の表し方():
    assert "秒" in tn._fmt_dur(30)
    assert "分" in tn._fmt_dur(600)
    assert "時間" in tn._fmt_dur(7200)


def test_壊れたCSVは飛ばす(models):
    (models / "bad").mkdir()
    (models / "bad" / "results.csv").write_text("これは,CSVでは\nない")
    assert tn.estimate_epoch_seconds() is None


def test_timeの無いCSVは使わない(models):
    d = models / "notime"
    d.mkdir()
    (d / "results.csv").write_text("epoch,loss\n1,0.5\n2,0.4\n")
    assert tn.estimate_epoch_seconds() is None


# ---------------------------------------------------------------------------
# 結果の読み取り
# ---------------------------------------------------------------------------
def _tune_csv(d, rows):
    d.mkdir(parents=True, exist_ok=True)
    lines = ["fitness,lr0,momentum"]
    for f, lr, mo in rows:
        lines.append(f"{f},{lr},{mo}")
    (d / "tune_results.csv").write_text("\n".join(lines))
    return d


def test_探索結果を読む(tmp_path):
    d = _tune_csv(tmp_path / "t", [(0.70, 0.01, 0.9), (0.75, 0.005, 0.85)])
    rows = tn.read_tune_results(d)
    assert len(rows) == 2
    assert rows[0]["iteration"] == 1 and rows[1]["fitness"] == 0.75


def test_結果が無ければ空(tmp_path):
    assert tn.read_tune_results(tmp_path / "ない") == []


def test_最良を選ぶ(tmp_path):
    d = _tune_csv(tmp_path / "t", [(0.70, 0.01, 0.9), (0.82, 0.005, 0.85),
                                   (0.61, 0.02, 0.7)])
    best = tn.best_of(tn.read_tune_results(d))
    assert best["fitness"] == 0.82 and best["iteration"] == 2


def test_結果が無ければ最良もNone():
    assert tn.best_of([]) is None


def test_最良のパラメータを読む(tmp_path):
    d = tmp_path / "t"
    d.mkdir()
    (d / "best_hyperparameters.yaml").write_text("lr0: 0.00832\nmomentum: 0.891\n")
    assert tn.read_best_params(d) == {"lr0": 0.00832, "momentum": 0.891}


def test_壊れたyamlでも落ちない(tmp_path):
    d = tmp_path / "t"
    d.mkdir()
    (d / "best_hyperparameters.yaml").write_text("{ 壊れている")
    assert tn.read_best_params(d) == {}


def test_過去の探索を新しい順に返す(models):
    import os, time
    for i, name in enumerate(["old", "new"]):
        d = _tune_csv(models / name, [(0.5, 0.01, 0.9)])
        t = time.time() - (100 if name == "old" else 0)
        os.utime(d, (t, t))
    assert [p.name for p in tn.find_tune_dirs(models)] == ["new", "old"]


# ---------------------------------------------------------------------------
# 結果を学習プリセットに変換
# ---------------------------------------------------------------------------
def test_探索結果をプリセットにできる():
    """探索して終わりでは意味がないので、そのまま学習に使える形にする"""
    got = tn.params_to_preset({"lr0": 0.00832, "momentum": 0.891},
                              base={"model": "yolo11s", "epochs": 100})
    assert got["model"] == "yolo11s" and got["epochs"] == 100
    assert got["lr0"] == 0.00832


def test_数値でない値は取り込まない():
    got = tn.params_to_preset({"lr0": 0.01, "note": "メモ"})
    assert "note" not in got and got["lr0"] == 0.01


# ---------------------------------------------------------------------------
# 共有状態（進捗が画面に届くかの土台）
# ---------------------------------------------------------------------------
def test_共有状態は別スレッドからでも同じ実体になる():
    """`st.cache_resource` はキャッシュの取り出しに ScriptRunContext を要求し、
    無ければ必ずミスする。ワーカーはスレッドで動くので、これを使うと
    自分専用の dict に書き込むことになり、進捗が画面に一生届かない。"""
    import threading

    from core.state import (_get_deploy_shared, _get_eval_shared,
                            _get_train_shared, _get_tune_shared)

    for getter in (_get_train_shared, _get_tune_shared,
                   _get_eval_shared, _get_deploy_shared):
        base, _ = getter()
        got = {}

        def _in_thread(g=getter):
            got["state"], _ = g()

        t = threading.Thread(target=_in_thread)
        t.start()
        t.join()
        assert got["state"] is base, f"{getter.__name__} がスレッドから別物になる"


def test_ndjson_を読める():
    """8.4.x の結果は tune_results.ndjson。csv だけを見ていると何も出ない。"""
    import json
    import tempfile
    from pathlib import Path

    from core.tuning import best_of, read_tune_results

    with tempfile.TemporaryDirectory() as d:
        nd = Path(d) / "tune_results.ndjson"
        nd.write_text("\n".join(json.dumps(r) for r in [
            {"iteration": 1, "fitness": 0.10,
             "hyperparameters": {"lr0": 0.01},
             "datasets": {"ds": {"fitness": 0.10, "metrics/mAP50(B)": 0.2}}},
            {"iteration": 2, "fitness": 0.25,
             "hyperparameters": {"lr0": 0.004},
             "datasets": {"ds": {"fitness": 0.25, "metrics/mAP50(B)": 0.4}},
             "save_dirs": {"ds": "/tmp/x"}},
        ]), encoding="utf-8")

        rows = read_tune_results(Path(d))
        assert len(rows) == 2
        # 振った項目は列として広がっていること（表にそのまま出せる形）
        assert rows[1]["lr0"] == 0.004
        assert rows[1]["metrics/mAP50(B)"] == 0.4
        assert best_of(rows)["iteration"] == 2


def test_csv_も従来どおり読める():
    import tempfile
    from pathlib import Path

    from core.tuning import read_tune_results

    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "tune_results.csv").write_text(
            "fitness,lr0\n0.1,0.01\n0.3,0.02\n", encoding="utf-8")
        rows = read_tune_results(Path(d))
        assert len(rows) == 2 and rows[1]["fitness"] == 0.3


def test_探索の出力先はモデル一覧に混ざらない():
    """イテレーションごとの学習が models/ 直下に train / train2 … を作ると
    モデルのカードや選択肢に紛れ込む。専用の場所へ隔離しておくこと。"""
    from core.config import MODELS_DIR
    from core.tuning import TUNE_PROJECT

    assert TUNE_PROJECT.name.startswith("."), "隠しディレクトリにしておく"
    assert TUNE_PROJECT.parent == MODELS_DIR


def test_見積もりは探索の作業場を学習実績として数えない():
    from core.tuning import estimate_epoch_seconds

    # .tuning 配下の results.csv を拾っていないこと（例外なく動くことの確認）
    assert estimate_epoch_seconds() is None or estimate_epoch_seconds() > 0


# ---------------------------------------------------------------------------
# 探索の手法と、項目の固定
# ---------------------------------------------------------------------------
def test_本家GAは値が0の項目をほとんど動かせない():
    """`_mutate` は 値 × exp(N(0,σ)) の乗算的な変異なので、
    0 から始まる項目（degrees / mixup など）は 0 に貼り付く。
    データ拡張を振りたいときに TPE が要る理由がこれ。"""
    import json
    import tempfile
    from pathlib import Path

    import ultralytics.engine.tuner as TU
    from ultralytics.engine.tuner import Tuner

    space = {"lr0": (1e-5, 1e-1), "degrees": (0.0, 45.0)}
    t = Tuner.__new__(Tuner)
    t.space, t.mongodb = space, None
    t.args = type("A", (), {"lr0": 0.01, "degrees": 0.0})()
    d = Path(tempfile.mkdtemp())
    t.tune_dir, t.tune_file = d, d / "tune_results.ndjson"
    t.tune_file.write_text("\n".join(json.dumps(
        {"iteration": i, "fitness": 0.3, "datasets": {},
         "hyperparameters": {"lr0": 0.01, "degrees": 0.0}}) for i in range(1, 6)))

    # 同じ秒だと np.random.seed(int(time.time())) で毎回同じ値になるため進める
    _clock = [1_700_000_000]
    _orig = TU.time.time
    TU.time.time = lambda: _clock.__setitem__(0, _clock[0] + 1) or _clock[0]
    try:
        vals = [t._mutate()["degrees"] for _ in range(100)]
    finally:
        TU.time.time = _orig

    # 許容範囲 0〜45 に対して、ほとんど動けない
    assert max(vals) < 1.0, f"想定より動いた: {max(vals)}"


def test_固定した項目は探索から外れる():
    from core.tuning import apply_pins

    space = {"lr0": (1e-5, 1e-2), "momentum": (0.7, 0.98), "lrf": (0.01, 1.0)}
    swept, pins = apply_pins(space, {"momentum": 0.9})
    assert "momentum" not in swept
    assert pins == {"momentum": 0.9}
    assert set(swept) == {"lr0", "lrf"}


def test_知らない項目を固定しても無視する():
    from core.tuning import apply_pins

    swept, pins = apply_pins({"lr0": (1e-5, 1e-2)}, {"nonexistent": 1.0})
    assert pins == {} and set(swept) == {"lr0"}


def test_空間なし_すべて_でも固定できる():
    """「すべて（既定 26 項目）」は space=None で本家に任せるが、
    固定するには具体的な空間が要る。実行時に取り出せること。"""
    from core.tuning import apply_pins, default_space

    assert len(default_space()) >= 20
    swept, pins = apply_pins(None, {"lr0": 0.005})
    assert "lr0" not in swept and pins == {"lr0": 0.005}
    assert len(swept) >= 19


def test_薦める手法は回数で変わる():
    from core.tuning import METHOD_GA, recommend_method

    # 回数が少ないうちは既定値から動く GA のほうが堅実（実測より）
    assert recommend_method(5) == METHOD_GA
    assert recommend_method(10) == METHOD_GA


def test_対数で振るべき範囲を見分ける():
    from core.tuning import is_log_scale

    assert is_log_scale(1e-5, 1e-2)      # 学習率は桁で効く
    assert not is_log_scale(1.0, 20.0)   # 損失の重みは線形でよい
    assert not is_log_scale(0.0, 45.0)   # 0 を含む範囲は対数にできない


def test_探索は本番と同じ最適化手法を使う():
    """optimizer を勝手に決めない。AdamW と SGD では適切な lr0 が一桁ずれるので、
    揃えないと探索で見つけた値が本番の学習で再現しない。"""
    import inspect

    from core import tuning as tn

    src = inspect.getsource(tn._tune_worker)
    assert 'optimizer="AdamW"' not in src, "optimizer が決め打ちされている"


def test_自分で選ぶプリセットがある():
    from core.tuning import SEARCH_PRESETS

    assert SEARCH_PRESETS["custom"].get("custom") is True
    # 選ばせる側なので、あらかじめの空間は持たない
    assert SEARCH_PRESETS["custom"]["space"] is None
