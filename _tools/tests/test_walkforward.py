"""
Тесты walk-forward валидации (osl_walkforward.py): anti-leakage сплита, корректность
метрик/skill/Diebold-Mariano, генерация отчёта. Core-зависимости (numpy/scipy/sklearn).
"""

import numpy as np
import pytest

import osl_panel
import osl_models as Mo
import osl_walkforward as W


def _rows():
    return [r for r in osl_panel.load_panel("metallurgy") if r.has_target and r.period_end]


def _need(rows):
    if len(rows) < 8:
        pytest.skip("панель мала")


# ---------- anti-leakage сплита ----------


def test_walkforward_train_strictly_before_test():
    """В каждом фолде ВСЕ train-строки старше тест-года (нет look-ahead)."""
    rows = _rows()
    _need(rows)
    # одна модель достаточно для проверки структуры фолдов
    preds, folds = W.walk_forward(rows, {"structural_osl": Mo.StructuralOSL})
    assert folds, "нет фолдов"
    years = sorted({r.period_end.year for r in rows})
    assert [f["test_year"] for f in folds] == years[1:]
    # expanding: n_train растёт
    ntr = [f["n_train"] for f in folds]
    assert ntr == sorted(ntr)


def test_walkforward_no_test_year_in_train():
    """Прямая проверка: для тест-года t среди train нет строк года t."""
    rows = _rows()
    _need(rows)
    years = sorted({r.period_end.year for r in rows})
    for t in years[1:]:
        train = [r for r in rows if r.period_end.year < t]
        assert all(r.period_end.year != t for r in train)


# ---------- метрики ----------


def test_metrics_for_known():
    recs = [("A", "21", 2021, 100.0, 110.0), ("B", "21", 2021, 200.0, 180.0)]
    m = W.metrics_for(recs)
    assert m["n"] == 2
    assert abs(m["mape"] - 10.0) < 1e-9  # (10% + 10%)/2
    assert abs(m["mae"] - 15.0) < 1e-9  # (10 + 20)/2


def test_metrics_ignores_nan_predictions():
    recs = [("A", "21", 2021, 100.0, np.nan), ("B", "21", 2021, 200.0, 220.0)]
    m = W.metrics_for(recs)
    assert m["n"] == 1 and abs(m["mape"] - 10.0) < 1e-9


def test_rmse_known():
    recs = [("A", "21", 2021, 100.0, 130.0), ("B", "21", 2021, 200.0, 160.0)]  # err 30, -40
    m = W.metrics_for(recs)
    assert abs(m["rmse"] - np.sqrt((30**2 + 40**2) / 2)) < 1e-9


def test_metrics_empty():
    m = W.metrics_for([])
    assert m["n"] == 0 and m["mape"] is None and m["rmse"] is None


# ---------- Diebold-Mariano ----------


def test_dm_identical_errors():
    e = np.array([0.1, 0.2, 0.3])
    stat, p = W.diebold_mariano(e, e)
    assert stat == 0.0 and p == 1.0


def test_dm_directional_sign():
    """base с большими ошибками vs модель с малыми → stat>0 (base хуже)."""
    base = np.array([0.4, 0.5, 0.6, 0.45])
    better = np.array([0.05, 0.04, 0.06, 0.05])
    stat, p = W.diebold_mariano(base, better)
    assert stat > 0 and p < 0.05


# ---------- skill / общий набор ----------


def test_common_keys_intersection():
    preds = {
        "m1": [("A", "21", 2021, 1.0, 1.1), ("B", "21", 2021, 1.0, 1.0)],
        "m2": [("A", "21", 2021, 1.0, 0.9), ("B", "21", 2021, 1.0, np.nan)],
    }
    common = W._common_keys(preds)
    assert common == {("A", "21")}  # B выпал (m2 дал NaN)


def test_skill_orientation_synthetic():
    """Модель с МЕНЬШЕЙ ошибкой, чем структурная → skill>0; с большей → skill<0."""
    preds = {
        "structural_osl": [("A", "21", 2021, 100.0, 120.0), ("B", "21", 2021, 100.0, 120.0)],  # 20%
        "better": [("A", "21", 2021, 100.0, 105.0), ("B", "21", 2021, 100.0, 105.0)],  # 5%
        "worse": [("A", "21", 2021, 100.0, 150.0), ("B", "21", 2021, 100.0, 150.0)],  # 50%
    }
    summary, _ = W.evaluate(preds)
    assert summary["better"]["skill_vs_struct"] > 0
    assert summary["worse"]["skill_vs_struct"] < 0


def test_structural_self_skill_zero():
    rows = _rows()
    _need(rows)
    preds, _ = W.walk_forward(rows, Mo.MODELS)
    summary, _ = W.evaluate(preds)
    assert abs(summary["structural_osl"]["skill_vs_struct"]) < 1e-9
    assert summary["structural_osl"]["dm_p_vs_struct"] is None


# ---------- end-to-end ----------


def test_run_writes_report(tmp_path, monkeypatch):
    rows = _rows()
    _need(rows)
    monkeypatch.setattr(W, "OUT", tmp_path)
    res = W.run("metallurgy")
    assert res is not None
    summary, md = res
    assert (tmp_path / "metallurgy.md").exists()
    assert (tmp_path / "metallurgy_metrics.json").exists()
    assert "structural_osl" in summary
    # все модели получили какие-то метрики
    for name in Mo.MODELS:
        assert summary[name]["mape"] is not None


def test_walkforward_bit_reproducible():
    """Детерминизм: два прогона walk-forward на одной панели → ИДЕНТИЧНЫЕ метрики.
    «Запусти дважды — то же число» — базовое требование воспроизводимости ML-результата
    (seed=42 / random_state=0 фиксированы; данные неизменны)."""
    rows = _rows()
    _need(rows)
    s1, _ = W.evaluate(W.walk_forward(rows, Mo.MODELS)[0])
    s2, _ = W.evaluate(W.walk_forward(rows, Mo.MODELS)[0])
    assert set(s1) == set(s2)
    for name in s1:
        for k in ("mape_common", "mae", "mape", "rmse", "skill_vs_struct", "dm_p_vs_struct"):
            assert s1[name][k] == s2[name][k], f"{name}.{k}: {s1[name][k]} != {s2[name][k]}"
