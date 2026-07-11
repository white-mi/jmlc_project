"""
DS-слой ОИВ (5-я отрасль) — реальная панель region×year собственных доходов бюджетов
субъектов, learned-модели на urals×FX, walk-forward, conformal. Структурная фискальная
Q×P ОТЛОЖЕНА (региональная налоговая механика непрозрачна) → база = persistence.

Эти тесты ЗАКРЕПЛЯЮТ честный результат капстоуна — НЕ «чинить» в сторону,
противоречащую документированным выводам (learned бьёт наивное по MAPE, но различие в
пределах DM-шума; структурная → NaN; scope у каждого региона внутренне однороден).
См. docs/DS_REPORT_OIV.md.
"""

import numpy as np
import pytest

import osl_panel
import osl_models as Mo
import osl_walkforward as W
import conformal_split as C

REGIONS = {"Республика Башкортостан", "Пермский край", "Сахалинская обл.", "Татарстан"}


def _oiv():
    return [r for r in osl_panel.load_panel("oiv") if r.has_target and r.period_end]


def _need(rows):
    if not rows:
        pytest.skip("oiv-панель пуста")


# ---------- панель ----------


def test_oiv_panel_loads():
    rows = _oiv()
    _need(rows)
    assert len(rows) >= 20, f"мало строк: {len(rows)}"
    assert REGIONS <= {r.issuer for r in rows}
    assert all(r.industry == "oiv" for r in rows)


def test_oiv_regions_have_3plus_periods():
    """Walk-forward требует ≥3 периода на регион (внутренне однородный по scope ряд)."""
    rows = _oiv()
    _need(rows)
    by = {}
    for r in rows:
        by.setdefault(r.issuer, set()).add(r.period)
    thin = {i: len(p) for i, p in by.items() if len(p) < 3}
    assert not thin, f"регионы с <3 периодов: {thin}"


def test_oiv_revenue_positive_and_rub():
    rows = _oiv()
    _need(rows)
    for r in rows:
        assert r.target_bn > 0
        assert r.revenue_currency == "RUB"


def test_oiv_prices_present():
    """Каждый oiv-период видит urals/usd_rub (нефтяной фискальный драйвер, NaN-free для learned)."""
    rows = _oiv()
    _need(rows)
    for r in rows:
        for k in ("urals", "usd_rub"):
            assert r.prices.get(k) is not None, f"{r.issuer} {r.period}: нет цены {k}"


# ---------- модели ----------


def test_oiv_structural_deferred_returns_nan():
    """StructuralOSL для ОИВ отложён (нет чистой фискальной Q×P) → NaN, не молчаливый ноль."""
    rows = _oiv()
    _need(rows)
    pred = Mo.StructuralOSL().fit(rows).predict(rows)
    assert np.all(
        np.isnan(pred)
    ), "structural не должен предсказывать ОИВ (фискальная Q×P отложена)"


def test_oiv_base_falls_back_to_persistence():
    rows = _oiv()
    _need(rows)
    preds, _ = W.walk_forward(rows, Mo.MODELS)
    assert W._pick_base(preds) == "persistence"


# ---------- walk-forward (честный headline-результат) ----------


def test_oiv_learned_beats_persistence():
    """Регуляризованно-линейная модель на urals×FX ТОЧНЕЕ наивного persistence (skill>0).
    Содержательно: налог на прибыль нефтянки движется с ценой нефти. Закрепляем знак."""
    rows = _oiv()
    _need(rows)
    preds, _ = W.walk_forward(rows, Mo.MODELS)
    summary, common = W.evaluate(preds)
    assert len(common) >= 15, f"common занулён: {len(common)}"
    ridge = summary["ridge"]["mape_common"]
    persistence = summary["persistence"]["mape_common"]
    assert ridge is not None and persistence is not None
    assert ridge < persistence, f"ridge {ridge:.2f} не бьёт persistence {persistence:.2f}"


def test_oiv_no_significant_dm():
    """Честный инвариант: превосходство learned — в пределах DM-шума (p>0.05), как у всех
    отраслей. Не выдавать за статистически значимое (даже на бо́льшей, чем металлургия, панели)."""
    rows = _oiv()
    _need(rows)
    preds, _ = W.walk_forward(rows, Mo.MODELS)
    summary, _ = W.evaluate(preds)
    for name in ("ridge", "elasticnet"):
        p = summary[name]["dm_p_vs_struct"]
        if p is not None:
            assert p > 0.05, f"{name}: DM p={p:.3f} стал значим — пересмотри нарратив DS_REPORT_OIV"


# ---------- conformal ----------


def test_oiv_conformal_produces_interval():
    """Conformal даёт квантиль; test-хвост честно крошечный (n=2) — не выдаём за надёжное покрытие."""
    rows = osl_panel.load_panel("oiv")
    _need([r for r in rows if r.has_target])
    res = C.temporal_holdout(rows, Mo.MODELS["ridge"])
    assert res["q"] is not None and res["q"] > 0, "conformal не дал квантиль"
    assert res["n_calib"] >= 3, f'мал калибровочный фолд: {res["n_calib"]}'
