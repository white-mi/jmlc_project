"""
DS-слой ЭНЕРГЕТИКИ (4-я отрасль, N=30 — самая полная панель). Структурная модель
ПОДКЛЮЧЕНА (Фаза C) и ЧИСТАЯ двухкомпонентная: генерация×РСВ + мощность×КОМ, БЕЗ
хардкод-интерсептов на теплосегмент (k поглощает — неидеально для теплоёмких).

Честный результат (закрепляем, не «улучшаем»): чистая структурная (~11.9%) НЕ бьёт
learned/persistence (energy — гладкий растущий тренд); но в conformal структурная
покрывает (использует контемпоральные физданные), а stale-autoregression — нет.
"""

import numpy as np
import pytest

import osl_panel
import osl_models as Mo
import osl_walkforward as W
import conformal_split as C

ISSUERS = {"РусГидро", "Мосэнерго", "ОГК-2", "ТГК-1", "Эл5-Энерго", "Юнипро"}


def _en():
    return [r for r in osl_panel.load_panel("energy") if r.has_target and r.period_end]


def _need(rows):
    if not rows:
        pytest.skip("energy-панель пуста")


# ---------- панель ----------


def test_energy_panel_loads():
    rows = _en()
    _need(rows)
    assert len(rows) >= 28, f"мало строк: {len(rows)}"
    assert ISSUERS <= {r.issuer for r in rows}
    assert all(r.industry == "energy" for r in rows)
    assert "Интер РАО" not in {r.issuer for r in rows}, "Интер РАО (сбыт) должен быть исключён"


def test_energy_issuers_5_periods():
    rows = _en()
    _need(rows)
    by = {}
    for r in rows:
        by.setdefault(r.issuer, set()).add(r.period)
    thin = {i: len(p) for i, p in by.items() if len(p) < 3}
    assert not thin, f"эмитенты с <3 периодов: {thin}"


def test_energy_revenue_positive_rub():
    rows = _en()
    _need(rows)
    for r in rows:
        assert r.target_bn > 0 and r.revenue_currency == "RUB"


def test_energy_required_prices_present():
    rows = _en()
    _need(rows)
    for r in rows:
        for k in ("electricity_rsv", "capacity_kom", "usd_rub"):
            assert r.prices.get(k) is not None, f"{r.issuer} {r.period}: нет цены {k}"


# ---------- структурная (Фаза C) ----------


def test_energy_structural_predicts():
    rows = _en()
    _need(rows)
    pred = Mo.StructuralOSL().fit(rows).predict(rows)
    assert np.isfinite(pred).sum() >= 24, "структурная должна покрывать почти все строки энергетики"


def test_energy_raw_unit_conversion():
    """SMOKE: двухкомпонентная формула + конверсия КОМ ₽/МВт·мес → ₽/ГВт·год (×12×1000).
    Проверяем _raw против пересчёта из самих панельных значений (robust к правкам данных).
    Закрепляет, что capacity-leg НЕ на 4 порядка ниже (главный unit-риск ревью)."""
    rows = _en()
    _need(rows)
    m = Mo.StructuralOSL()
    r = next(x for x in rows if x.issuer == "Юнипро" and x.period == "2024FY")
    gen, cap = r.volumes["vol_generation_twh"], r.volumes["vol_capacity_gw"]
    rsv, kom = r.prices["electricity_rsv"], r.prices["capacity_kom"]
    expected = (gen * 1e6 * rsv + cap * kom * 12 * 1000) / 1e9  # млрд ₽, other=0
    raw = m._raw(r)
    assert raw is not None and abs(raw - expected) < 0.5, f"_raw={raw} vs ожид {expected}"
    # capacity-leg существенна (не схлопнулась из-за пропущенной конверсии)
    cap_leg = cap * kom * 12 * 1000 / 1e9
    assert cap_leg > 20, f"capacity-leg подозрительно мал ({cap_leg}) — проверь ×12×1000"


def test_energy_dispatch_not_via_metallurgy():
    """energy-эмитентов нет в M.PROFILES → без диспетча _raw вернул бы None. Диспетч энергетики
    даёт конечный прогноз — значит ветка industry=='energy' срабатывает раньше M.PROFILES-гейта."""
    rows = _en()
    _need(rows)
    m = Mo.StructuralOSL()
    assert all(r.issuer not in __import__("osl_metallurgy").PROFILES for r in rows)
    assert any(m._raw(r) is not None for r in rows)


# ---------- walk-forward (честный результат) ----------


def test_energy_base_is_structural():
    rows = _en()
    _need(rows)
    preds, _ = W.walk_forward(rows, Mo.MODELS)
    assert W._pick_base(preds) == "structural_osl"
    summary, _ = W.evaluate(preds)
    assert summary["structural_osl"]["mape"] is not None


def test_energy_pure_structural_not_inflated():
    """ЧЕСТНОСТЬ-guard: чистая двухкомпонентная структурная даёт ~11–13% (как у др. отраслей),
    НЕ ~8% (это было бы признаком вернувшихся хардкод-интерсептов на тепло). Защита от
    «тихой подгонки» обратно к раздутому фиту."""
    rows = _en()
    _need(rows)
    summary, _ = W.evaluate(W.walk_forward(rows, Mo.MODELS)[0])
    mape = summary["structural_osl"]["mape_common"]
    assert (
        mape is not None and mape > 9.5
    ), f"структурная MAPE {mape} подозрительно низка — вернулись интерсепты на тепло?"


# ---------- conformal ----------


def test_energy_conformal_structural():
    """Структурная в conformal покрывает (использует контемпоральные цены/объёмы)."""
    rows = osl_panel.load_panel("energy")
    _need([r for r in rows if r.has_target])
    res = C.temporal_holdout(rows, Mo.MODELS["structural_osl"])
    assert res["q"] is not None and res["q"] > 0
    assert res["n_calib"] >= 4 and res["n_test"] >= 6
