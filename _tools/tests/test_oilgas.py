"""
DS-слой НЕФТЕГАЗА (2-я отрасль после металлургии) — панель, модели, walk-forward,
conformal. Зеркало смысловых инвариантов металлургии, но с честными отличиями:
структурная модель пока ОТКЛЮЧЕНА для нефтегаза (нет годовых НДПИ/демпфера), поэтому
базой сравнения служит наивный persistence, а не structural_osl.

Эти тесты ЗАКРЕПЛЯЮТ ожидаемое поведение — не «чинить» их в сторону, противоречащую
документированным ограничениям (ЛУКОЙЛ-2025 разрыв ряда; structural→NaN).
"""

import numpy as np
import pytest

import osl_panel
import osl_models as Mo
import osl_walkforward as W
import conformal_split as C


def _oil():
    return [r for r in osl_panel.load_panel('oilgas') if r.has_target and r.period_end]


def _need(rows):
    if not rows:
        pytest.skip('oilgas-панель пуста')


# ---------- панель ----------

def test_oilgas_panel_loads():
    rows = _oil()
    _need(rows)
    assert len(rows) >= 16, f'мало строк: {len(rows)}'
    assert {'Роснефть', 'ЛУКОЙЛ', 'Газпром', 'Новатэк'} <= {r.issuer for r in rows}
    assert all(r.industry == 'oilgas' for r in rows)


def test_oilgas_issuers_have_3plus_periods():
    """Walk-forward требует ≥3 периода на эмитента."""
    rows = _oil()
    _need(rows)
    by = {}
    for r in rows:
        by.setdefault(r.issuer, set()).add(r.period)
    thin = {i: len(p) for i, p in by.items() if len(p) < 3}
    assert not thin, f'эмитенты с <3 периодов: {thin}'


def test_oilgas_revenue_positive_and_rub():
    rows = _oil()
    _need(rows)
    for r in rows:
        assert r.target_bn > 0
        assert r.revenue_currency == 'RUB'


def test_oilgas_prices_present():
    """Каждый нефтегаз-период видит urals/gas_eu/usd_rub (required-серии learned-моделей)."""
    rows = _oil()
    _need(rows)
    for r in rows:
        for k in ('urals', 'gas_eu', 'usd_rub'):
            assert r.prices.get(k) is not None, f'{r.issuer} {r.period}: нет цены {k}'


# ---------- модели ----------

def test_oilgas_structural_deferred_returns_nan():
    """StructuralOSL отключён для нефтегаза (нет per-year НДПИ/демпфера) → NaN, НЕ падение.
    Закрепляем graceful-skip, а не молчаливый ноль."""
    rows = _oil()
    _need(rows)
    pred = Mo.StructuralOSL().fit(rows).predict(rows)
    assert np.all(np.isnan(pred)), 'structural не должен предсказывать нефтегаз (Фаза C ещё не подключена)'


def test_oilgas_learned_models_predict_finite():
    rows = _oil()
    _need(rows)
    preds, _ = W.walk_forward(rows, Mo.MODELS)
    for name in ('hist_gbm', 'ridge', 'persistence', 'issuer_mean'):
        finite = [p for (*_x, p) in preds[name] if np.isfinite(p)]
        assert len(finite) >= 8, f'{name}: слишком мало конечных прогнозов ({len(finite)})'


# ---------- walk-forward ----------

def test_oilgas_base_falls_back_to_persistence():
    """structural отсутствует → база skill/DM = persistence, common-набор НЕ занулён."""
    rows = _oil()
    _need(rows)
    preds, _ = W.walk_forward(rows, Mo.MODELS)
    assert W._pick_base(preds) == 'persistence'
    summary, common = W.evaluate(preds)
    assert len(common) >= 8, f'common занулён отсутствующим structural: {len(common)}'
    assert summary['hist_gbm']['mape'] is not None


def test_metallurgy_base_stays_structural():
    """Регресс: для металлургии база остаётся structural_osl (он предсказывает)."""
    rows = [r for r in osl_panel.load_panel('metallurgy') if r.has_target and r.period_end]
    _need(rows)
    preds, _ = W.walk_forward(rows, Mo.MODELS)
    assert W._pick_base(preds) == 'structural_osl'


# ---------- conformal ----------

def test_oilgas_conformal_produces_interval():
    rows = osl_panel.load_panel('oilgas')
    _need([r for r in rows if r.has_target])
    res = C.temporal_holdout(rows, Mo.MODELS['hist_gbm'])
    assert res['q'] is not None and res['q'] > 0, 'conformal не дал квантиль'
    assert res['n_calib'] >= 3, f'мал калибровочный фолд: {res["n_calib"]}'
    assert res['n_test'] >= 4, f'мало test-строк: {res["n_test"]}'


# ---------- документированные ограничения (защита от «тихой починки») ----------

def test_oilgas_lukoil_2025_structural_break():
    """ЛУКОЙЛ-2025 = выручка ТОЛЬКО продолжающих операций (деконсолидация LUKOIL Intl,
    санкции окт-2025): ~3768, НЕ ~8600. Это разрыв ряда — не сопоставим с 2021-24.
    Тест фиксирует значение и его несопоставимость (см. SOURCES_oilgas.md)."""
    rows = osl_panel.load_panel('oilgas')
    lk = {r.period: r.target_bn for r in rows if r.issuer == 'ЛУКОЙЛ'}
    _need(lk)
    assert lk.get('2025FY') is not None and lk['2025FY'] < 5000, \
        'ЛУКОЙЛ-2025 должна быть ~3768 (continuing ops), а не полный периметр'
    assert lk.get('2024FY', 0) > 8000, 'ЛУКОЙЛ-2024 — полный периметр ~8622'
