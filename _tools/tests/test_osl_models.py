"""
Тесты трёх моделей прогноза выручки (osl_models.py). Core-зависимости (numpy/sklearn) —
без importorskip. Пропуск только при пустой панели.
"""

import numpy as np
import pytest

import osl_panel
import osl_models


def _rows():
    return [r for r in osl_panel.load_panel('metallurgy') if r.has_target]


def _need(rows):
    if len(rows) < 10:
        pytest.skip('панель мала/пуста')


def test_all_models_fit_predict_positive():
    rows = _rows(); _need(rows)
    for name, ctor in osl_models.MODELS.items():
        m = ctor().fit(rows)
        p = m.predict(rows)
        assert len(p) == len(rows), name
        good = p[~np.isnan(p)]
        assert np.all(good > 0), f'{name}: неположительные прогнозы'
        assert np.all(np.isfinite(good)), f'{name}: не-конечные прогнозы'


def test_structural_raw_available_for_complete_inputs():
    """Где входы полны (Норникель 2025 — есть объёмы; Полюс 2024 — есть золото),
    сырой структурный прогноз не None и в пределах ±50% факта."""
    rows = _rows(); _need(rows)
    s = osl_models.StructuralOSL()
    for iss, per in (('Норникель', '2025FY'), ('Полюс', '2024FY')):
        cand = [r for r in rows if r.issuer == iss and r.period == per]
        assert cand, (iss, per)
        raw = s._raw(cand[0])
        assert raw is not None, (iss, per)
        assert 0.5 * cand[0].target_bn < raw < 1.5 * cand[0].target_bn, (iss, raw, cand[0].target_bn)


def test_structural_none_when_driver_volume_missing():
    """Сталевар 2025 без vol_steel (документированный gap) → raw=None, НЕ падение."""
    rows = _rows(); _need(rows)
    s = osl_models.StructuralOSL()
    nlmk25 = [r for r in rows if r.issuer == 'НЛМК' and r.period == '2025FY']
    if nlmk25:
        assert nlmk25[0].volumes.get('vol_steel_t') is None  # подтверждаем gap
        assert s._raw(nlmk25[0]) is None


def test_structural_in_sample_mape_reasonable():
    rows = _rows(); _need(rows)
    s = osl_models.StructuralOSL().fit(rows)
    y = osl_models._targets(rows)
    assert osl_models.mape(y, s.predict(rows)) < 15.0


def test_structural_scalar_improves_fit():
    """Скаляр-коррекция (из train) не должна ухудшать in-sample MAPE против сырого."""
    rows = _rows(); _need(rows)
    s = osl_models.StructuralOSL().fit(rows)
    raw = np.array([s._raw(r) for r in rows], dtype=float)
    cal = s.predict(rows)
    y = osl_models._targets(rows)
    assert osl_models.mape(y, cal) <= osl_models.mape(y, raw) + 1e-6


def test_issuer_fe_fallback_for_unseen_issuer():
    """_IssuerFE на train без Полюса → level(Полюс) = global_mean (fallback), не падает."""
    rows = _rows(); _need(rows)
    train = [r for r in rows if r.issuer != 'Полюс']
    fe = osl_models._IssuerFE().fit(train)
    assert 'Полюс' not in fe.issuer_mean_
    polyus = [r for r in rows if r.issuer == 'Полюс']
    lvl = fe.level(polyus)
    assert lvl.shape[0] == len(polyus)
    assert np.allclose(lvl, fe.global_mean_)  # неизвестный эмитент → глобальное среднее


def test_issuer_fe_is_train_only_mean():
    """level эмитента = среднее log-таргета именно по train-строкам этого эмитента."""
    rows = _rows(); _need(rows)
    train = [r for r in rows if r.period_end and r.period_end.year <= 2023]
    fe = osl_models._IssuerFE().fit(train)
    pol = [r for r in train if r.issuer == 'Полюс']
    expected = float(np.mean([np.log(r.target_bn) for r in pol]))
    assert abs(fe.issuer_mean_['Полюс'] - expected) < 1e-9


def test_design_scaler_fit_on_train_only():
    """StandardScaler фитится ТОЛЬКО на train: его mean отличается от mean всей панели
    (иначе была бы утечка распределения теста в препроцессинг)."""
    rows = _rows(); _need(rows)
    train = [r for r in rows if r.period_end and r.period_end.year <= 2023]
    full = rows
    d_train = osl_models._Design(use_volumes=False).fit(train)
    d_full = osl_models._Design(use_volumes=False).fit(full)
    # mean_ скейлера на train != mean_ на полной панели (распределения цен разные по годам)
    assert not np.allclose(d_train.scaler_.mean_, d_full.scaler_.mean_), \
        'scaler train == scaler full → подозрение на фит не только на train'


def test_structural_k_computed_from_passed_rows_only():
    """k_ эмитента = median(target/raw) ТОЛЬКО по переданным строкам этого эмитента,
    и не зависит от строк ДРУГИХ эмитентов (фит изолирован → нет утечки в walk-forward)."""
    rows = _rows(); _need(rows)
    s = osl_models.StructuralOSL()
    polyus = [r for r in rows if r.issuer == 'Полюс']
    s.fit(polyus)
    expected = np.median([r.target_bn / s._raw(r) for r in polyus if s._raw(r)])
    assert abs(s.k_['Полюс'] - expected) < 1e-9
    # добавление строки ММК не меняет k Полюса (per-issuer изоляция)
    mmk1 = [r for r in rows if r.issuer == 'ММК'][:1]
    s2 = osl_models.StructuralOSL().fit(polyus + mmk1)
    assert abs(s2.k_['Полюс'] - s.k_['Полюс']) < 1e-9


def test_structural_steel_price_varies_by_period():
    """После wiring iron-ore прокси цена стали (а значит raw сталевара) РАЗНАЯ по годам
    (раньше была заморожена → одинаковый raw на 2023/2024)."""
    rows = _rows(); _need(rows)
    s = osl_models.StructuralOSL()
    sev = {r.period: s._raw(r) for r in rows
           if r.issuer == 'Северсталь' and r.period in ('2023FY', '2024FY')}
    vals = [v for v in sev.values() if v is not None]
    if len(vals) == 2:
        assert abs(vals[0] - vals[1]) > 1e-6, f'raw сталевара заморожен: {sev}'


def test_linear_predictions_finite():
    rows = _rows(); _need(rows)
    for kind in ('ridge', 'elasticnet'):
        m = osl_models.LinearPanel(kind).fit(rows)
        p = m.predict(rows)
        assert np.all(np.isfinite(p)) and np.all(p > 0), kind


def test_gbm_handles_nan_volumes():
    """GBM обучается несмотря на NaN в объёмах (нативный handling), даёт конечные прогнозы."""
    rows = _rows(); _need(rows)
    m = osl_models.GBMPanel().fit(rows)
    p = m.predict(rows)
    assert np.all(np.isfinite(p)) and np.all(p > 0)


def test_mape_known_value():
    y = np.array([100.0, 200.0])
    p = np.array([110.0, 180.0])
    assert abs(osl_models.mape(y, p) - 10.0) < 1e-9


def test_mape_ignores_nan_predictions():
    y = np.array([100.0, 200.0])
    p = np.array([np.nan, 220.0])
    assert abs(osl_models.mape(y, p) - 10.0) < 1e-9
