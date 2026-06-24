"""
DS-слой ХИМИИ (3-я отрасль). Отличие от нефтегаза: структурная модель ПОДКЛЮЧЕНА
(Фаза C) — у химии есть публичные годовые цены (World Bank удобрения/нефть) и нет
налогового блокера, поэтому StructuralOSL предсказывает, и база сравнения = structural_osl.

Закрепляем: structural активна и competitive; КуйбышевАзот без объёма → _raw None
(graceful, идёт по learned/persistence); ФосАгро/Акрон/КОС структурно покрыты.
"""

import numpy as np
import pytest

import osl_panel
import osl_models as Mo
import osl_walkforward as W
import conformal_split as C

ISSUERS = {'ФосАгро', 'Акрон', 'КуйбышевАзот', 'КОС'}


def _chem():
    return [r for r in osl_panel.load_panel('chemistry') if r.has_target and r.period_end]


def _need(rows):
    if not rows:
        pytest.skip('chemistry-панель пуста')


# ---------- панель ----------

def test_chemistry_panel_loads():
    rows = _chem()
    _need(rows)
    assert len(rows) >= 16, f'мало строк: {len(rows)}'
    assert ISSUERS <= {r.issuer for r in rows}
    assert all(r.industry == 'chemistry' for r in rows)
    assert 'Нижнекамскнефтехим' not in {r.issuer for r in rows}, 'НКНХ должен быть удалён (плохой клиент)'


def test_chemistry_issuers_3plus_periods():
    rows = _chem()
    _need(rows)
    by = {}
    for r in rows:
        by.setdefault(r.issuer, set()).add(r.period)
    thin = {i: len(p) for i, p in by.items() if len(p) < 3}
    assert not thin, f'эмитенты с <3 периодов: {thin}'


def test_chemistry_revenue_positive_rub():
    rows = _chem()
    _need(rows)
    for r in rows:
        assert r.target_bn > 0 and r.revenue_currency == 'RUB'


def test_chemistry_required_prices_present():
    """Каждый период видит dap/urea/crude_brent/usd_rub (NaN-free для learned + structural)."""
    rows = _chem()
    _need(rows)
    for r in rows:
        for k in ('dap', 'urea', 'crude_brent', 'usd_rub'):
            assert r.prices.get(k) is not None, f'{r.issuer} {r.period}: нет цены {k}'


# ---------- структурная модель (Фаза C — ПОДКЛЮЧЕНА) ----------

def test_chemistry_structural_predicts():
    """StructuralOSL даёт конечные прогнозы для химии (ФосАгро/Акрон с объёмом)."""
    rows = _chem()
    _need(rows)
    pred = Mo.StructuralOSL().fit(rows).predict(rows)
    finite = np.isfinite(pred).sum()
    assert finite >= 6, f'структурная покрыла слишком мало строк: {finite}'


def test_chemistry_kuibyshevazot_raw_is_none():
    """КуйбышевАзот — объём не раскрыт (gap) → _raw None для ВСЕХ его строк (graceful,
    не молчаливый ноль; в ансамбле падает на learned/persistence). Контракт-guard."""
    rows = _chem()
    _need(rows)
    m = Mo.StructuralOSL()
    kaz = [r for r in rows if r.issuer == 'КуйбышевАзот']
    assert kaz, 'нет строк КуйбышевАзот'
    assert all(m._raw(r) is None for r in kaz), 'КуйбышевАзот без объёма должен давать _raw=None'


def test_chemistry_phosagro_structural_finite():
    """ФосАгро (объём есть все годы) — структурный raw конечен и положителен."""
    rows = _chem()
    _need(rows)
    m = Mo.StructuralOSL()
    phos = [r for r in rows if r.issuer == 'ФосАгро']
    raws = [m._raw(r) for r in phos]
    assert all(x is not None and x > 0 for x in raws), f'ФосАгро raw: {raws}'


# ---------- walk-forward ----------

def test_chemistry_base_is_structural():
    """structural предсказывает → база skill/DM = structural_osl (НЕ persistence, в отличие от нефтегаза)."""
    rows = _chem()
    _need(rows)
    preds, _ = W.walk_forward(rows, Mo.MODELS)
    assert W._pick_base(preds) == 'structural_osl'
    summary, common = W.evaluate(preds)
    assert summary['structural_osl']['mape'] is not None
    # < 16 брекетит заявленный результат (~13.8) и ловит регресс до уровня persistence (~16)
    assert summary['structural_osl']['mape'] < 16, 'структурная MAPE деградировала до наивного уровня'


# ---------- conformal ----------

def test_chemistry_conformal_elasticnet():
    """Conformal на elasticnet (full-coverage остатки, calib n=4). Структурная даёт calib n=2
    (разреженный объём) → для интервалов берём learned-модель (документированное ограничение)."""
    rows = osl_panel.load_panel('chemistry')
    _need([r for r in rows if r.has_target])
    res = C.temporal_holdout(rows, Mo.MODELS['elasticnet'])
    assert res['q'] is not None and res['q'] > 0
    assert res['n_calib'] >= 3 and res['n_test'] >= 4
