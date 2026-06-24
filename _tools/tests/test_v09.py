"""Тесты улучшений v0.9 (Спринты 1–3): маршрутизация 27 подкатегорий,
деградация EPU, spillover из severity / credit channel, региональный множитель L3,
recovery JSON в orchestrator."""

import sys
from datetime import datetime
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(TOOLS / 'agents'))

VALID_INDUSTRIES = {'oilgas', 'metallurgy', 'chemistry', 'retail',
                    'energy', 'oiv', 'pharma'}
ALL_SUBCATS = (
    ['1.1', '1.2', '1.3', '1.4', '1.5']
    + ['2.1', '2.2', '2.3', '2.4', '2.5', '2.6']
    + ['3.1', '3.2', '3.3', '3.4', '3.5']
    + ['4.1', '4.2', '4.3', '4.4', '4.5', '4.6', '4.7']
    + ['5.1', '5.2', '5.3', '5.4']
)


# ---------- S2.1: маршрутизация ----------

@pytest.mark.parametrize('subcat', ALL_SUBCATS)
def test_routing_all_27_subcategories(subcat):
    """Каждая из 27 подкатегорий маршрутизируется в валидные отрасли без fallback."""
    import run_pipeline as rp
    state = {}
    inds = rp.get_industries_for_shock(subcat, 'oilgas', state)
    assert inds, f'{subcat}: пустой список отраслей'
    assert set(inds) <= VALID_INDUSTRIES, f'{subcat}: невалидные отрасли {inds}'
    # 5.x намеренно используют primary_industry (fallback-аргумент) — это не routing_fallback
    if not subcat.startswith('5'):
        assert 'routing_fallback' not in state, (
            f'{subcat}: попал на top-level fallback, хотя должен быть в карте')


def test_routing_confirmed_findings():
    """Подтверждённые на перегоне исправления маршрутизации (primary = первый)."""
    import run_pipeline as rp
    assert rp.get_industries_for_shock('2.5', 'x')[0] == 'oiv'          # был retail
    assert rp.get_industries_for_shock('2.6', 'x')[0] == 'metallurgy'   # был retail
    assert rp.get_industries_for_shock('1.4', 'x')[0] == 'metallurgy'   # был oilgas
    assert rp.get_industries_for_shock('4.7', 'x')[0] == 'oilgas'       # был retail


def test_routing_unknown_subcat_uses_fallback_with_flag():
    """Неизвестная подкатегория → top-level резерв + пометка в state."""
    import run_pipeline as rp
    state = {}
    inds = rp.get_industries_for_shock('9.9', 'pharma', state)
    assert inds  # вернулся резерв/fallback
    assert state.get('routing_fallback', {}).get('subcategory') == '9.9'


# ---------- S2.2: деградация EPU ----------

def test_epu_degraded_on_empty_window():
    """Окно в далёком будущем не покрывает корпус → epu_degraded=True, не тихий 0."""
    from calc_rf_epu import compute_epu
    res = compute_epu(window_days=30, end_date=datetime(2027, 12, 31))
    assert res.epu_degraded is True
    assert res.n_total_texts < 3


def test_epu_not_degraded_on_full_corpus():
    """Без окна корпус анализов даёт валидный EPU, degraded=False."""
    from calc_rf_epu import compute_epu
    res = compute_epu(window_days=None)
    assert res.n_total_texts >= 3
    assert res.epu_degraded is False
    assert 0.0 <= res.epu_value <= 100.0


# ---------- S3.4: spillover из severity / credit channel ----------

def test_severity_to_magnitude_range():
    from spillover import severity_to_magnitude
    assert severity_to_magnitude(0) == 0.0
    assert severity_to_magnitude(50) == pytest.approx(0.75, abs=1e-6)
    assert severity_to_magnitude(100) == pytest.approx(1.5, abs=1e-6)
    assert severity_to_magnitude(150) == pytest.approx(1.5, abs=1e-6)  # cap


def test_credit_channel_multi_source():
    """Broad credit channel: retail — наибольший источник; результат покрывает все 7."""
    from spillover import propagate_credit_channel
    r = propagate_credit_channel(magnitude_pp=0.8)
    assert r.source.startswith('multi:')
    assert len(r.impacts) == 7
    # retail (вес 1.0) должен дать наибольший прямой удар
    assert r.ranked[0][0] == 'retail'


# ---------- S3.5: региональный множитель L3 ----------

def test_region_multiplier_amplifies_oil_channel():
    """region='oil_region' усиливает oil_revenue-канал → больший |ΔPD| для НГ-сегмента."""
    from segment_impact import predict_segment_impact
    base = predict_segment_impact('1.2', 'moderate_stress',
                                  segments=['ml_public_oilgas'])['ml_public_oilgas']
    oilreg = predict_segment_impact('1.2', 'moderate_stress',
                                    segments=['ml_public_oilgas'],
                                    region='oil_region')['ml_public_oilgas']
    assert abs(oilreg.delta_pd) > abs(base.delta_pd)


def test_region_unknown_raises():
    from segment_impact import predict_segment_impact
    with pytest.raises(ValueError):
        predict_segment_impact('1.2', 'moderate_stress', region='atlantis')


def test_confidence_is_data_field():
    """confidence приходит из таблицы (S3.5), а не литерал в коде."""
    from segment_impact import predict_segment_impact, load_table
    table = load_table()
    assert table.get('confidence_default') == 'low'
    imp = predict_segment_impact('1.2', 'moderate_stress', segments=['fl_massovy'])
    assert imp['fl_massovy'].confidence == 'low'


# ---------- S1.5: extract_json recovery ----------

def test_extract_json_recovers_truncated():
    from orchestrator import extract_json
    fence = chr(96) * 3
    assert extract_json(fence + 'json\n{"a": 1, "b": [1,2,3') == {'a': 1, 'b': [1, 2, 3]}
    assert extract_json('{"WHAT":"x"') == {'WHAT': 'x'}
    assert extract_json('шум {"k":"v"} хвост') == {'k': 'v'}


# ---------- S4.1: fetch_macro_state — чистая запись ----------

def test_update_macro_state_touches_only_current_state(tmp_path):
    """update_macro_state меняет лишь current_state, не трогая baseline/historical."""
    import json
    import fetch_macro_state as fms

    p = tmp_path / 'macro_state.json'
    p.write_text(json.dumps({
        'indicators': {'key_rate': {'baseline_mean': 8}},
        'current_state': {'_period': '2026-06', 'key_rate': 14.5,
                          'usd_rub': 76.5, 'brent_usd': 68.0, 'inflation_yoy': 5.9},
        'historical_snapshots': [{'period': '2020-04', 'key_rate': 5.5}],
    }, ensure_ascii=False), encoding='utf-8')

    res = fms.update_macro_state({'key_rate': 15.0, 'usd_rub': 77.0},
                                 path=p, period='2026-07')
    assert res['current_state']['key_rate'] == 15.0
    assert res['current_state']['usd_rub'] == 77.0

    data = json.loads(p.read_text(encoding='utf-8'))
    assert data['current_state']['key_rate'] == 15.0
    assert data['current_state']['_period'] == '2026-07'
    # baseline и historical не тронуты
    assert data['indicators']['key_rate']['baseline_mean'] == 8
    assert data['historical_snapshots'][0]['key_rate'] == 5.5


def test_parse_yahoo_chart_brent():
    import fetch_macro_state as fms
    sample = '{"chart":{"result":[{"meta":{"currency":"USD","regularMarketPrice":68.45}}]}}'
    assert fms._parse_yahoo_chart(sample) == 68.45
    assert fms._parse_yahoo_chart('{"chart":{"result":[]}}') is None
    assert fms._parse_yahoo_chart('not json') is None


def test_parse_worldbank_inflation():
    import fetch_macro_state as fms
    sample = ('[{"page":1},[{"date":"2024","value":8.43},'
              '{"date":"2023","value":5.87},{"date":"2025","value":null}]]')
    # берётся последнее доступное (2024), null-год (2025) пропускается
    assert fms._parse_worldbank_inflation(sample) == 8.4
    assert fms._parse_worldbank_inflation('[{"page":1},[]]') is None
    assert fms._parse_worldbank_inflation('garbage') is None


def test_parse_cbr_keyrate_xml():
    import fetch_macro_state as fms
    # упрощённый KeyRate-ответ; берётся запись с максимальной датой
    xml = ("<KeyRate><KR><DT>2026-04-24T00:00:00</DT><Rate>14.50</Rate></KR>"
           "<KR><DT>2026-06-19T00:00:00</DT><Rate>14.00</Rate></KR></KeyRate>")
    assert fms._parse_cbr_keyrate_xml(xml) == 14.0
    # namespaced — парсер по local-name тоже находит
    xml_ns = ('<d:KeyRate xmlns:d="http://web.cbr.ru/"><d:KR>'
              '<d:DT>2026-06-19T00:00:00</d:DT><d:Rate>14,00</d:Rate></d:KR></d:KeyRate>')
    assert fms._parse_cbr_keyrate_xml(xml_ns) == 14.0
    assert fms._parse_cbr_keyrate_xml("<bad") is None


def test_update_macro_state_dry_run_no_write(tmp_path):
    import json
    import fetch_macro_state as fms

    p = tmp_path / 'macro_state.json'
    p.write_text(json.dumps({'current_state': {'key_rate': 14.5}}), encoding='utf-8')
    fms.update_macro_state({'key_rate': 99.0}, path=p, dry_run=True)
    data = json.loads(p.read_text(encoding='utf-8'))
    assert data['current_state']['key_rate'] == 14.5  # не записано


# ---------- v0.9 loop: infer_direction (исправление знака) ----------

def test_infer_direction_stimulus_and_worsening():
    from run_pipeline import infer_direction
    # «снятие санкций» — стимул (-1), не должно перебиваться стемом «санкц»
    assert infer_direction('США объявили о снятии санкций с банка') == -1
    assert infer_direction('ЦБ снизил ставку до 14%') == -1
    assert infer_direction('Деэскалация на иранском треке') == -1
    # ужесточение / удары — +1
    assert infer_direction('Обвал рубля на фоне нового пакета санкций') == 1
    assert infer_direction('ЕС готовит новый пакет санкций против РФ') == 1
    assert infer_direction('Удары дронов по НПЗ') == 1


# ---------- v0.9 loop: контракт --json чист (stdout не загрязнён import-print) ----------

def test_run_pipeline_json_stdout_is_clean():
    import json as _json
    import subprocess
    r = subprocess.run(
        [sys.executable, 'run_pipeline.py', '--smoke-shock', '1.2',
         '--smoke-industry', 'oilgas', '--json'],
        cwd=str(TOOLS), capture_output=True, text=True, timeout=120,
    )
    assert r.returncode == 0, r.stderr[-500:]
    data = _json.loads(r.stdout)  # упадёт, если stdout загрязнён import-print
    assert 'L3_segments' in data and 'L2_spillover' in data
