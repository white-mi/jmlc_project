"""
Smoke-тесты кросс-отраслевого синтеза (ds_synthesis.py). Числа агрегируются из уже посчитанных
output/osl_metrics/*_metrics.json; графики требуют matplotlib (extra [eda]).
"""

import pytest

import ds_synthesis  # noqa: E402


@pytest.fixture(scope='module', autouse=True)
def _ensure_metrics():
    """Гарантирует наличие output/osl_metrics/<ind>_metrics.json. `output/` в .gitignore → на
    ЧИСТОМ клоне (CI) JSON отсутствуют; генерим их walk-forward'ом. Делает синтез-тесты
    воспроизводимыми без закоммиченных артефактов. osl_walkforward — core (numpy/sklearn),
    БЕЗ matplotlib, поэтому фикстура работает и в core-CI без extra [eda]."""
    import osl_walkforward
    for ind in ds_synthesis.INDUSTRIES:
        if not (ds_synthesis.METRICS_DIR / f'{ind}_metrics.json').exists():
            osl_walkforward.run(ind)


def test_load_metrics_has_core_industries():
    m = ds_synthesis.load_metrics()
    # как минимум металлургия и энергетика должны быть посчитаны walk-forward'ом
    assert 'metallurgy' in m and 'energy' in m


def test_assemble_numbers_match_reports():
    rows = ds_synthesis.assemble(ds_synthesis.load_metrics())
    by = {r['industry']: r for r in rows}
    # panel-N как в DS_REPORT_*.md (headline)
    assert by['metallurgy']['panel_n'] == 24
    assert by['energy']['panel_n'] == 30
    # энергетика: победитель elasticnet, и WF-порядок elasticnet < persistence < structural
    en = by['energy']
    assert en['winner'] == 'elasticnet'
    assert en['winner_mape'] < en['persistence'] < en['structural']
    # нефтегаз: структурная ОТЛОЖЕНА → None; база = persistence
    if 'oilgas' in by:
        assert by['oilgas']['structural'] is None
        assert by['oilgas']['base'] == 'persistence'


def test_dm_no_significant_winner():
    """Честный инвариант синтеза: ни один победитель не значим (DM p>0.05) на этих N.
    Если тест упал — данные изменились, и нарратив «паттерн, не закон» надо пересмотреть."""
    rows = ds_synthesis.assemble(ds_synthesis.load_metrics())
    for r in rows:
        p = r['winner_dm_p']
        if p is not None:                       # None ⇒ победитель совпал с базой
            assert p > 0.05, f'{r["industry"]}: DM p={p:.3f} стал значим — проверь синтез-нарратив'


def test_conformal_transcription_is_honest():
    """CONFORMAL — транскрипция из DS-отчётов: у каждой записи есть источник, и зафиксирована
    ключевая честность (энергетика 100% vs stale 17%; химия структурный conformal слабый)."""
    c = ds_synthesis.CONFORMAL
    assert set(c) == {'metallurgy', 'oilgas', 'chemistry', 'energy'}
    assert all('src' in v and v['src'].endswith('.md') for v in c.values())
    assert ds_synthesis.ENERGY_CONFORMAL['structural_osl'] == 100
    assert ds_synthesis.ENERGY_CONFORMAL['persistence'] == 17
    assert '17%' in c['chemistry']['note']      # химия: структурный conformal слабый — показан


def test_load_metrics_missing_dir_graceful(tmp_path):
    """Нет JSON в каталоге → пустой dict и пустой run(), без падения."""
    assert ds_synthesis.load_metrics(tmp_path) == {}
    assert ds_synthesis.run(out_dir=tmp_path / 'out', metrics_dir=tmp_path) == {}


def test_run_emits_artifacts(tmp_path):
    pytest.importorskip('matplotlib')
    art = ds_synthesis.run(out_dir=tmp_path)
    assert art, 'нет *_metrics.json — сначала walk-forward по отраслям'
    assert (tmp_path / 'mape_by_industry.png').exists()
    assert (tmp_path / 'conformal_coverage.png').exists()
    table = tmp_path / 'summary_table.md'
    assert table.exists()
    txt = table.read_text(encoding='utf-8')
    for ru in ('Металлургия', 'Нефтегаз', 'Химия', 'Энергетика'):
        assert ru in txt
