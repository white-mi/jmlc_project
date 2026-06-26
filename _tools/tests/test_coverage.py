"""
Тест-страж тиринга покрытия (industry_coverage.json ↔ реальность). Запрещает тихий дрейф:
манифест должен соответствовать коду/данным/артефактам. Только core-зависимости.

Проверки используют ЗАКОММИЧЕННЫЕ артефакты (DS-отчёт + INDUSTRY_PRICE_FEATURES), а НЕ генерируемые
output/osl_metrics/*.json (они в .gitignore) → тест детерминирован на чистом клоне.
"""

import json

import coverage
import osl_models

REPO = coverage.REPO
SPILLOVER = REPO / '_tools' / 'data' / 'spillover_matrix.json'
VALIDATED_TIERS = {'validated', 'validated_structural_deferred'}
KNOWN_TIERS = VALIDATED_TIERS | {'illustrative', 'illustrative_pending'}


def _man():
    return coverage.load()


def _stable(md_text: str) -> list:
    """Строки доку БЕЗ таблицы живых чисел: она зависит от версии sklearn/окружения (MAPE из
    walk-forward), поэтому из сравнения исключается. Остальное — детерминированно из манифеста,
    и именно там был frontmatter-дрейф, который ловит фриш-тест."""
    out, skip = [], False
    for ln in md_text.splitlines():
        if ln.startswith('## Валидированные отрасли'):
            skip = True
        elif ln.startswith('## Почему'):
            skip = False
        if not skip:
            out.append(ln)
    return out


def test_all_tiers_known():
    """Любой неизвестный тир (опечатка) обвалит тест — иначе он бы тихо ускользнул от обоих гардов."""
    for ind, meta in _man()['industries'].items():
        assert meta['tier'] in KNOWN_TIERS, f'{ind}: неизвестный тир {meta["tier"]!r}'


def test_manifest_loads_and_has_industries():
    man = _man()
    assert man['industries'], 'манифест без отраслей'
    assert 'tiers' in man and 'layer_legend' in man


def test_every_industry_has_all_layers():
    for ind, meta in _man()['industries'].items():
        assert set(meta['layers']) == {'L0', 'L1', 'L1.5', 'L2', 'L3'}, ind
        assert meta.get('rationale'), f'{ind}: нет мотивации (rationale)'


def test_validated_industries_have_committed_artifacts():
    """Каждый validated/deferred: DS-отчёт закоммичен И отрасль в INDUSTRY_PRICE_FEATURES."""
    for ind, meta in _man()['industries'].items():
        if meta['tier'] not in VALIDATED_TIERS:
            continue
        assert ind in osl_models.INDUSTRY_PRICE_FEATURES, f'{ind}: нет в INDUSTRY_PRICE_FEATURES'
        rep = REPO / meta['ds_report']
        assert rep.exists(), f'{ind}: нет DS-отчёта {meta["ds_report"]}'
        assert meta.get('metrics', '').endswith('_metrics.json'), f'{ind}: кривой путь metrics'


def test_illustrative_industries_are_not_wired_as_structural():
    """Каждый illustrative: модуль osl_*.py есть, но отрасли НЕТ в INDUSTRY_PRICE_FEATURES
    (иначе она бы претендовала на структурную Q×P) и НЕТ в ds_synthesis.INDUSTRIES."""
    import ds_synthesis
    for ind, meta in _man()['industries'].items():
        if meta['tier'] not in {'illustrative', 'illustrative_pending'}:
            continue
        mod = REPO / meta['module']
        assert mod.exists(), f'{ind}: нет модуля {meta["module"]}'
        assert ind not in osl_models.INDUSTRY_PRICE_FEATURES, f'{ind}: иллюстративный, но в PRICE_FEATURES'
        assert ind not in ds_synthesis.INDUSTRIES, f'{ind}: иллюстративный, но в ds_synthesis.INDUSTRIES'


def test_manifest_industries_match_spillover_matrix():
    """Манифест покрывает РОВНО те же отрасли, что и матрица спилловера L2."""
    man_inds = set(_man()['industries'])
    spill = set(json.loads(SPILLOVER.read_text(encoding='utf-8'))['industries'])
    assert man_inds == spill, f'манифест {man_inds} != спилловер {spill}'


def test_ds_synthesis_industries_are_exactly_validated():
    """ds_synthesis (кросс-отраслевой синтез) перечисляет РОВНО валидированные отрасли."""
    import ds_synthesis
    val = {i for i, m in _man()['industries'].items() if m['tier'] in VALIDATED_TIERS}
    assert set(ds_synthesis.INDUSTRIES) == val, f'{set(ds_synthesis.INDUSTRIES)} != validated {val}'


def test_render_markdown_smoke():
    md = coverage.render_markdown()
    assert '# Покрытие отраслей' in md
    for ru in ('Металлургия', 'Энергетика', 'Фарма', 'ОИВ'):
        assert ru in md
    assert isinstance(coverage.summary(), str)


def test_committed_doc_matches_generator():
    """Закоммиченный COVERAGE_TIERS.md == вывод генератора ВНЕ таблицы живых чисел (числа зависят
    от версии sklearn → сравниваем детерминированную манифест-driven часть). Ловит ручные правки
    сгенерированного файла (как был frontmatter-дрейф). Не требует output/ JSON → clean-clone-safe."""
    gen = _stable(coverage.render_markdown())
    committed = _stable(coverage.DOC.read_text(encoding='utf-8'))
    assert gen == committed, 'COVERAGE_TIERS.md разошёлся с генератором — перегенери: python coverage.py'
