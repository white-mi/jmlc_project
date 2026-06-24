"""
End-to-End Macro-Radar pipeline (v0.7 MVP).

Один прогон: новость → 4-слойный анализ с числами на каждом уровне.

Слои:
  L0: Multi-Agent классификация (через orchestrator.py)
  L1: текущее состояние макро (CAI + EPU)
  L1.5: OSL прогноз выручки эмитентов с Conformal интервалами
  L2: Industry spillover (числовая матрица)
  L3: Segment impact (lookup-таблица)

Output: один markdown в _Анализы/ + структурированный JSON-state.

Использование:
  # Полный pipeline (требует ANTHROPIC_API_KEY):
  python run_pipeline.py --news-file news.txt --source "ТАСС" --date 2026-04-26

  # Smoke-тест без LLM (использует --shock + --industry для имитации классификации):
  python run_pipeline.py --smoke-shock 4 --smoke-industry oilgas

  # Тест на готовом анализе из _Анализы/ (skip L0):
  python run_pipeline.py --skip-l0 --shock 1 --industry oilgas
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

TOOLS_DIR = Path(__file__).parent
sys.path.insert(0, str(TOOLS_DIR))
sys.path.insert(0, str(TOOLS_DIR / 'agents'))

# Импорты слоёв
from calc_rf_cai import get_current_cai
from calc_rf_epu import get_current_epu
from spillover import (propagate_shock, propagate_credit_channel,
                       severity_to_magnitude)
from segment_impact import predict_segment_impact


# ============================================================
# OSL forward-сценарий (упрощённый)
# ============================================================

# Чувствительность годовой выручки к Brent для разных бизнес-моделей нефтегаза.
# Доля выручки, привязанной к нефтяным ценам (остальное — газ, downstream-маржа,
# регулируемые цены, нефтехимия). Грубое приближение для демо-целей.
OILGAS_BRENT_SENSITIVITY = {
    'oil_vink':   0.65,  # Роснефть, ЛУКОЙЛ — преимущественно нефть
    'gas_vink':   0.25,  # Газпром — гл. газ, частично нефть/нефтепродукты
    'lng_player': 0.40,  # Новатэк — конденсат + LNG (косвенная привязка)
    'pure_oil':   0.75,
}

OSL_BRENT_BASELINE_2025 = 78.0  # средняя Brent 2025 (на чём откалиброван OSL)


def osl_oilgas_forward_scenarios(brent_pre: float, brent_post: float) -> dict:
    """Два forward-сценария выручки нефтегаза для сравнения.

    brent_pre  — Brent статус-кво (например, до новости / на момент Q1)
    brent_post — Brent после рассматриваемого события

    Возвращает {company: {pred_pre, pred_post, delta_pct, sensitivity}}.

    Метод: масштабируем выручку 2025 на отношение Brent / 78,
    взвешенно по чувствительности (sensitivity × ratio + (1−sensitivity)).
    Это упрощённое линейное приближение — реальный пересчёт через OSL модель
    требовал бы полного прогона с overridden ценами; для демо-целей достаточно.
    """
    try:
        from osl_oilgas import PROFILES, ACTUAL_REVENUE_12M_2025
    except Exception as e:
        return {'error': f'OSL oilgas import failed: {e}'}

    out = {}
    for company in ('Роснефть', 'ЛУКОЙЛ', 'Газпром', 'Новатэк'):
        if company not in PROFILES or company not in ACTUAL_REVENUE_12M_2025:
            continue
        profile = PROFILES[company]
        actual_2025 = ACTUAL_REVENUE_12M_2025[company]['rub_bn']
        sensitivity = OILGAS_BRENT_SENSITIVITY.get(profile.business_model, 0.5)

        ratio_pre = brent_pre / OSL_BRENT_BASELINE_2025
        ratio_post = brent_post / OSL_BRENT_BASELINE_2025

        pred_pre = actual_2025 * (1 - sensitivity + sensitivity * ratio_pre)
        pred_post = actual_2025 * (1 - sensitivity + sensitivity * ratio_post)
        delta_pct = (pred_post / pred_pre - 1) * 100 if pred_pre else 0.0

        out[company] = {
            'pred_pre_rub_bn': round(pred_pre, 0),
            'pred_post_rub_bn': round(pred_post, 0),
            'delta_pct': round(delta_pct, 1),
            'sensitivity': sensitivity,
            'business_model': profile.business_model,
        }
    return out


# ============================================================
# Mapping: shock category → industry override
# ============================================================

# S2.1: маршрутизация вынесена в data/shock_to_industries.json (все 27 подкатегорий).
# Раньше inline-словарь покрывал лишь 9 → 18 подкатегорий падали на грубый
# top-level fallback (перегон 15 новостей: 7/15 уходили не в ту отрасль).
_SHOCK_MAP_PATH = TOOLS_DIR / 'data' / 'shock_to_industries.json'


def _load_shock_map() -> dict:
    try:
        data = json.loads(_SHOCK_MAP_PATH.read_text(encoding='utf-8'))
        return data.get('map', {})
    except Exception as e:
        print(f'  ⚠️ Не удалось загрузить shock_to_industries.json: {e}',
              file=sys.stderr)
        return {}


SHOCK_TO_INDUSTRIES = _load_shock_map()

# S2.5: forward-сценарии Brent вынесены в data/brent_scenarios.json.
_BRENT_SCEN_PATH = TOOLS_DIR / 'data' / 'brent_scenarios.json'


def _load_brent_scenarios() -> dict:
    try:
        data = json.loads(_BRENT_SCEN_PATH.read_text(encoding='utf-8'))
        return data.get('scenarios', {})
    except Exception as e:
        print(f'  ⚠️ Не удалось загрузить brent_scenarios.json: {e}',
              file=sys.stderr)
        return {}


BRENT_SCENARIOS = _load_brent_scenarios()

# Грубый top-level резерв — только для подкатегорий, отсутствующих в карте
# (например, новый/некорректный код). Применение логируется в state.
_TOP_LEVEL_FALLBACK = {
    '1': ['oilgas'],
    '2': ['retail'],
    '3': ['oiv'],
    '4': ['retail', 'oiv', 'metallurgy', 'chemistry', 'oilgas'],
}


def get_industries_for_shock(subcategory: str, fallback: str,
                             state: Optional[dict] = None) -> list[str]:
    """Маппинг подкатегории шока на затронутые отрасли (S2.1).

    Приоритет: карта из JSON → primary_industry (пустой список в карте, напр. 5.x)
    → грубый top-level резерв с пометкой routing_fallback в state.
    """
    if subcategory in SHOCK_TO_INDUSTRIES:
        result = SHOCK_TO_INDUSTRIES[subcategory]
        return result if result else [fallback]
    # Подкатегории нет в карте — резерв + предупреждение
    top = subcategory.split('.')[0]
    if state is not None:
        state['routing_fallback'] = {
            'subcategory': subcategory,
            'reason': 'нет в shock_to_industries.json — использован top-level резерв',
        }
    print(f'  ⚠️ routing fallback для подкатегории {subcategory!r}',
          file=sys.stderr)
    return _TOP_LEVEL_FALLBACK.get(top, [fallback])


# Direction-классификатор (legacy, только для top-level пути без подкатегории).
# При наличии подкатегории direction берётся per-channel из таблицы L3.
# Стимул-фразы СПЕЦИФИЧНЫ и имеют приоритет, чтобы «снятие санкций» (-1) не
# перебивалось общим стемом «санкц» (+1). Стемы покрывают морфоформы.
DIRECTION_STIMULUS = (   # → -1 (смягчение / улучшение условий)
    'снятие санкц', 'отмена санкц', 'снят санкц', 'смягчение санкц',
    'снижение ставк', 'снизил ставк', 'снижение кс', 'снижение ключев',
    'смягчен', 'деэскал', 'стимул', 'поддержк',
    'оживлен', 'восстанов', 'укреплен рубл',
)
DIRECTION_WORSENING = (  # → +1 (ужесточение / удар / кризис)
    'повыш', 'ужесточ', 'обвал', 'кризис', 'дефолт',
    'новый пакет санкц', 'новые санкц', 'эмбарго',
    'удар', 'атак', 'катастроф', 'ослаблен рубл', 'падени',
)


def infer_direction(text: str, subcategory: str = '') -> int:
    """Определяет направление шока (+1 ухудшение / -1 смягчение) по тексту.
    Приоритет у специфичных стимул-фраз, затем worsening, иначе +1 (default)."""
    text_low = (text + ' ' + subcategory).lower()
    # Снятие/отмена/смягчение санкций → стимул (приоритет над общим стемом «санкц»);
    # стем-комбо ловит морфоформы (снятие/снятии/сняты).
    if 'санкц' in text_low and any(s in text_low for s in ('снят', 'отмен', 'смягч')):
        return -1
    if any(h in text_low for h in DIRECTION_STIMULUS):
        return -1
    if any(h in text_low for h in DIRECTION_WORSENING):
        return 1
    return 1


def kc_regime_from_rate(key_rate_pct: float) -> str:
    """Определяет режим КС по текущей ставке."""
    if key_rate_pct <= 10:
        return 'normal'
    elif key_rate_pct <= 18:
        return 'moderate_stress'
    else:
        return 'acute_stress'


# ============================================================
# OSL: Conformal интервал для эмитентов в затронутой отрасли
# ============================================================

def osl_for_industry(industry: str) -> dict:
    """Возвращает Conformal интервалы для эмитентов отрасли."""
    try:
        from conformal_prediction import (
            make_interval_from_metallurgy, make_interval_generic,
            make_interval_retail, make_interval_energy,
            make_interval_oiv, make_interval_pharma,
        )
    except Exception as e:
        return {'error': f'conformal_prediction import failed: {e}'}

    try:
        if industry == 'metallurgy':
            import osl_metallurgy as m
            companies = list(m.PROFILES.keys())
            results = [make_interval_from_metallurgy(c) for c in companies]
        elif industry == 'oilgas':
            import osl_oilgas as m
            companies = ['Роснефть', 'ЛУКОЙЛ', 'Газпром', 'Новатэк']
            results = [make_interval_generic(c, 'osl_oilgas')
                        for c in companies if c in m.ACTUAL_REVENUE_12M_2025]
        elif industry == 'chemistry':
            import osl_chemistry as m
            companies = ['ФосАгро', 'Акрон', 'СИБУР']
            results = [make_interval_generic(c, 'osl_chemistry')
                        for c in companies if c in m.ACTUAL_REVENUE_2025]
        elif industry == 'retail':
            results = [make_interval_retail(c) for c in ['Wildberries', 'Ozon', 'М.Видео']]
        elif industry == 'energy':
            results = [make_interval_energy(c) for c in
                       ['Интер РАО', 'РусГидро', 'Юнипро', 'Т Плюс', 'Росатом-Энергоатом']]
        elif industry == 'oiv':
            results = [make_interval_oiv(r) for r in
                       ['ХМАО-Югра', 'Тюменская обл.', 'ЯНАО', 'Татарстан', 'Сахалинская обл.']]
        elif industry == 'pharma':
            results = [make_interval_pharma(c) for c in ['Пульс', 'Протек', 'Катрен']]
        else:
            return {'error': f'unknown industry: {industry}'}

        return {
            'industry': industry,
            'emitents': [
                {
                    'company': r.company,
                    'base_pred_rub_bn': round(r.predicted_base, 1),
                    'low_rub_bn': round(r.predicted_low, 1),
                    'high_rub_bn': round(r.predicted_high, 1),
                    'width_pct': round(r.interval_width_pct, 1),
                    'actual_rub_bn': r.actual,
                    'inside_interval': r.actual_in_interval,
                } for r in results
            ],
        }
    except Exception as e:
        return {'error': f'OSL run failed: {e}', 'industry': industry}


# ============================================================
# Pipeline
# ============================================================

def run_full_pipeline(
    news_text: Optional[str] = None,
    source: str = '',
    date: str = '',
    smoke_shock: Optional[str] = None,
    smoke_industry: Optional[str] = None,
    skip_l0: bool = False,
    llm_mode: str = 'cli',
    smoke_severity: int = 60,
) -> dict:
    """Полный pipeline. Возвращает state-dict с числами на всех слоях.
    smoke_severity — сила шока для smoke-режима (S3.4: влияет на L2 magnitude)."""

    state = {
        'pipeline_version': '0.9',
        'timestamp': datetime.now().isoformat(),
        'date_news': date or datetime.now().strftime('%Y-%m-%d'),
        'source_news': source,
    }

    # ---- L1: текущее макро-состояние (CAI + EPU) ----
    cai_result = get_current_cai()
    # S2.2: якорим окно EPU на дате новости, а не на now() — иначе при свежей
    # дате окно не покрывает корпус _Анализы/ и EPU схлопывается в 0.
    epu_end = None
    try:
        epu_end = datetime.strptime(state['date_news'], '%Y-%m-%d')
    except (ValueError, KeyError):
        epu_end = None
    epu_result = get_current_epu(window_days=30, end_date=epu_end)

    state['L1_macro'] = {
        'cai': cai_result.cai,
        'phase': cai_result.phase,
        'yield_curve_slope_pp': cai_result.yield_curve_slope_pp,
        'epu': epu_result.epu_value,
        'epu_negative_pct': epu_result.epu_negative_pct,
        'epu_n_uncertainty_texts': epu_result.n_uncertainty_texts,
        'epu_n_negative_texts': epu_result.n_negative_texts,
        'epu_n_total_texts': epu_result.n_total_texts,
        'epu_degraded': epu_result.epu_degraded,
    }

    # КС регим — для L3 amplifier
    key_rate = cai_result.components.get('key_rate', {}).get('current', 16.0)
    kc_regime = kc_regime_from_rate(key_rate)
    state['kc_regime'] = kc_regime

    # ---- L0: классификация ----
    if smoke_shock:
        # Smoke-режим — не вызываем LLM, используем флаг
        state['L0_classification'] = {
            'WHAT': news_text[:200] if news_text else '(smoke test)',
            'main_category': smoke_shock.split('.')[0],
            'subcategory': smoke_shock,
            'severity_score': smoke_severity,
            'severity_level': 'M',
            'mode': 'smoke',
        }
    elif skip_l0:
        # Используется когда уже есть готовый L0 в state
        state['L0_classification'] = {'mode': 'skipped'}
    else:
        # Реальный LLM прогон через orchestrator
        try:
            from orchestrator import run_pipeline as run_agents
            agent_state, _md = run_agents(
                news_text or '', source, state['date_news'],
                llm_mode=llm_mode)
            state['L0_classification'] = agent_state
        except Exception as e:
            state['L0_classification'] = {'error': str(e), 'mode': 'failed'}

    # ---- L1.5 + L2: OSL для затронутых отраслей + Spillover ----
    subcat = state['L0_classification'].get('subcategory', '5.1')
    industries = get_industries_for_shock(subcat, smoke_industry or 'oilgas', state)
    primary_industry = industries[0]

    state['L1_5_osl'] = {
        ind: osl_for_industry(ind) for ind in industries
    }

    # Forward-сценарий для нефтегаза (если затронут): сравнение
    # «статус-кво до события» vs «после события» при разных уровнях Brent.
    # S2.5: пары pre/post берутся из data/brent_scenarios.json; null → текущий Brent.
    if 'oilgas' in industries:
        cur_brent = float(cai_result.components.get('brent_usd', {}).get('current', 78.0))
        scen = BRENT_SCENARIOS.get(subcat, BRENT_SCENARIOS.get('default', {}))
        brent_pre = scen.get('pre') if scen.get('pre') is not None else cur_brent
        brent_post = scen.get('post') if scen.get('post') is not None else cur_brent
        state['L1_5_oilgas_forward'] = {
            'brent_pre': brent_pre,
            'brent_post': brent_post,
            'companies': osl_oilgas_forward_scenarios(brent_pre, brent_post),
        }

    # S3.4: magnitude из severity L0 (раньше фикс 0.8); для шоков ставки ЦБ
    # (категория 4) — broad credit channel из 5 отраслей, а не одна.
    sev = state['L0_classification'].get('severity_score')
    magnitude = severity_to_magnitude(sev) if sev is not None else 0.8
    if subcat.split('.')[0] == '4':
        spill = propagate_credit_channel(magnitude_pp=magnitude)
    else:
        spill = propagate_shock(primary_industry, magnitude_pp=magnitude)
    state['L2_spillover'] = {
        'source': spill.source,
        'magnitude_pp': spill.magnitude_pp,
        'impacts': spill.impacts,
        'ranked': spill.ranked,
    }

    # ---- L3: Segment Impact (v0.8 channel-decomposition) ----
    # При наличии подкатегории direction берётся per-channel из таблицы;
    # глобальный direction не нужен и не применяется.
    direction = infer_direction(news_text or '', subcat)
    state['shock_direction'] = direction
    state['shock_subcategory_used'] = subcat
    has_subcat = '.' in subcat
    segments_result = predict_segment_impact(
        shock_category=subcat if has_subcat else subcat.split('.')[0],
        kc_regime=kc_regime,
        direction=1 if has_subcat else direction,
        include_breakdown=True,
    )
    state['L3_segments'] = {
        sgmt: {
            'delta_pd_pp': imp.delta_pd,
            'delta_demand_pct': imp.delta_demand,
            'delta_churn_pp': imp.delta_churn,
            'confidence': imp.confidence,
            'channel_breakdown': imp.channel_breakdown,
        } for sgmt, imp in segments_result.items()
    }

    return state


def render_markdown(state: dict) -> str:
    """Рендеринг pipeline-state в markdown."""
    lines = []
    lines.append('---')
    lines.append('tags: [макро-радар, pipeline, mvp]')
    lines.append(f'дата_анализа: "{state["date_news"]}"')
    lines.append(f'pipeline_version: "{state["pipeline_version"]}"')
    lines.append('---\n')

    lines.append('# Макро-радар End-to-End анализ')
    lines.append(f'> Источник: {state.get("source_news", "—")}\n')

    # L0
    l0 = state.get('L0_classification', {})
    lines.append('## L0 — Классификация события')
    lines.append(f'- **Что:** {l0.get("WHAT", "?")}')
    lines.append(f'- **Категория:** {l0.get("subcategory") or l0.get("main_category", "?")}')
    lines.append(f'- **Сила:** {l0.get("severity_score", "?")} ({l0.get("severity_level", "?")})')

    # L1
    l1 = state.get('L1_macro', {})
    lines.append('\n## L1 — Макро-состояние РФ')
    lines.append(f'- **CAI:** {l1.get("cai", "?"):+.2f} → phase: **{l1.get("phase", "?")}**')
    epu_degraded_note = '  ⚠️ degraded (корпус в окне мал)' if l1.get('epu_degraded') else ''
    lines.append(f'- **EPU (30d):** {l1.get("epu", "?")}/100 — общая частота uncertainty-текстов{epu_degraded_note}')
    lines.append(f'  - **Negative-EPU:** {l1.get("epu_negative_pct", "?")}% '
                 f'({l1.get("epu_n_negative_texts", 0)}/'
                 f'{l1.get("epu_n_uncertainty_texts", 0)} текстов с явно негативным сентиментом)')
    lines.append(f'- **Yield curve slope:** {l1.get("yield_curve_slope_pp", "?")} п.п.')
    lines.append(f'- **КС режим:** {state.get("kc_regime", "?")}')
    direction = state.get('shock_direction', 1)
    dir_label = 'classical (+1)' if direction == 1 else 'inverted (-1, стимул/смягчение)'
    sub_used = state.get('shock_subcategory_used', '?')
    if '.' in sub_used:
        lines.append(f'- **Подкатегория шока:** {sub_used} → L3 использует per-channel direction из таблицы (legacy direction игнорируется)')
    else:
        lines.append(f'- **Direction шока (legacy):** {dir_label}')

    # L1.5 OSL
    lines.append('\n## L1.5 — OSL: прогноз выручки эмитентов')
    osl = state.get('L1_5_osl', {})
    for industry, data in osl.items():
        lines.append(f'\n### {industry}')
        if 'error' in data:
            lines.append(f'⚠️ {data["error"]}')
            continue
        lines.append('| Эмитент | Прогноз | 90% interval | Факт | Inside |')
        lines.append('|---|---|---|---|---|')
        for em in data.get('emitents', []):
            inside = '✅' if em.get('inside_interval') else (
                '❌' if em.get('inside_interval') is False else '—')
            actual = f'{em["actual_rub_bn"]:.0f}' if em.get('actual_rub_bn') else '—'
            lines.append(
                f'| {em["company"]} | {em["base_pred_rub_bn"]:.0f} | '
                f'[{em["low_rub_bn"]:.0f}; {em["high_rub_bn"]:.0f}] '
                f'(±{em["width_pct"]:.1f}%) | {actual} | {inside} |'
            )

    # L1.5 forward для нефтегаза — динамика прогноза от события
    fwd = state.get('L1_5_oilgas_forward')
    if fwd and fwd.get('companies'):
        lines.append('\n### L1.5 forward — динамика прогноза нефтегаза от события')
        lines.append(f'**Brent:** статус-кво ${fwd["brent_pre"]:.0f}/барр → '
                     f'после события ${fwd["brent_post"]:.0f}/барр')
        lines.append('\n| Эмитент | Прогноз pre, млрд ₽ | Прогноз post, млрд ₽ | Δ (%) | Чувствит. к Brent |')
        lines.append('|---|---:|---:|---:|---:|')
        for company, d in fwd['companies'].items():
            delta_marker = '🔴' if d['delta_pct'] < -3 else (
                '🟢' if d['delta_pct'] > 3 else '⚪')
            lines.append(
                f'| {company} | {d["pred_pre_rub_bn"]:.0f} | '
                f'{d["pred_post_rub_bn"]:.0f} | {delta_marker} {d["delta_pct"]:+.1f}% | '
                f'{d["sensitivity"]:.2f} |'
            )

    # L2
    l2 = state.get('L2_spillover', {})
    lines.append('\n## L2 — Industry Spillover')
    lines.append(f'**Источник шока:** {l2.get("source", "?")} (magnitude {l2.get("magnitude_pp", "?")} п.п.)')
    lines.append('\n| Отрасль | ΔPD (п.п.) |')
    lines.append('|---|---|')
    for ind, dpp in l2.get('ranked', []):
        marker = '🔴' if dpp >= 0.5 else ('🟡' if dpp >= 0.2 else '🟢')
        lines.append(f'| {marker} {ind} | {dpp:+.3f} |')

    # L3
    l3 = state.get('L3_segments', {})
    lines.append('\n## L3 — Client Behavior (10 сегментов, channel-decomposition v0.8)')
    lines.append('\n| Сегмент | ΔPD (п.п.) | Δdemand (%) | Δchurn (п.п.) | Confidence |')
    lines.append('|---|---|---|---|---|')
    for sgmt, data in l3.items():
        # Маркер бифуркации: ΔPD < 0 → улучшение, > 0 → ухудшение
        pd = data["delta_pd_pp"]
        marker = '🟢' if pd < -0.05 else ('🔴' if pd > 0.05 else '⚪')
        lines.append(
            f'| {marker} {sgmt} | {pd:+.3f} | '
            f'{data["delta_demand_pct"]:+.2f}% | {data["delta_churn_pp"]:+.3f} | '
            f'{data["confidence"]} |'
        )

    # Channel breakdown: показываем для каждого сегмента вклад каналов
    has_breakdown = any(data.get('channel_breakdown') for data in l3.values())
    if has_breakdown:
        lines.append('\n### Channel breakdown (вклад каналов в ΔPD сегмента)')
        lines.append('\n| Сегмент | consumer | oil_revenue | fiscal | fx | supply_chain |')
        lines.append('|---|---|---|---|---|---|')
        for sgmt, data in l3.items():
            br = data.get('channel_breakdown') or {}
            row = [sgmt]
            for ch in ('consumer', 'oil_revenue', 'fiscal', 'fx', 'supply_chain'):
                if ch in br:
                    row.append(f'{br[ch]["delta_pd_contrib"]:+.3f}')
                else:
                    row.append('—')
            lines.append('| ' + ' | '.join(row) + ' |')

    lines.append('\n---')
    lines.append(f'*MVP pipeline v0.8 · L3 channel-decomposition · timestamp {state["timestamp"]}*')

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='Macro-Radar End-to-End MVP')
    parser.add_argument('--news-file', help='Файл с текстом новости')
    parser.add_argument('--source', default='', help='Источник новости')
    parser.add_argument('--date', default=datetime.now().strftime('%Y-%m-%d'))
    parser.add_argument('--smoke-shock', help='Smoke: подкатегория шока (например 4.1)')
    parser.add_argument('--smoke-industry', help='Smoke: затронутая отрасль')
    parser.add_argument('--skip-l0', action='store_true',
                        help='Пропустить L0 (использовать --smoke-* для классификации)')
    parser.add_argument('--shock', help='alias для --smoke-shock')
    parser.add_argument('--industry', help='alias для --smoke-industry')
    parser.add_argument('--llm-mode', choices=['cli', 'sdk', 'dry-run'],
                        default='cli',
                        help='Способ вызова LLM для L0 (cli — через `claude -p`, '
                             'sdk — через ANTHROPIC_API_KEY, dry-run — без LLM)')
    parser.add_argument('--out', help='Файл для сохранения markdown')
    parser.add_argument('--json', action='store_true', help='Вывод JSON state')
    args = parser.parse_args()

    news_text = None
    if args.news_file:
        news_text = Path(args.news_file).read_text(encoding='utf-8').strip()

    smoke_shock = args.smoke_shock or args.shock
    smoke_industry = args.smoke_industry or args.industry

    state = run_full_pipeline(
        news_text=news_text, source=args.source, date=args.date,
        smoke_shock=smoke_shock, smoke_industry=smoke_industry,
        skip_l0=args.skip_l0, llm_mode=args.llm_mode,
    )

    if args.json:
        print(json.dumps(state, ensure_ascii=False, indent=2, default=str))
        return

    md = render_markdown(state)
    if args.out:
        Path(args.out).write_text(md, encoding='utf-8')
        print(f'✅ Сохранено: {args.out}', file=sys.stderr)
    else:
        print(md)


if __name__ == '__main__':
    main()
