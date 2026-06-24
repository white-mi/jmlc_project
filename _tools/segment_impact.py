"""
L3 — Client Behavior / Segment Impact (v0.8 channel-decomposition).

Новая архитектура: шок раскладывается на 5 каналов передачи, каждый сегмент
имеет карту чувствительностей. Решает проблему бифуркации (один шок может
ухудшить положение одних сегментов и одновременно улучшить других).

Каналы:
  consumer      — потребительский (КС, инфляция, реальные доходы)
  oil_revenue   — выручка нефтегаза (Brent, добыча)
  fiscal        — региональные/федеральные бюджеты
  fx            — валютная экспозиция (USD/RUB)
  supply_chain  — цепочки поставок (санкции, логистика, тарифы)

Формула:
  ΔPD(сегмент) = Σ_k [ sens(сегмент,k) × intensity(шок,k) × direction(шок,k)
                       × baseline_pd(k) ] × kc_amplifier
  Δdemand аналогично.
  Δchurn = abs( Σ_k [ sens × intensity × baseline_churn(k) ] ) × kc_amplifier
           — без direction (отток зависит от ИЗМЕНЕНИЯ среды в любую сторону).

Совместимость:
  predict_segment_impact(shock_category='4', kc_regime=..., direction=±1)
    → внутри: разрешает category в default подкатегорию (4 → 4.1),
      применяет глобальный direction как множитель ко всем каналам.
  predict_segment_impact(shock_category='1.2', kc_regime=...)
    → если передана подкатегория, direction игнорируется (берётся
      per-channel из таблицы).

10 сегментов:
  ФЛ × 4: massovy, sredniy, premium, private
  SME × 3: micro, small, mid
  M+/L × 3: mid_corp, large_corp, public

3 режима КС: normal (≤10%), moderate_stress (10-18%), acute_stress (>18%)

Что НЕ делает (TODO Фаза 1+):
  - Channel sensitivity по регионам (Сургут vs Москва — разная экспозиция к oil_revenue)
  - Sub-segments внутри ml_public (нефтегазовые регионы vs прочие)
  - Дополнительные каналы (labor_market, real_estate, agro)
  - Causal layer через DoWhy
  - Реальная ML-модель (KAN/MoE) — нужны данные банка

Использование:
  python segment_impact.py --shock 4.1 --kc-regime acute_stress
  python segment_impact.py --shock 1.2 --kc-regime moderate_stress
  python segment_impact.py --shock 1.2 --kc-regime moderate_stress --segment ml_large_corp
  python segment_impact.py --shock 4 --direction -1   # legacy режим
  python segment_impact.py --check
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

DATA_PATH = Path(__file__).parent / 'data' / 'segment_impact_table_v0_8.json'
LEGACY_DATA_PATH = Path(__file__).parent / 'data' / 'segment_impact_table.json'


@dataclass
class SegmentImpact:
    segment: str
    delta_pd: float
    delta_demand: float
    delta_churn: float
    confidence: str
    amplifier_applied: float
    channel_breakdown: Optional[dict] = None


def load_table(path: Path = DATA_PATH) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


# S3.5: региональные профили экспозиции — множители на каналы для уточнения L3
# по типу региона (нефтегазовый Сургут/ЯНАО vs диверсифицированная Москва).
# Применяются к sensitivity сегмента поверх базовой карты; default (region=None)
# не меняет поведение. Это решает находку F16 (плоская sensitivity на сегмент).
REGION_PROFILES = {
    'oil_region':          {'oil_revenue': 1.6, 'fiscal': 1.4},   # ХМАО/ЯНАО/Сургут
    'metal_monotown':      {'supply_chain': 1.5, 'fiscal': 1.3},  # Магнитогорск/Череповец
    'capital_diversified': {'oil_revenue': 0.5, 'fiscal': 0.7},   # Москва/Казань
    'agricultural_rural':  {'consumer': 1.2, 'fx': 0.8},          # Кубань/Ставрополье
}


def resolve_subcategory(table: dict, shock_category: str) -> str:
    """Разрешает '1' → '1.1' (default), '1.2' → '1.2' (как есть)."""
    cat = str(shock_category)
    if cat in table['shock_subcategories']:
        return cat
    if cat in table.get('category_default_subcategory', {}):
        return table['category_default_subcategory'][cat]
    raise ValueError(
        f"Unknown shock category/subcategory: {shock_category}. "
        f"Допустимые подкатегории: {list(table['shock_subcategories'].keys())}"
    )


def predict_segment_impact(shock_category: str,
                           kc_regime: str = 'acute_stress',
                           direction: int = 1,
                           segments: Optional[list] = None,
                           table_path: Path = DATA_PATH,
                           include_breakdown: bool = False,
                           region: Optional[str] = None) -> dict:
    """Возвращает {segment_id: SegmentImpact} для всех или указанных сегментов.

    Args:
        shock_category: '1'…'5' (top-level) или '1.1'…'5.4' (subcategory).
                        Top-level разрешается в default подкатегорию.
        kc_regime: 'normal' | 'moderate_stress' | 'acute_stress'
        direction: legacy-параметр. +1 — оставить direction из таблицы, -1 —
                   инвертировать direction всех каналов (для случаев, когда
                   подкатегория не указана и нужен глобальный флип).
        segments: подмножество сегментов; если None — все 10
        include_breakdown: если True, добавляет channel_breakdown в результат
    """
    if direction not in (-1, 1):
        raise ValueError(f'direction must be +1 or -1, got {direction}')

    table = load_table(table_path)
    sub = resolve_subcategory(table, shock_category)
    sub_def = table['shock_subcategories'][sub]

    if kc_regime not in table['kc_regimes']:
        raise ValueError(
            f"Unknown КС regime: {kc_regime}. "
            f"Допустимые: {list(table['kc_regimes'].keys())}"
        )
    amp = table['kc_regimes'][kc_regime]['amplifier']

    target_segments = segments or table['segments']
    sensitivity_map = table['channel_sensitivity']
    baseline = table['channel_baseline']
    channels_def = sub_def['channels']
    all_channels = table['channels']
    # S3.5: confidence — поле данных (по умолчанию 'low'); регион уточняет sensitivity
    conf = sub_def.get('confidence', table.get('confidence_default', 'low'))
    region_mult = REGION_PROFILES.get(region, {}) if region else {}
    if region and region not in REGION_PROFILES:
        raise ValueError(
            f'Неизвестный region: {region}. Доступны: {list(REGION_PROFILES)}'
        )

    out = {}
    for sgmt in target_segments:
        if sgmt not in sensitivity_map:
            continue
        sens = sensitivity_map[sgmt]

        delta_pd = 0.0
        delta_demand = 0.0
        delta_churn_signed_sum = 0.0
        breakdown = {}

        for ch in all_channels:
            sens_ch = sens.get(ch, 0.0) * region_mult.get(ch, 1.0)  # S3.5 region
            ch_def = channels_def.get(ch, {})
            intensity = ch_def.get('intensity', 0.0)
            ch_dir = ch_def.get('direction', 1) * direction  # legacy flip
            base = baseline.get(ch, {})

            contrib_pd = sens_ch * intensity * ch_dir * base.get('delta_pd', 0.0)
            contrib_demand = sens_ch * intensity * ch_dir * base.get('delta_demand', 0.0)
            contrib_churn = sens_ch * intensity * base.get('delta_churn', 0.0)
            # Δchurn БЕЗ direction — отток на любое изменение среды

            delta_pd += contrib_pd
            delta_demand += contrib_demand
            delta_churn_signed_sum += contrib_churn

            if include_breakdown and (sens_ch * intensity) > 0:
                breakdown[ch] = {
                    'sensitivity': sens_ch,
                    'intensity': intensity,
                    'direction': ch_dir,
                    'delta_pd_contrib': round(contrib_pd * amp, 4),
                    'delta_demand_contrib': round(contrib_demand * amp, 4),
                }

        out[sgmt] = SegmentImpact(
            segment=sgmt,
            delta_pd=round(delta_pd * amp, 3),
            delta_demand=round(delta_demand * amp, 2),
            delta_churn=round(abs(delta_churn_signed_sum) * amp, 3),
            confidence=conf,  # S3.5: из данных (по умолчанию 'low' — не калибровано)
            amplifier_applied=amp,
            channel_breakdown=breakdown if include_breakdown else None,
        )
    return out


def coverage_check(table_path: Path = DATA_PATH) -> dict:
    """Тестовый инвариант: все каналы и подкатегории заполнены корректно."""
    table = load_table(table_path)
    issues = []

    expected_channels = set(table['channels'])
    expected_segments = set(table['segments'])

    # 1. baseline покрытие
    for ch in expected_channels:
        if ch not in table['channel_baseline']:
            issues.append(f'channel_baseline: missing {ch}')
        else:
            for fld in ('delta_pd', 'delta_demand', 'delta_churn'):
                if fld not in table['channel_baseline'][ch]:
                    issues.append(f'channel_baseline[{ch}]: missing {fld}')

    # 2. sensitivity покрытие
    for sgmt in expected_segments:
        if sgmt not in table['channel_sensitivity']:
            issues.append(f'channel_sensitivity: missing segment {sgmt}')
        else:
            sens = table['channel_sensitivity'][sgmt]
            for ch in expected_channels:
                if ch not in sens:
                    issues.append(f'channel_sensitivity[{sgmt}]: missing channel {ch}')

    # 3. подкатегории шоков
    subs = {k: v for k, v in table['shock_subcategories'].items()
            if not k.startswith('_')}
    for sub_id, sub_def in subs.items():
        for ch in expected_channels:
            if ch not in sub_def.get('channels', {}):
                issues.append(f'shock {sub_id}: missing channel {ch}')
            else:
                cd = sub_def['channels'][ch]
                if 'intensity' not in cd:
                    issues.append(f'shock {sub_id}.{ch}: missing intensity')
                if 'direction' not in cd:
                    issues.append(f'shock {sub_id}.{ch}: missing direction')
                elif cd['direction'] not in (-1, 1):
                    issues.append(f'shock {sub_id}.{ch}: direction must be ±1, got {cd["direction"]}')

    return {
        'ok': len(issues) == 0,
        'channels': len(expected_channels),
        'segments': len(expected_segments),
        'subcategories': len(subs),
        'issues': issues,
    }


def main():
    parser = argparse.ArgumentParser(description='L3 Segment Impact v0.8 (channel-decomposition)')
    parser.add_argument('--shock', help='Категория шока: 1|2|3|4|5 (top-level) или 1.1..5.4 (подкатегория)')
    parser.add_argument('--kc-regime', default='acute_stress',
                        choices=['normal', 'moderate_stress', 'acute_stress'])
    parser.add_argument('--direction', type=int, default=1, choices=[-1, 1],
                        help='Legacy: глобальный флип. При указании подкатегории '
                             'обычно не нужен (direction берётся из таблицы каналов).')
    parser.add_argument('--segment', help='Конкретный сегмент (опционально)')
    parser.add_argument('--breakdown', action='store_true',
                        help='Показать вклад каждого канала в ΔPD сегмента')
    parser.add_argument('--check', action='store_true',
                        help='Проверить покрытие таблицы')
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()

    if args.check:
        result = coverage_check()
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print('=' * 70)
            print(f'  Coverage v0.8: {"✅ OK" if result["ok"] else "❌ ISSUES"}')
            print('=' * 70)
            print(f'  Channels: {result["channels"]} | Segments: {result["segments"]} '
                  f'| Subcategories: {result["subcategories"]}')
            for issue in result['issues']:
                print(f'  ⚠️ {issue}')
        return

    if not args.shock:
        parser.error('--shock обязателен (или --check)')

    segments = [args.segment] if args.segment else None
    impacts = predict_segment_impact(
        args.shock, args.kc_regime,
        direction=args.direction,
        segments=segments,
        include_breakdown=args.breakdown,
    )

    if args.json:
        out = {sgmt: {
            'delta_pd': imp.delta_pd,
            'delta_demand': imp.delta_demand,
            'delta_churn': imp.delta_churn,
            'confidence': imp.confidence,
            'amplifier': imp.amplifier_applied,
            'breakdown': imp.channel_breakdown,
        } for sgmt, imp in impacts.items()}
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    table = load_table()
    sub = resolve_subcategory(table, args.shock)
    sub_def = table['shock_subcategories'][sub]
    regime_info = table['kc_regimes'][args.kc_regime]
    print('=' * 70)
    print(f'  Segment Impact v0.8 · shock={sub} ({sub_def["label"]})')
    print(f'  КС режим: {args.kc_regime} ({regime_info["range_pct"]}%) '
          f'× amp={regime_info["amplifier"]}')
    print('  Каналы шока:')
    for ch in table['channels']:
        cd = sub_def['channels'][ch]
        sign = '+' if cd['direction'] > 0 else '−'
        print(f'    {ch:<14} intensity={cd["intensity"]:.2f} direction={sign}')
    print('=' * 70)
    print(f'  {"segment":<15} | {"ΔPD":>7} | {"Δdemand":>9} | {"Δchurn":>7}')
    print('-' * 70)
    for sgmt, imp in impacts.items():
        print(f'  {sgmt:<15} | {imp.delta_pd:+.3f} | {imp.delta_demand:+8.2f}% | '
              f'{imp.delta_churn:+.3f}')

    if args.breakdown:
        print()
        for sgmt, imp in impacts.items():
            if not imp.channel_breakdown:
                continue
            print(f'\n  Breakdown {sgmt}:')
            for ch, b in imp.channel_breakdown.items():
                sign = '+' if b['direction'] > 0 else '−'
                print(f'    {ch:<14} sens={b["sensitivity"]:.2f} '
                      f'int={b["intensity"]:.2f} dir={sign} '
                      f'→ ΔPD {b["delta_pd_contrib"]:+.4f}')


if __name__ == '__main__':
    main()
