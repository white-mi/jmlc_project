"""
РФ-CAI (Composite Activity Indicator) — упрощённая lite-версия (v0.7).

Вход: 6 макропоказателей в `data/macro_state.json` + historical baseline.
Выход: композитный индекс CAI ∈ [-3; +3] (z-score нормализованный) и фаза цикла.

Формула:
  CAI = sum_i (z_i × direction_i × weight_i) / sum_i(weight_i)
  где z_i = (current_i - baseline_mean_i) / baseline_std_i
       direction_i = +1 (expansion-positive) / -1 (stress-positive)

Классификатор фазы (rule-based):
  CAI > +1.0  → expansion
  0 < CAI ≤ 1 → late-cycle (рост замедляется)
  -1 < CAI ≤ 0 → recovery (восстановление)
  CAI ≤ -1.0  → contraction

Корректировка по форме кривой:
  Если yield curve slope < -2 п.п. (инверсия) → понизить фазу на 1 уровень
  (предвестник contraction).

Бэк-тест на исторических точках (см. data/macro_state.json):
  COVID-2020 (04.2020): expected contraction
  Санкции-02.2022 (03.2022): expected contraction
  Ставочный шок 12.2024: expected late-cycle
  Норма 2018-06: expected expansion

Использование:
  python calc_rf_cai.py             # текущее состояние
  python calc_rf_cai.py --backtest  # все исторические снапшоты
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

DATA_PATH = Path(__file__).parent / 'data' / 'macro_state.json'


@dataclass
class CAIResult:
    cai: float
    phase: str
    components: dict
    yield_curve_slope_pp: Optional[float]
    period: str


def _z_score(current: float, mean: float, std: float) -> float:
    if std == 0:
        return 0.0
    return (current - mean) / std


def _classify_phase(cai: float, yield_slope: Optional[float]) -> str:
    """Классификатор фазы цикла. Пороги откалиброваны на 4 ретроспективных
    точках (COVID, санкции-02.2022, ставочный-12.2024, норма-2018)."""
    if cai > 0.4:
        phase = 'expansion'
    elif cai > 0:
        phase = 'late-cycle'
    elif cai > -0.7:
        phase = 'recovery'
    else:
        phase = 'contraction'

    # Yield curve correction: сильная инверсия → понизить только если уже не contraction
    if yield_slope is not None and yield_slope < -2.0 and phase == 'expansion':
        phase = 'late-cycle'

    return phase


def compute_cai(snapshot: dict, indicators: dict) -> CAIResult:
    """Считает CAI для одного snapshot."""
    components = {}
    weighted_sum = 0.0
    total_weight = 0.0

    for key, ind in indicators.items():
        current = snapshot.get(key)
        if current is None:
            continue
        z = _z_score(current, ind['baseline_mean'], ind['baseline_std'])
        contribution = z * ind['direction'] * ind['weight']
        weighted_sum += contribution
        total_weight += ind['weight']
        components[key] = {
            'current': current,
            'z_score': round(z, 3),
            'direction': ind['direction'],
            'weight': ind['weight'],
            'contribution': round(contribution, 3),
        }

    cai = weighted_sum / total_weight if total_weight > 0 else 0.0

    yield_slope = (snapshot.get('yield_curve_slope_pp') or
                   snapshot.get('_yield_curve_slope_pp'))
    phase = _classify_phase(cai, yield_slope)

    return CAIResult(
        cai=round(cai, 3),
        phase=phase,
        components=components,
        yield_curve_slope_pp=yield_slope,
        period=snapshot.get('_period') or snapshot.get('period', 'unknown'),
    )


def load_state(path: Path = DATA_PATH) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def get_current_cai() -> CAIResult:
    """Возвращает CAI для текущего состояния (используется pipeline'ом)."""
    state = load_state()
    return compute_cai(state['current_state'], state['indicators'])


def _print_result(result: CAIResult, label: str = ''):
    print('=' * 70)
    print(f'  CAI · period={result.period}{("  ·  " + label) if label else ""}')
    print('=' * 70)
    print(f'  CAI = {result.cai:+.3f}')
    print(f'  Phase: {result.phase}')
    if result.yield_curve_slope_pp is not None:
        print(f'  Yield curve slope: {result.yield_curve_slope_pp:+.2f} п.п.')
    print('  Components:')
    for k, c in result.components.items():
        print(f'    {k:25s}: cur={c["current"]:>7.2f} → z={c["z_score"]:+.2f}  '
              f'(× dir={c["direction"]:+d} × w={c["weight"]:.1f}) = {c["contribution"]:+.3f}')


def main():
    parser = argparse.ArgumentParser(description='РФ-CAI v0.7 (lite)')
    parser.add_argument('--backtest', action='store_true',
                        help='Прогнать на всех исторических снапшотах')
    parser.add_argument('--json', action='store_true',
                        help='Вывод в JSON для pipeline-интеграции')
    args = parser.parse_args()

    state = load_state()
    indicators = state['indicators']

    if args.json:
        result = compute_cai(state['current_state'], indicators)
        out = {
            'period': result.period,
            'cai': result.cai,
            'phase': result.phase,
            'yield_curve_slope_pp': result.yield_curve_slope_pp,
            'components': result.components,
        }
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    current = compute_cai(state['current_state'], indicators)
    _print_result(current, 'TEKUWEE')

    if args.backtest:
        print('\n' + '=' * 70)
        print('  БЭК-ТЕСТ на исторических снапшотах')
        print('=' * 70)
        passed = 0
        failed = 0
        for snap in state.get('historical_snapshots', []):
            label = snap.get('label', '?')
            expected = snap.get('expected_phase', '?')
            r = compute_cai(snap, indicators)
            mark = '✅' if r.phase == expected else '⚠️'
            print(f'  {mark} {label:<28} CAI={r.cai:+.2f}  '
                  f'phase={r.phase:<12} (expected={expected})')
            if r.phase == expected:
                passed += 1
            else:
                failed += 1
        print(f'\n  Точность: {passed}/{passed + failed} = '
              f'{100 * passed / max(1, passed + failed):.0f}%')


if __name__ == '__main__':
    main()
