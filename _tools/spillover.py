"""
L2 — Industry Spillover (v0.7 минимальная версия).

Numeric matrix-based трансмиссия шока между 7 отраслями Радара.
Веса заданы в `data/spillover_matrix.json` на основе качественной карты
Fialkowski 2025 supply-chain stress test (H/M/L → числа 0.0-1.0).

API:
  propagate_shock(source_industry, magnitude_pp) → {industry: ΔPD_pp}

magnitude_pp — масштаб первичного шока в процентных пунктах ΔPD (например, 0.8).
Для отрасли TO эффект: matrix[FROM][TO] × magnitude_pp.

Что НЕ делает (TODO v0.9):
  - Diebold-Yilmaz VAR на отраслевых индексах (нужен 3+ года истории revenue)
  - DebtRank каскад (нужны capital buffers топ-50 заёмщиков)
  - Multi-step propagation (currently 1-hop only)

Использование:
  python spillover.py --shock oilgas --magnitude 0.8
  python spillover.py --shock metallurgy --magnitude 1.5 --json
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

DATA_PATH = Path(__file__).parent / 'data' / 'spillover_matrix.json'


@dataclass
class SpilloverResult:
    source: str
    magnitude_pp: float
    impacts: dict  # {industry: ΔPD_pp}
    ranked: list   # [(industry, ΔPD_pp), ...] sorted desc


def load_matrix(path: Path = DATA_PATH) -> tuple[dict, list]:
    """Возвращает (matrix_dict, industries_list)."""
    data = json.loads(path.read_text(encoding='utf-8'))
    return data['matrix'], data['industries']


def propagate_shock(source_industry: str,
                    magnitude_pp: float = 1.0,
                    matrix_path: Path = DATA_PATH) -> SpilloverResult:
    """Распространяет шок от source_industry на все 7 отраслей.

    Args:
        source_industry: ключ из ['metallurgy', 'oilgas', 'chemistry',
                                   'retail', 'energy', 'pharma', 'oiv']
        magnitude_pp: масштаб шока в процентных пунктах ΔPD

    Returns:
        SpilloverResult с impacts={industry: ΔPD_pp} и ranked-list
    """
    matrix, industries = load_matrix(matrix_path)
    if source_industry not in matrix:
        raise ValueError(
            f'Неизвестная отрасль: {source_industry}. '
            f'Допустимые: {industries}'
        )

    row = matrix[source_industry]
    impacts = {ind: round(row[ind] * magnitude_pp, 3) for ind in industries}
    ranked = sorted(impacts.items(), key=lambda x: x[1], reverse=True)

    return SpilloverResult(
        source=source_industry,
        magnitude_pp=magnitude_pp,
        impacts=impacts,
        ranked=ranked,
    )


# S3.4: severity (0-100) → magnitude (п.п. ΔPD). Раньше magnitude была фикс 0.8
# для любого шока; теперь масштаб выводится из силы шока L0.
def severity_to_magnitude(severity_score: float, scale: float = 1.5) -> float:
    """Линейно: severity 0→0, 50→0.75, 100→1.5 п.п. ΔPD."""
    s = max(0.0, min(100.0, float(severity_score)))
    return round(s / 100.0 * scale, 3)


# S3.4: относительные веса broad credit channel для шоков ставки ЦБ (категория 4) —
# несколько отраслей бьются одновременно (retail сильнее всего, дальше по убыванию).
CREDIT_CHANNEL_WEIGHTS = {
    'retail': 1.0, 'oiv': 0.75, 'metallurgy': 0.5, 'chemistry': 0.5, 'oilgas': 0.4,
}


def propagate_multi_source(sources: dict,
                           matrix_path: Path = DATA_PATH,
                           aggregate: str = 'max') -> SpilloverResult:
    """S3.4: распространяет шок от НЕСКОЛЬКИХ отраслей-источников, агрегируя по target.

    Args:
        sources: {industry: magnitude_pp} — источники с их силой
        aggregate: 'max' (худший сценарий по target) | 'sum' (накопление)
    """
    matrix, industries = load_matrix(matrix_path)
    agg = {ind: 0.0 for ind in industries}
    for src, mag in sources.items():
        if src not in matrix:
            continue
        row = matrix[src]
        for ind in industries:
            val = row[ind] * mag
            agg[ind] = (agg[ind] + val) if aggregate == 'sum' else max(agg[ind], val)
    impacts = {ind: round(agg[ind], 3) for ind in industries}
    ranked = sorted(impacts.items(), key=lambda x: x[1], reverse=True)
    return SpilloverResult(
        source='multi:' + ','.join(sources.keys()),
        magnitude_pp=round(max(sources.values()), 3) if sources else 0.0,
        impacts=impacts,
        ranked=ranked,
    )


def propagate_credit_channel(magnitude_pp: float = 0.8,
                             matrix_path: Path = DATA_PATH) -> SpilloverResult:
    """S3.4: broad credit channel для шоков ставки ЦБ (категория 4) —
    5 debt-чувствительных отраслей бьются одновременно с разной силой."""
    sources = {ind: round(w * magnitude_pp, 3)
               for ind, w in CREDIT_CHANNEL_WEIGHTS.items()}
    return propagate_multi_source(sources, matrix_path, aggregate='max')


def matrix_invariants_check(matrix_path: Path = DATA_PATH) -> dict:
    """Базовые инварианты для тестов:
       - Диагональ = 1.0 для всех отраслей
       - Веса в [0; 1]
       - Все строки имеют одинаковый набор колонок
    """
    matrix, industries = load_matrix(matrix_path)
    issues = []
    for ind in industries:
        row = matrix.get(ind, {})
        if set(row.keys()) != set(industries):
            issues.append(f'{ind}: row keys != industries set')
        if abs(row.get(ind, 0) - 1.0) > 1e-6:
            issues.append(f'{ind}: diagonal != 1.0 (got {row.get(ind)})')
        for col, val in row.items():
            if not (0.0 <= val <= 1.0):
                issues.append(f'{ind}→{col}: weight {val} out of [0,1]')
    return {
        'ok': len(issues) == 0,
        'industries_count': len(industries),
        'issues': issues,
    }


def main():
    parser = argparse.ArgumentParser(description='L2 Spillover v0.7')
    parser.add_argument('--shock', help='Отрасль-источник шока (required если без --check)')
    parser.add_argument('--magnitude', type=float, default=1.0,
                        help='Масштаб шока в п.п. ΔPD (default: 1.0)')
    parser.add_argument('--json', action='store_true',
                        help='Вывод JSON')
    parser.add_argument('--check', action='store_true',
                        help='Только проверить инварианты матрицы')
    args = parser.parse_args()

    if args.check:
        result = matrix_invariants_check()
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print('=' * 70)
            print(f'  Matrix invariants: {"✅ OK" if result["ok"] else "❌ ISSUES"}')
            print('=' * 70)
            print(f'  Industries: {result["industries_count"]}')
            for issue in result['issues']:
                print(f'  ⚠️ {issue}')
        return

    if not args.shock:
        parser.error('--shock обязателен (либо используй --check)')
    result = propagate_shock(args.shock, args.magnitude)

    if args.json:
        print(json.dumps({
            'source': result.source,
            'magnitude_pp': result.magnitude_pp,
            'impacts': result.impacts,
            'ranked': result.ranked,
        }, ensure_ascii=False, indent=2))
        return

    print('=' * 70)
    print(f'  Spillover: {result.source} → 7 отраслей')
    print(f'  Шок magnitude: {result.magnitude_pp:+.2f} п.п. ΔPD')
    print('=' * 70)
    for ind, impact in result.ranked:
        bar_len = int(impact / max(result.magnitude_pp, 0.01) * 40)
        bar = '█' * max(0, bar_len)
        marker = '🔴' if impact >= 0.5 else ('🟡' if impact >= 0.2 else '🟢')
        print(f'  {marker} {ind:<12} {impact:+.3f} п.п. {bar}')


if __name__ == '__main__':
    main()
