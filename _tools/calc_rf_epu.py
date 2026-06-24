"""
РФ-EPU lite (Economic Policy Uncertainty index, v0.7).

Реализация принципа Baker-Bloom-Davis (2016):
   Считаем частоту слов из триады словарей в текстах:
     - ECONOMIC: экономика, экономический, ВВП, рынок, инфляция, рубль...
     - UNCERTAINTY: неопределённость, риск, угроза, нестабильность, шок...
     - POLICY/CRISIS: ставка, ЦБ, санкции, кризис, регулирование, война...

   Текст считается uncertainty-related если содержит ≥1 слово из каждого словаря.
   EPU = (#таких текстов в окне) / (#всех текстов в окне) × 100

Корпус: `_Анализы/*.md` (анализы новостей с frontmatter).
Группировка: по месяцу из frontmatter `дата_новости` или имени файла YYYY-MM-DD.

Бэк-тест:
  Реальная валидация требует архива анализов за 2017-2026 (нет в текущей версии).
  Текущий бэк-тест: показать что текущий период (апрель 2026) даёт высокий EPU
  на фоне базы (если корпус ~однородный) и проверить что любой апрельский анализ
  с шоком категории «1.» даёт более 50% EPU score.

TODO v0.8:
  - Подключить feedparser к 5 RSS (ТАСС, Коммерсантъ, РБК, Ведомости, Forbes)
  - Считать EPU на полном новостном архиве, не только на анализах Радара
  - Калибровка на исторических точках (03.2022 — пик; 06.2023 — нормализация)
  - Альтернативный fallback: US-EPU из FRED как proxy

Использование:
  python calc_rf_epu.py                    # current EPU
  python calc_rf_epu.py --window-days 30   # с окном
  python calc_rf_epu.py --json             # для интеграции pipeline
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ANALYSES_DIR = Path(__file__).parent.parent / '_Анализы'

# Словари для Baker-Bloom-Davis (адаптация для русского)
ECONOMIC_TERMS = {
    'эконом', 'ввп', 'инфляц', 'рубл', 'валют', 'курс', 'рынок', 'рынк',
    'банк', 'кредит', 'долг', 'выручк', 'прибыл', 'спрос', 'предложен',
    'отрасл', 'компани', 'эмитент', 'облигаци', 'акци',
    'торговл', 'экспорт', 'импорт', 'нефт', 'газ', 'бюджет',
}

UNCERTAINTY_TERMS = {
    'неопредел', 'риск', 'угроз', 'нестабильн', 'шок', 'волатил',
    'неясн', 'неуверен', 'опасен', 'опасн', 'тревог',
    'обвал', 'коллапс', 'обвал', 'паник', 'падени', 'кризис',
}

POLICY_CRISIS_TERMS = {
    'ставк', 'ключев', 'цб', 'центробанк', 'минфин', 'минэк',
    'санкци', 'эмбарго', 'запрет', 'ограничен',
    'регулир', 'указ', 'постановлен', 'решени',
    'кризис', 'войн', 'конфликт', 'удар', 'атак',
    'дефицит', 'дефолт', 'дефлят',
}

# Sentiment-словари для разделения negative_uncertainty (плохие новости)
# vs neutral_uncertainty (просто высокая частота policy/economic слов).
# Слова из «двойного» контекста (снижен/рост) не включены — они ambiguous.
NEGATIVE_SENTIMENT_TERMS = {
    'обвал', 'рухн', 'кризис', 'паник', 'катастроф', 'разоча',
    'хуже', 'ущерб', 'убыт', 'потер', 'провал', 'дефолт',
    'санкци', 'эмбарго', 'удар', 'атак', 'войн', 'эскалац',
    'угроз', 'риск', 'опасен', 'опасн', 'тревог',
    'ослаб', 'ускори.*инфляц', 'дефицит',
}


@dataclass
class EPUResult:
    epu_value: float          # ∈ [0; 100]: доля uncertainty-текстов
    epu_negative_pct: float   # ∈ [0; 100]: доля negative-uncertainty
                              #   среди uncertainty-текстов
    n_uncertainty_texts: int
    n_negative_texts: int
    n_total_texts: int
    period_label: str
    matched_files: list[str]
    negative_files: list[str]
    epu_degraded: bool = False   # S2.2: True если корпус в окне слишком мал
    source: str = 'corpus'       # 'corpus' | 'fred'


def _tokenize(text: str) -> set[str]:
    """Простая токенизация: слова → нижний регистр, без знаков препинания."""
    text = text.lower()
    words = re.findall(r'\w+', text, re.UNICODE)
    return set(words)


def _has_term_from(words: set[str], dictionary: set[str]) -> bool:
    """True если ≥1 слово из words начинается с любого префикса из dictionary."""
    for w in words:
        for term in dictionary:
            if w.startswith(term):
                return True
    return False


def _is_uncertainty_text(text: str) -> bool:
    """Текст считается uncertainty-related, если содержит ≥1 слово из каждого
    из 3 словарей (Baker-Bloom-Davis triple-criterion)."""
    words = _tokenize(text)
    return (_has_term_from(words, ECONOMIC_TERMS) and
            _has_term_from(words, UNCERTAINTY_TERMS) and
            _has_term_from(words, POLICY_CRISIS_TERMS))


def _is_negative_uncertainty(text: str) -> bool:
    """Текст negative-uncertainty: uncertainty + ≥1 negative sentiment слово."""
    if not _is_uncertainty_text(text):
        return False
    words = _tokenize(text)
    return _has_term_from(words, NEGATIVE_SENTIMENT_TERMS)


def _extract_date(file_path: Path) -> Optional[datetime]:
    """Извлекает дату из имени файла YYYY-MM-DD или из frontmatter."""
    name = file_path.name
    match = re.match(r'(\d{4}-\d{2}-\d{2})', name)
    if match:
        try:
            return datetime.strptime(match.group(1), '%Y-%m-%d')
        except ValueError:
            pass
    # fallback: парсинг frontmatter
    try:
        text = file_path.read_text(encoding='utf-8')
        fm = re.search(r'^---\s*\n(.*?)\n---', text, re.DOTALL)
        if fm:
            for line in fm.group(1).splitlines():
                m = re.match(r'\s*дата[_а-я]*:\s*"?(\d{4}-\d{2}-\d{2})"?', line)
                if m:
                    return datetime.strptime(m.group(1), '%Y-%m-%d')
    except Exception:
        pass
    return None


def compute_epu(
    analyses_dir: Path = ANALYSES_DIR,
    window_days: Optional[int] = None,
    end_date: Optional[datetime] = None,
    min_texts: int = 3,
) -> EPUResult:
    """Считает EPU на корпусе анализов в заданном окне.
    Возвращает обе метрики: общий EPU и долю negative-uncertainty.
    S2.2: при числе текстов в окне < min_texts ставит epu_degraded=True
    (вместо тихого 0.0, который читается как «нет неопределённости»)."""
    if not analyses_dir.exists():
        return EPUResult(0.0, 0.0, 0, 0, 0, 'no analyses', [], [],
                         epu_degraded=True)

    end_date = end_date or datetime.now()
    start_date = end_date - timedelta(days=window_days) if window_days else None

    total = 0
    matched_files = []
    negative_files = []

    for md_file in sorted(analyses_dir.glob('*.md')):
        file_date = _extract_date(md_file)
        if file_date is None:
            continue
        if start_date and file_date < start_date:
            continue
        if file_date > end_date:
            continue

        try:
            text = md_file.read_text(encoding='utf-8')
        except Exception:
            continue
        total += 1
        if _is_uncertainty_text(text):
            matched_files.append(md_file.name)
            if _is_negative_uncertainty(text):
                negative_files.append(md_file.name)

    epu = 100.0 * len(matched_files) / max(1, total)
    neg_pct = (100.0 * len(negative_files) / max(1, len(matched_files))) if matched_files else 0.0
    period = (f"{start_date.strftime('%Y-%m-%d')} ... "
              f"{end_date.strftime('%Y-%m-%d')}") if start_date else f"all ... {end_date.strftime('%Y-%m-%d')}"

    return EPUResult(
        epu_value=round(epu, 1),
        epu_negative_pct=round(neg_pct, 1),
        n_uncertainty_texts=len(matched_files),
        n_negative_texts=len(negative_files),
        n_total_texts=total,
        period_label=period,
        matched_files=matched_files,
        negative_files=negative_files,
        epu_degraded=(total < min_texts),
        source='corpus',
    )


def fetch_us_epu_fred(timeout: int = 10) -> Optional[float]:
    """S2.2: fallback — последнее значение US-EPU (FRED USEPUINDXD) как proxy
    global-uncertainty. Возвращает float или None при недоступности сети.
    Сетевой вызов — НЕ используется автоматически в pipeline (только CLI)."""
    url = ('https://fred.stlouisfed.org/graph/fredgraph.csv?id=USEPUINDXD')
    try:
        import urllib.request
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
            lines = resp.read().decode('utf-8').strip().splitlines()
        for line in reversed(lines[1:]):
            parts = line.split(',')
            if len(parts) >= 2 and parts[1] not in ('', '.'):
                return float(parts[1])
    except Exception as e:
        print(f'  ⚠️ FRED US-EPU недоступен: {e}', file=sys.stderr)
    return None


def get_current_epu(window_days: int = 30,
                    end_date: Optional[datetime] = None,
                    min_texts: int = 3) -> EPUResult:
    """Возвращает EPU за последние N дней (используется pipeline'ом).
    S2.2: end_date позволяет якорить окно на дате новости, а не на now()."""
    return compute_epu(window_days=window_days, end_date=end_date,
                       min_texts=min_texts)


def main():
    parser = argparse.ArgumentParser(description='РФ-EPU lite v0.9')
    parser.add_argument('--window-days', type=int, default=None,
                        help='Окно в днях; если не задано — весь корпус')
    parser.add_argument('--end-date', default=None,
                        help='Якорная дата окна YYYY-MM-DD (по умолчанию now)')
    parser.add_argument('--source', choices=['corpus', 'fred'], default='corpus',
                        help='corpus — анализы Радара; fred — US-EPU proxy (сеть)')
    parser.add_argument('--json', action='store_true',
                        help='Вывод в JSON')
    args = parser.parse_args()

    end_date = (datetime.strptime(args.end_date, '%Y-%m-%d')
                if args.end_date else None)

    if args.source == 'fred':
        val = fetch_us_epu_fred()
        result = EPUResult(val or 0.0, 0.0, 0, 0, 0, 'FRED USEPUINDXD',
                           [], [], epu_degraded=(val is None), source='fred')
    else:
        result = compute_epu(window_days=args.window_days, end_date=end_date)

    if args.json:
        print(json.dumps({
            'epu_value': result.epu_value,
            'epu_negative_pct': result.epu_negative_pct,
            'n_uncertainty_texts': result.n_uncertainty_texts,
            'n_negative_texts': result.n_negative_texts,
            'n_total_texts': result.n_total_texts,
            'period_label': result.period_label,
            'epu_degraded': result.epu_degraded,
            'source': result.source,
        }, ensure_ascii=False, indent=2))
        return

    print('=' * 70)
    print(f'  РФ-EPU lite · {result.period_label}')
    print('=' * 70)
    print(f'  EPU general:  {result.epu_value:.1f} / 100  '
          f'({result.n_uncertainty_texts}/{result.n_total_texts} текстов)')
    print(f'  EPU negative: {result.epu_negative_pct:.1f}%  '
          f'({result.n_negative_texts}/{result.n_uncertainty_texts} negative '
          f'из uncertainty)')
    if result.negative_files:
        print(f'  Negative uncertainty:')
        for f in result.negative_files[:10]:
            print(f'    • {f}')
        if len(result.negative_files) > 10:
            print(f'    ... и ещё {len(result.negative_files) - 10}')


if __name__ == '__main__':
    main()
