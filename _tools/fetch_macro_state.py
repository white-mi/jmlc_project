"""
S4.1 (ROADMAP A1) — авто-обновление current_state в data/macro_state.json.

Раньше current_state правился вручную (на свежей новости обнаруживалось устаревшее
значение КС). Скрипт тянет КС / USD-RUB / Brent / инфляцию из открытых источников
и перезаписывает ТОЛЬКО блок current_state (baseline и historical_snapshots не
трогаются — изоляция данных сохраняется).

Сетевые вызовы graceful: при недоступности источника соответствующее значение
остаётся прежним (берётся из текущего current_state), скрипт не падает.

Использование:
  python fetch_macro_state.py                 # обновить из сети
  python fetch_macro_state.py --dry-run       # показать, что было бы записано
  python fetch_macro_state.py --set key_rate=15 usd_rub=77   # ручной апдейт
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

DATA_PATH = Path(__file__).parent / 'data' / 'macro_state.json'
CURRENT_STATE_KEYS = ('key_rate', 'usd_rub', 'brent_usd', 'inflation_yoy')


# ============================================================
# Сетевые фетчеры (graceful — возвращают None при недоступности)
# ============================================================

def _http_json(url: str, timeout: int = 10):
    import urllib.request
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
        return json.loads(resp.read().decode('utf-8'))


def fetch_usd_rub(timeout: int = 10) -> Optional[float]:
    """USD/RUB из открытого зеркала ЦБ (cbr-xml-daily)."""
    try:
        data = _http_json('https://www.cbr-xml-daily.ru/daily_json.js', timeout)
        return round(float(data['Valute']['USD']['Value']), 2)
    except Exception as e:
        print(f'  ⚠️ USD/RUB недоступен: {e}', file=sys.stderr)
        return None


def _parse_cbr_keyrate_xml(xml_text: str) -> Optional[float]:
    """Извлекает последнюю ставку из ответа CBR KeyRate (SOAP/XML).
    Берёт запись <KR> с максимальной датой <DT>. Чистая функция (тестируется
    без сети), устойчива к namespace (сравнение по local-name тега)."""
    import xml.etree.ElementTree as ET

    def local(tag: str) -> str:
        return tag.split('}')[-1]

    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return None
    best_dt, best_rate = None, None
    for el in root.iter():
        if local(el.tag) != 'KR':
            continue
        dt = rate = None
        for child in el:
            ln = local(child.tag)
            if ln == 'DT':
                dt = (child.text or '').strip()
            elif ln == 'Rate':
                rate = (child.text or '').strip()
        if dt and rate and (best_dt is None or dt > best_dt):
            best_dt, best_rate = dt, rate
    if best_rate:
        try:
            return round(float(best_rate.replace(',', '.')), 2)
        except ValueError:
            return None
    return None


def fetch_key_rate(timeout: int = 10) -> Optional[float]:
    """Ключевая ставка ЦБ через SOAP KeyRate (DailyInfoWebServ). Graceful."""
    soap = (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<soap:Envelope xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/">'
        '<soap:Body><KeyRate xmlns="http://web.cbr.ru/">'
        '<fromDate>2026-01-01</fromDate><ToDate>2026-12-31</ToDate>'
        '</KeyRate></soap:Body></soap:Envelope>'
    )
    url = 'https://www.cbr.ru/DailyInfoWebServ/DailyInfo.asmx'
    try:
        import urllib.request
        req = urllib.request.Request(
            url, data=soap.encode('utf-8'),
            headers={'Content-Type': 'text/xml; charset=utf-8',
                     'SOAPAction': 'http://web.cbr.ru/KeyRate'})
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return _parse_cbr_keyrate_xml(resp.read().decode('utf-8'))
    except Exception as e:
        print(f'  ⚠️ KeyRate недоступен: {e}', file=sys.stderr)
        return None


def _parse_yahoo_chart(json_text: str) -> Optional[float]:
    """Извлекает regularMarketPrice из ответа Yahoo Finance chart API.
    Чистая функция — тестируется без сети."""
    try:
        data = json.loads(json_text)
        meta = data['chart']['result'][0]['meta']
        price = meta.get('regularMarketPrice')
        return round(float(price), 2) if price is not None else None
    except Exception:
        return None


def fetch_brent(timeout: int = 10) -> Optional[float]:
    """Brent (фьючерс BZ=F) — Yahoo Finance chart API. Сетевой вызов graceful."""
    url = ('https://query1.finance.yahoo.com/v8/finance/chart/'
           'BZ=F?interval=1d&range=1d')
    try:
        import urllib.request
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return _parse_yahoo_chart(resp.read().decode('utf-8'))
    except Exception as e:
        print(f'  ⚠️ Brent недоступен: {e}', file=sys.stderr)
        return None


def _parse_worldbank_inflation(json_text: str) -> Optional[float]:
    """Извлекает последнее доступное значение CPI-инфляции (annual %) из ответа
    World Bank API. Чистая функция — тестируется без сети."""
    try:
        data = json.loads(json_text)
        rows = data[1] if isinstance(data, list) and len(data) > 1 else []
        best_date, best_val = None, None
        for r in rows:
            v, d = r.get('value'), r.get('date')
            if v is not None and (best_date is None or str(d) > str(best_date)):
                best_date, best_val = d, v
        return round(float(best_val), 1) if best_val is not None else None
    except Exception:
        return None


def fetch_inflation_yoy(timeout: int = 10) -> Optional[float]:
    """Инфляция (CPI annual %) РФ через World Bank API. Значение годовое и
    лагированное (квартальная оперативность — через Росстат, отдельно). Graceful."""
    url = ('https://api.worldbank.org/v2/country/RU/indicator/'
           'FP.CPI.TOTL.ZG?format=json&per_page=5&mrv=5')
    try:
        import urllib.request
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
            return _parse_worldbank_inflation(resp.read().decode('utf-8'))
    except Exception as e:
        print(f'  ⚠️ Инфляция (World Bank) недоступна: {e}', file=sys.stderr)
        return None


def fetch_all(timeout: int = 10) -> dict:
    """Собирает доступные значения; недоступные опускает (остаются прежними)."""
    raw = {
        'usd_rub': fetch_usd_rub(timeout),
        'key_rate': fetch_key_rate(timeout),
        'brent_usd': fetch_brent(timeout),
        'inflation_yoy': fetch_inflation_yoy(timeout),
    }
    return {k: v for k, v in raw.items() if v is not None}


# ============================================================
# Чистая запись (тестируемая, без сети)
# ============================================================

def update_macro_state(values: dict, path: Path = DATA_PATH,
                       period: Optional[str] = None,
                       note: Optional[str] = None,
                       dry_run: bool = False) -> dict:
    """Обновляет ТОЛЬКО current_state значениями из values (пересечение с
    CURRENT_STATE_KEYS). Возвращает новый current_state. baseline/historical —
    не трогаются. При dry_run не пишет файл."""
    data = json.loads(path.read_text(encoding='utf-8'))
    cur = dict(data.get('current_state', {}))

    applied = {}
    for k in CURRENT_STATE_KEYS:
        if k in values and values[k] is not None:
            cur[k] = values[k]
            applied[k] = values[k]
    if period:
        cur['_period'] = period
    if note:
        cur['_notes'] = note

    if not dry_run:
        data['current_state'] = cur
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                        encoding='utf-8')
    return {'applied': applied, 'current_state': cur}


def _parse_set(pairs: list) -> dict:
    out = {}
    for p in pairs or []:
        if '=' not in p:
            continue
        k, v = p.split('=', 1)
        try:
            out[k.strip()] = float(v)
        except ValueError:
            out[k.strip()] = v
    return out


def main():
    parser = argparse.ArgumentParser(description='S4.1 — обновление current_state')
    parser.add_argument('--dry-run', action='store_true',
                        help='Показать, что было бы записано, без записи')
    parser.add_argument('--set', nargs='*', default=None,
                        help='Ручной апдейт key=value (например key_rate=15)')
    parser.add_argument('--period', default=None, help='Метка периода current_state')
    parser.add_argument('--note', default=None, help='Заметка к current_state')
    args = parser.parse_args()

    values = _parse_set(args.set) if args.set else fetch_all()
    if not values:
        print('  ⚠️ Нет новых значений (сеть недоступна и --set не задан). '
              'current_state без изменений.', file=sys.stderr)
        return

    result = update_macro_state(values, period=args.period, note=args.note,
                                dry_run=args.dry_run)
    mode = 'DRY-RUN' if args.dry_run else 'WRITTEN'
    print(f'  [{mode}] applied: {result["applied"]}')
    for k in CURRENT_STATE_KEYS:
        print(f'    {k}: {result["current_state"].get(k)}')


if __name__ == '__main__':
    main()
