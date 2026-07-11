"""КСП budget-PDF харвестер — источник строк oiv-панели из первичных документов.

Извлекает налоговые+неналоговые доходы (графа «Исполнено») бюджета субъекта из годовых
заключений Контрольно-счётных палат (публичные PDF). curl_cffi (browser-TLS) + PyMuPDF.

Робастность парсера:
 * авто-масштаб единиц (отчёты бывают в тыс.руб ИЛИ млн.руб);
 * перенос слова в метке ('неналого-вые'), 'доходы' после чисел;
 * колонка 'Исполнено' = последнее млрд-число ПЕРЕД процентом исполнения (102,5% / 121,4);
 * sum-fallback: раздельные 'Налоговые доходы' + 'Неналоговые доходы';
 * детект скан-PDF (нет текст-слоя) → honest skip.

Валидация: `python scripts/ksp_harvester.py` воспроизводит закоммиченные строки oiv-панели
Пермского края (2018-2024, `data/panel/panel_revenue.csv`) БИТ-В-БИТ из первичных КСП-PDF.
Зависимости curl_cffi/PyMuPDF — extra (не в core requirements.lock); модуль standalone,
ядром/тестами/CI не импортируется.
"""

import re
import sys

MONEY = re.compile(r"\d[\d \xa0]{6,},\d")  # 160 264 502,9; пробел/nbsp, НЕ \n (иначе склейка чисел)
RATIO = re.compile(r"\d{1,3},\d")  # 121,4 (процент исполнения)
LBL_COMBINED = re.compile(r"Налоговые\s+и\s+неналоговые", re.I)
LBL_TAX = re.compile(r"Налоговые\s+доходы", re.I)
LBL_NONTAX = re.compile(r"Неналоговые\s+доходы", re.I)


def bln(s):
    """Строка числа → млрд ₽, авто-масштаб (тыс.руб ИЛИ млн.руб). Выбираем масштаб, дающий
    бюджет-величину [10,900] млрд; диапазоны /1e6 и /1e3 не пересекаются → дизамбигуация
    однозначна для налог+неналог (≥10 млрд)."""
    raw = float(s.replace(" ", "").replace("\xa0", "").replace(",", "."))
    v6, v3 = raw / 1e6, raw / 1e3  # тыс.руб / млн.руб
    if 10 <= v6 <= 900:
        return round(v6, 3)
    if 10 <= v3 <= 900:
        return round(v3, 3)
    return round(v6, 3)


def dehyphen(t):
    """Склеить перенос внутри слова: 'неналого- вые' → 'неналоговые'."""
    return re.sub(r"([а-яёА-ЯЁ])[\-­]\s+([а-яё])", r"\1\2", t)


def _in_money(pos, spans):
    return any(a <= pos < b for a, b in spans)


def _exec_after(win):
    """Исполнено из окна текста после метки: последнее млрд-число перед %-исполнения."""
    money = list(MONEY.finditer(win))
    if not money:
        return None
    spans = [(m.start(), m.end()) for m in money]
    pct = None
    for pm in RATIO.finditer(win):
        if _in_money(pm.start(), spans):
            continue
        v = float(pm.group().replace(",", "."))
        if 30 <= v <= 400:  # процент исполнения бюджета
            pct = pm
            break
    if pct:
        before = [bln(m.group()) for m in money if m.start() < pct.start()]
        return before[-1] if before else None
    vals = [bln(m.group()) for m in money]
    return vals[2] if len(vals) >= 3 else (vals[-1] if vals else None)  # 3-col [утв,уточн,испол]


def _plaus(v):
    return v is not None and 10 <= v <= 900  # бюджет субъекта, млрд ₽


def _first_plausible(t, label):
    """Первый _exec_after среди ВСЕХ вхождений метки, дающий бюджет-масштаб (пропуск
    нарратив-упоминаний метки в тексте)."""
    for m in label.finditer(t):
        v = _exec_after(t[m.end() : m.end() + 220])
        if _plaus(v):
            return v
    return None


def parse_exec(txt):
    """(значение_млрд, метод) или (None, причина)."""
    t = dehyphen(txt)
    v = _first_plausible(t, LBL_COMBINED)
    if v:
        return v, "combined"
    vt = _first_plausible(t, LBL_TAX)  # sum-fallback: раздельные строки
    vn = _first_plausible(t, LBL_NONTAX)
    if _plaus(vt) and vn is not None:
        return round(vt + vn, 3), "sum(tax+nontax)"
    return None, "метка/числа не распознаны"


def harvest(url, timeout=15):
    """Скачать КСП-PDF и распарсить налог+неналог 'Исполнено'. (значение_млрд, метод|причина)."""
    from curl_cffi import requests as cr  # extra-зависимости, ленивый импорт
    import fitz

    try:
        r = cr.get(url, impersonate="chrome", timeout=timeout, verify=False)
    except Exception as e:
        return None, f"DL {e.__class__.__name__}"
    if r.status_code != 200 or "pdf" not in (r.headers.get("content-type", "") or ""):
        return None, f"http {r.status_code}"
    doc = fitz.open(stream=r.content, filetype="pdf")
    txt = "\n".join(p.get_text() for p in doc)
    if len(txt) < 300 * doc.page_count and len(txt) < 3000:
        return None, "скан (нет текст-слоя)"
    return parse_exec(txt)


# Эталон валидации: закоммиченные строки oiv-панели Пермского края (краевой бюджет, КСП).
PERM_REFERENCE = [
    (2018, "https://ksppk.ru/documents/154/z24.pdf", 109.954),
    (2019, "https://ksppk.ru/documents/272/z92_IFOvtpr.pdf", 125.350),
    (2020, "https://ksppk.ru/documents/289/z27_2020.pdf", 100.880),
    (2021, "https://ksppk.ru/documents/329/z17_2022.pdf", 160.265),
    (2022, "https://ksppk.ru/documents/335/zakl_2022.pdf", 160.950),
    (2023, "https://ksppk.ru/documents/359/zakl_2023_h3PzwKt.pdf", 201.589),
    (2024, "https://ksppk.ru/documents/378/zakl_2024.pdf", 220.455),
]


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print("КСП-харвестер — валидация на Пермском крае (эталон = закоммиченная oiv-панель)")
    ok = 0
    for year, url, exp in PERM_REFERENCE:
        v, how = harvest(url)
        match = v is not None and abs(v - exp) < 0.01
        ok += match
        got = f"{v}" if v is not None else f"СБОЙ:{how}"
        print(f"  {'OK ' if match else '!! '} {year}: {got:<12} эталон={exp} [{how}]")
    print(f"Совпало с панелью: {ok}/{len(PERM_REFERENCE)}")


if __name__ == "__main__":
    main()
