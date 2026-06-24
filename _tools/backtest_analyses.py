"""
backtest_analyses.py — сводка корпуса ручных разборов новостей (`_Анализы/`).

Продуктовый слой / «обратная связь от рынка»: трактует корпус сохранённых анализов как
**proxy-feedback** — воспроизводимый «трек-рекорд» вызовов радара. Делает корпус
вычислимым (а не «18 файлов на словах»): считает объём, период, долю OSL/бэктест-разборов.

ИЗОЛЯЦИЯ/ДИСЦИПЛИНА: только ЧИТАЕТ `../_Анализы/*.md`; ничего туда не пишет; не трогает
``. Stdlib-only (без pyyaml/numpy), чтобы не тянуть зависимости и
не ломать CI. Вывод `--emit` идёт в `_tools/output/backtest/feedback.md`.

Запуск:
    cd _tools
    python backtest_analyses.py            # печатает сводку в stdout
    python backtest_analyses.py --emit      # дополнительно пишет output/backtest/feedback.md
"""

import argparse
import glob
import os
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_HERE = os.path.dirname(os.path.abspath(__file__))
# Корпус разборов лежит на уровень выше _tools/
ANALYSES_DIR = os.path.normpath(os.path.join(_HERE, "..", "_Анализы"))

_DATE_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
_FM_LINE_RE = re.compile(r"^([A-Za-zА-Яа-я_]+):\s*(.*)$")


def _parse_frontmatter(text):
    """Минимальный парс YAML-фронтматтера верхнего уровня → dict. Без pyyaml."""
    fm = {}
    if not text.startswith("---"):
        return fm
    end = text.find("\n---", 3)
    if end == -1:
        return fm
    for line in text[3:end].splitlines():
        m = _FM_LINE_RE.match(line)
        if m:
            fm[m.group(1)] = m.group(2).strip().strip('"').strip("'")
    return fm


def _is_osl_or_backtest(name):
    # Эвристика по имени файла (не по содержимому): помечает dev-разборы OSL/conformal/бэктеста.
    # Сводный счётчик n_osl — производная от имён файлов, поэтому устойчив, но чувствителен к ренеймам.
    low = name.lower()
    return "бэк-тест" in low or "бэктест" in low or "osl" in low or "conformal" in low


def load_corpus(directory=None):
    """Читает все *.md из `_Анализы/` (без подпапки `_batch`). Возвращает список dict.

    directory=None → берётся `ANALYSES_DIR` в момент вызова (позволяет тестам подменять каталог).
    """
    if directory is None:
        directory = ANALYSES_DIR
    items = []
    for path in sorted(glob.glob(os.path.join(directory, "*.md"))):
        name = os.path.basename(path)
        with open(path, encoding="utf-8") as f:
            text = f.read()
        fm = _parse_frontmatter(text)
        # дата из имени файла приоритетна (формат 'YYYY-MM-DD — ...'),
        # иначе — из фронтматтера.
        m = _DATE_RE.search(name) or _DATE_RE.search(
            fm.get("дата_анализа", "") or fm.get("дата", "")
        )
        items.append(
            {
                "file": name,
                "date": m.group(0) if m else None,
                "tags": fm.get("tags", ""),
                "is_osl_backtest": _is_osl_or_backtest(name),
                "size": len(text),
            }
        )
    return items


def summarize_corpus(directory=None):
    """Агрегированная сводка корпуса (без побочных эффектов)."""
    items = load_corpus(directory)
    dated = [it for it in items if it["date"]]
    return {
        "n_total": len(items),
        "n_dated": len(dated),
        "date_min": min((it["date"] for it in dated), default=None),
        "date_max": max((it["date"] for it in dated), default=None),
        "n_osl": sum(1 for it in items if it["is_osl_backtest"]),
        "files": [it["file"] for it in items],
    }


def render_summary(summary, items=None):
    """Markdown-рендер сводки. `items` — для таблицы (если None, перечитываем корпус)."""
    if items is None:
        items = load_corpus()
    lines = [
        "# Сводка корпуса `_Анализы/` (proxy-feedback)",
        "",
        f"- Всего разборов: **{summary['n_total']}**",
        f"- С датой: {summary['n_dated']} "
        f"(период {summary['date_min']} … {summary['date_max']})",
        f"- OSL / conformal / бэктест-разборов: {summary['n_osl']}",
        "",
        "| Файл | Дата |",
        "|---|---|",
    ]
    for it in items:
        lines.append(f"| {it['file']} | {it['date'] or '—'} |")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--emit", action="store_true", help="записать output/backtest/feedback.md"
    )
    args = ap.parse_args(argv)

    items = load_corpus()
    summary = summarize_corpus()
    text = render_summary(summary, items)
    print(text)

    if args.emit:
        out_dir = os.path.join(_HERE, "output", "backtest")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "feedback.md")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"\n[emit] {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
