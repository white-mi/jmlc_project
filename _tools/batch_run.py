"""
Batch-перегон пайплайна Макро-радара на наборе новостей.

Берёт:
  - _tools/output/batch_news_<date>.json   — тексты новостей (date, source, title, news_text)
  - _tools/output/batch_l0_<date>.json     — L0-классификации от 5-агентного конвейера

Для каждой новости прогоняет numeric-слои L1/L1.5/L2/L3 через run_full_pipeline
в smoke-режиме с подкатегорией и отраслью из L0, патчит блок L0 реальными
данными (вместо generic-заглушки smoke) и пишет:
  - индивидуальный markdown в _Анализы/_batch/<date> — <id> — <slug>.md
  - сводный сравнительный отчёт в _tools/output/Перегон <N> новостей — <date>.md

Использование:
  python batch_run.py --date 2026-06-14
"""

import argparse
import json
import re
import sys
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

TOOLS_DIR = Path(__file__).parent
sys.path.insert(0, str(TOOLS_DIR))

from run_pipeline import run_full_pipeline, render_markdown  # noqa: E402

RADAR_ROOT = TOOLS_DIR.parent
OUTPUT_DIR = TOOLS_DIR / 'output'
BATCH_DIR = RADAR_ROOT / '_Анализы' / '_batch'


def slugify(text: str, max_len: int = 48) -> str:
    text = re.sub(r'[^\w\sА-Яа-яЁё-]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text[:max_len].strip()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def worst_best_segment(l3: dict):
    """Возвращает (worst_sgmt, worst_pd, best_sgmt, best_pd) по ΔPD."""
    items = [(s, d['delta_pd_pp']) for s, d in l3.items()]
    if not items:
        return ('—', 0.0, '—', 0.0)
    worst = max(items, key=lambda x: x[1])
    best = min(items, key=lambda x: x[1])
    return (worst[0], worst[1], best[0], best[1])


def is_bifurcated(l3: dict, thr: float = 0.05) -> bool:
    pos = any(d['delta_pd_pp'] > thr for d in l3.values())
    neg = any(d['delta_pd_pp'] < -thr for d in l3.values())
    return pos and neg


def main():
    parser = argparse.ArgumentParser(description='Batch-перегон Макро-радара')
    parser.add_argument('--date', required=True, help='Дата набора (YYYY-MM-DD)')
    args = parser.parse_args()
    date = args.date

    news = load_json(OUTPUT_DIR / f'batch_news_{date}.json')
    l0 = load_json(OUTPUT_DIR / f'batch_l0_{date}.json')

    news_by_id = {it['id']: it for it in news['items']}
    l0_by_id = {it['id']: it for it in l0['items']}
    ids = [it['id'] for it in l0['items']]

    BATCH_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    l1_snapshot = None

    for nid in ids:
        n = news_by_id[nid]
        c = l0_by_id[nid]
        subcat = c['subcategory']
        industry = c['primary_industry']

        state = run_full_pipeline(
            news_text=n['news_text'], source=n['source'], date=n['date'],
            smoke_shock=subcat, smoke_industry=industry,
            smoke_severity=c.get('severity_score', 60),  # S3.4: реальная сила → L2 magnitude
        )

        # Патчим L0 реальными данными 5-агентного конвейера
        state['L0_classification'] = {
            'WHAT': c['WHAT'],
            'main_category': subcat.split('.')[0],
            'subcategory': subcat,
            'severity_score': c['severity_score'],
            'severity_level': c['severity_level'],
            'direction': c['direction'],
            'shortcut': c.get('shortcut', False),
            'cumulative_boost_applied': c.get('cumulative_boost_applied', False),
            'mode': '5-agent-llm',
        }

        if l1_snapshot is None:
            l1_snapshot = state['L1_macro'] | {'kc_regime': state['kc_regime']}

        md = render_markdown(state)
        slug = slugify(n['title'])
        fpath = BATCH_DIR / f'{n["date"]} — {nid} — {slug}.md'
        fpath.write_text(md, encoding='utf-8')

        l2 = state['L2_spillover']
        l2_top_ind, l2_top_pd = l2['ranked'][0]
        l3 = state['L3_segments']
        w_s, w_pd, b_s, b_pd = worst_best_segment(l3)
        # multi-source (credit channel категории 4) — намеренная маршрутизация,
        # не считаем расхождением LLM↔pipeline.
        is_multi = str(l2['source']).startswith('multi')
        is_diverge = (not is_multi) and (industry != l2['source'])

        rows.append({
            'id': nid,
            'title': n['title'],
            'date': n['date'],
            'subcat': subcat,
            'sev': f'{c["severity_score"]} {c["severity_level"]}',
            'dir': c['direction'],
            'shortcut': c.get('shortcut', False),
            'llm_industry': industry,
            'l2_source': l2['source'],
            'is_diverge': is_diverge,
            'l2_top': f'{l2_top_ind} +{l2_top_pd:.2f}',
            'worst_seg': f'{w_s} {w_pd:+.2f}',
            'best_seg': f'{b_s} {b_pd:+.2f}',
            'bifurcated': is_bifurcated(l3),
            'file': fpath.name,
        })
        print(f'  ✅ {nid} [{subcat}] → {fpath.name}', file=sys.stderr)

    # ---- Сводный отчёт ----
    n_total = len(rows)
    n_bif = sum(1 for r in rows if r['bifurcated'])
    n_short = sum(1 for r in rows if r['shortcut'])
    n_diverge = sum(1 for r in rows if r['is_diverge'])

    out = []
    out.append('---')
    out.append('tags: [макро-радар, pipeline, batch, перегон]')
    out.append(f'дата_перегона: "{date}"')
    out.append('pipeline_version: "0.8-mvp"')
    out.append(f'новостей: {n_total}')
    out.append('---\n')
    out.append(f'# Перегон пайплайна на {n_total} новостях — {date}\n')
    out.append('> L0 — реальный 5-агентный LLM-конвейер (подагенты Claude Code). '
               'L1/L1.5/L2/L3 — numeric `run_full_pipeline` (smoke-вход = LLM-классификация).\n')

    # L1 — общий снимок
    out.append('## L1 — макро-снимок (общий для всех новостей)\n')
    if l1_snapshot:
        out.append(f'- **CAI:** {l1_snapshot["cai"]:+.2f} → **{l1_snapshot["phase"]}**')
        out.append(f'- **EPU (30d):** {l1_snapshot["epu"]}/100, '
                   f'Negative-EPU {l1_snapshot["epu_negative_pct"]}%')
        out.append(f'- **Yield curve slope:** {l1_snapshot["yield_curve_slope_pp"]} п.п.')
        out.append(f'- **Режим КС:** {l1_snapshot["kc_regime"]} (КС 14.5%)\n')
    out.append('> ⚠️ В smoke-режиме L1 одинаков для всех новостей — различаются '
               'L0, L1.5-forward, L2 и L3.\n')

    # Сводная таблица
    out.append('## Сводная таблица\n')
    out.append('| # | Новость | Подкат. | Сила | Dir | L2 источник→топ | Худший сегмент (ΔPD) | Лучший сегмент (ΔPD) | Бифурк. |')
    out.append('|---|---|---|---|---|---|---|---|---|')
    for i, r in enumerate(rows, 1):
        title = r['title'][:42]
        dirc = '−1' if r['dir'] == -1 else '+1'
        sc = ' 🔇' if r['shortcut'] else ''
        bif = '✅' if r['bifurcated'] else '—'
        out.append(
            f'| {i} | {title}{sc} | {r["subcat"]} | {r["sev"]} | {dirc} | '
            f'{r["l2_source"]}→{r["l2_top"]} | {r["worst_seg"]} | {r["best_seg"]} | {bif} |'
        )
    out.append('')

    # Наблюдения
    out.append('## Наблюдения\n')
    out.append(f'- **Бифуркаций (знак ΔPD расходится между сегментами):** {n_bif}/{n_total}')
    out.append(f'- **Информационный шум (shortcut=true):** {n_short}/{n_total} '
               '(в production — однострочная пометка без L1-L3)')
    out.append(f'- **Расхождений маршрутизации отрасли (LLM primary_industry ≠ L2 source pipeline):** '
               f'{n_diverge}/{n_total}')
    if n_diverge:
        out.append('\n  Расхождения (LLM → pipeline):')
        for r in rows:
            if r['is_diverge']:
                out.append(f'  - {r["id"]} [{r["subcat"]}]: LLM={r["llm_industry"]} → pipeline={r["l2_source"]}')
        out.append('\n  Причина: маршрутизация (`data/shock_to_industries.json`) задана на '
                   'уровне ПОДКАТЕГОРИИ (primary = первая отрасль списка), а не конкретной '
                   'новости. Разные новости одной подкатегории могут иметь разную primary-'
                   'отрасль (напр. 1.4: primary=metallurgy для экспорта металлов, но новость '
                   'о нефтетрейдерах → LLM ставит oilgas). Это не баг карты (она покрывает все '
                   '27 подкат), а гранулярность; опция — учитывать LLM primary_industry, если '
                   'он входит в mapped-список подкатегории.')
    out.append('')

    # Подробные one-line по новостям
    out.append('## L0 one-line по каждой новости\n')
    for i, nid in enumerate(ids, 1):
        c = l0_by_id[nid]
        out.append(f'{i}. **[{c["subcategory"]}] {news_by_id[nid]["title"]}** — {c["one_line"]}')
    out.append('')
    out.append('## Индивидуальные отчёты\n')
    out.append(f'Сохранены в `_Анализы/_batch/` ({n_total} файлов).')
    out.append('')
    out.append(f'*Перегон {n_total} новостей · pipeline v0.8 · {date}*')

    report_path = OUTPUT_DIR / f'Перегон {n_total} новостей — {date}.md'
    report_path.write_text('\n'.join(out), encoding='utf-8')
    print(f'\n✅ Сводный отчёт: {report_path}', file=sys.stderr)
    print(f'✅ Индивидуальных отчётов: {n_total} в {BATCH_DIR}', file=sys.stderr)
    print(f'   Бифуркаций: {n_bif}/{n_total} · shortcut: {n_short}/{n_total} · '
          f'расхождений маршрутизации: {n_diverge}/{n_total}', file=sys.stderr)


if __name__ == '__main__':
    main()
