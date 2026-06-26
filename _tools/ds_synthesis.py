"""
Кросс-отраслевой DS-синтез (Макро-радар) — агрегатор уже посчитанных walk-forward результатов.

`python ds_synthesis.py` читает 4 файла output/osl_metrics/<отрасль>_metrics.json (их пишет
osl_walkforward.py) и строит:
  • docs/figures/synthesis/mape_by_industry.png — MAPE_common по отраслям × модели (structural /
    лучшая learned / persistence). Воспроизводимо из JSON, числа не вводятся руками.
  • docs/figures/synthesis/conformal_coverage.png — энергетика (N=30, самый чистый случай):
    OOS-покрытие structural vs stale-моделей (ценность операционного сигнала).
  • docs/figures/synthesis/summary_table.md — сводная таблица (N, победитель, MAPE, DM p).

ВАЖНО (честность): conformal-покрытие в JSON НЕ хранится → числа в CONFORMAL транскрибированы из
DS_REPORT_*.md (источник в каждой записи). Это визуализация уже опубликованных/отревьюенных чисел,
а не новый расчёт. MAPE/DM — напрямую из JSON walk-forward.

Зависимости — только в extra [eda] (matplotlib). Без сети.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# matplotlib импортится ЛЕНИВО внутри plot_*-функций: агрегация чисел (load/assemble/таблица) —
# чистый core (numpy), а тяжёлый matplotlib только для графиков (extra [eda]). Так `import
# ds_synthesis` работает в core-CI без [eda], и data-тесты гоняются без графического стека.

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

TOOLS = Path(__file__).parent
sys.path.insert(0, str(TOOLS))
import osl_panel          # noqa: E402

METRICS_DIR = TOOLS / 'output' / 'osl_metrics'
OUT = TOOLS.parent / 'docs' / 'figures' / 'synthesis'

INDUSTRIES = ['metallurgy', 'oilgas', 'chemistry', 'energy']
RU = {'metallurgy': 'Металлургия', 'oilgas': 'Нефтегаз', 'chemistry': 'Химия', 'energy': 'Энергетика'}
LEARNED = ('elasticnet', 'ridge', 'hist_gbm')

# Conformal OOS-покрытие — ТРАНСКРИПЦИЯ из DS_REPORT_*.md (в walk-forward JSON этих чисел нет).
# calib-набор мал (n=2–6) → покрытие шумное; не выдавать за свежий расчёт. Источник — в 'src'.
CONFORMAL = {
    'metallurgy': {'model': 'structural_osl', 'coverage': '6/6', 'pct': 100, 'width': '±24.7%',
                   'n_calib': 5, 'note': 'малый calib → артефакт малого N',
                   'src': 'DS_REPORT.md'},
    'oilgas':     {'model': 'hist_gbm', 'coverage': '6/8', 'pct': 75, 'width': '±28%',
                   'n_calib': 4, 'note': 'структурная отложена; persistence 7/8=88%',
                   'src': 'DS_REPORT_OILGAS.md'},
    'chemistry':  {'model': 'elasticnet', 'coverage': '6/8', 'pct': 75, 'width': '±20%',
                   'n_calib': 4, 'note': 'структурный conformal ненадёжен (1/6=17%, объёмы разрежены)',
                   'src': 'DS_REPORT_CHEMISTRY.md'},
    'energy':     {'model': 'structural_osl', 'coverage': '12/12', 'pct': 100, 'width': '±34%',
                   'n_calib': 6, 'note': 'контемпоральные физданные покрывают тренд; stale 2/12=17%',
                   'src': 'DS_REPORT_ENERGY.md'},
}
# Энергетика — единственный случай, где все три числа на одном номинале и N=30: чистое сравнение
# структурной (контемпоральные физданные) против stale-моделей. Источник: DS_REPORT_ENERGY.md.
ENERGY_CONFORMAL = {'structural_osl': 100, 'elasticnet': 17, 'persistence': 17}


def load_metrics(metrics_dir: Path = METRICS_DIR) -> dict:
    """industry → распарсенный *_metrics.json (пропускает отсутствующие файлы)."""
    out = {}
    for ind in INDUSTRIES:
        p = metrics_dir / f'{ind}_metrics.json'
        if p.exists():
            out[ind] = json.loads(p.read_text(encoding='utf-8'))
    return out


def _mape(summary: dict, model: str):
    v = summary.get(model) or {}
    return v.get('mape_common')


def _best_learned(summary: dict):
    """(model, mape) лучшей learned-модели; (None, None) если ни одной."""
    cand = [(m, _mape(summary, m)) for m in LEARNED if _mape(summary, m) is not None]
    return min(cand, key=lambda t: t[1]) if cand else (None, None)


def _winner(summary: dict):
    """(model, mape) лучшей модели по MAPE_common среди всех (включая persistence)."""
    cand = [(m, v.get('mape_common')) for m, v in summary.items()
            if v.get('mape_common') is not None]
    return min(cand, key=lambda t: t[1]) if cand else (None, None)


def _panel_n(industry: str) -> int:
    """Размер панели (строк с таргетом) — headline-N, как в DS_REPORT_*.md."""
    return sum(1 for r in osl_panel.load_panel(industry) if r.has_target)


def assemble(metrics: dict) -> list:
    """Список строк-сводок по отраслям (для таблицы и графика)."""
    rows = []
    for ind in INDUSTRIES:
        if ind not in metrics:
            continue
        d = metrics[ind]
        s = d['summary']
        bl_model, bl_mape = _best_learned(s)
        win_model, win_mape = _winner(s)
        win_dm = (s.get(win_model) or {}).get('dm_p_vs_struct')
        rows.append({
            'industry': ind,
            'panel_n': _panel_n(ind),
            'n_common': d.get('n_common'),
            'base': d.get('base'),
            'structural': _mape(s, 'structural_osl'),
            'persistence': _mape(s, 'persistence'),
            'best_learned': bl_mape,
            'best_learned_model': bl_model,
            'winner': win_model,
            'winner_mape': win_mape,
            'winner_dm_p': win_dm,
        })
    return rows


# ============================================================
# Графики
# ============================================================

def plot_mape(rows: list, out: Path) -> Path:
    import matplotlib
    matplotlib.use('Agg')  # headless
    import matplotlib.pyplot as plt
    labels = [RU[r['industry']] for r in rows]
    x = np.arange(len(rows))
    w = 0.26
    series = [('structural', 'Структурная', 'tab:blue'),
              ('best_learned', 'Лучшая learned', 'tab:orange'),
              ('persistence', 'Persistence (наив)', 'tab:green')]
    fig, ax = plt.subplots(figsize=(9.5, 5))
    for i, (key, lbl, color) in enumerate(series):
        vals = [r[key] for r in rows]
        xs = x + (i - 1) * w
        bars = ax.bar(xs, [v if v is not None else 0 for v in vals], w, label=lbl, color=color)
        for xi, v, b in zip(xs, vals, bars):
            if v is None:
                ax.text(xi, 0.4, 'отложена', rotation=90, ha='center', va='bottom',
                        fontsize=7, color='gray')
            else:
                ax.text(xi, v + 0.2, f'{v:.1f}', ha='center', va='bottom', fontsize=7)
    # отметить победителя ромбиком
    for xi, r in zip(x, rows):
        if r['winner_mape'] is not None:
            ax.scatter([xi], [r['winner_mape']], marker='D', s=42, color='black', zorder=5)
    ax.set_xticks(x); ax.set_xticklabels([f'{lab}\n(N={r["panel_n"]})'
                                          for lab, r in zip(labels, rows)])
    ax.set_ylabel('MAPE_common, % (меньше — лучше)')
    ax.set_title('Walk-forward MAPE по отраслям × модели  (◆ = победитель отрасли)')
    ax.legend(fontsize=8, loc='upper left')
    ax.yaxis.grid(True, color=(0.5, 0.5, 0.5, 0.25), lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    p = out / 'mape_by_industry.png'
    fig.savefig(p, dpi=120); plt.close(fig)
    return p


def plot_conformal(out: Path) -> Path:
    import matplotlib
    matplotlib.use('Agg')  # headless
    import matplotlib.pyplot as plt
    items = [('structural_osl', 'Структурная\n(контемпоральные физданные)', 'tab:blue'),
             ('elasticnet', 'ElasticNet (stale)', 'tab:orange'),
             ('persistence', 'Persistence (stale)', 'tab:green')]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    xs = np.arange(len(items))
    vals = [ENERGY_CONFORMAL[k] for k, _l, _c in items]
    colors = [c for _k, _l, c in items]
    bars = ax.bar(xs, vals, color=colors, width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f'{v}%', ha='center', fontsize=11,
                fontweight='bold')
    ax.set_xticks(xs); ax.set_xticklabels([lbl for _k, lbl, _c in items], fontsize=8)
    ax.set_ylim(0, 112)
    ax.set_ylabel('OOS-покрытие conformal-интервала, %')
    ax.set_title('Энергетика (N=30): conformal-покрытие — ценность опер-сигнала', fontsize=11)
    ax.axhline(100, color='gray', lw=0.6, ls='--')
    ax.annotate('контемпоральные физданные года t\nотслеживают растущий тренд',
                xy=(0, 100), xytext=(0.3, 60), fontsize=8,
                arrowprops=dict(arrowstyle='->', color='gray'))
    ax.annotate('stale-модели отстают\nна растущем тренде',
                xy=(1.5, 17), xytext=(1.1, 38), fontsize=8,
                arrowprops=dict(arrowstyle='->', color='gray'))
    fig.tight_layout()
    p = out / 'conformal_coverage.png'
    fig.savefig(p, dpi=120); plt.close(fig)
    return p


# ============================================================
# Таблица
# ============================================================

def _fmt(v, suf='%'):
    return f'{v:.1f}{suf}' if isinstance(v, (int, float)) else '—'


def write_table(rows: list, out: Path) -> Path:
    lines = ['# Кросс-отраслевая сводка walk-forward (авто-сборка из *_metrics.json)', '',
             'N — размер панели (строк с таргетом); в скобках n — общий набор, где все модели дали '
             'прогноз (на нём считается MAPE_common; меньше — лучше). DM p — Diebold-Mariano '
             'победителя против БАЗЫ сравнения (структурная, где активна; persistence в нефтегазе — '
             'там структурная отложена). p>0.05 ⇒ различие не значимо.', '',
             '| Отрасль | N (n) | Структурная | Лучшая learned | Persistence | Победитель | MAPE | DM p (vs база) |',
             '|---|---|---|---|---|---|---|---|']
    win_ru = {'structural_osl': 'структурная', 'persistence': 'persistence',
              'elasticnet': 'elasticnet', 'ridge': 'ridge', 'hist_gbm': 'hist_gbm',
              'issuer_mean': 'issuer_mean'}
    for r in rows:
        dm = r['winner_dm_p']
        dm_s = '— (база)' if dm is None else f'{dm:.3f}' + ('' if dm < 0.05 else ' (н.з.)')
        bl = f'{_fmt(r["best_learned"])} ({r["best_learned_model"]})' if r['best_learned'] else '—'
        lines.append(
            f'| {RU[r["industry"]]} | {r["panel_n"]} ({r["n_common"]}) | {_fmt(r["structural"])} | '
            f'{bl} | {_fmt(r["persistence"])} | {win_ru.get(r["winner"], r["winner"])} | '
            f'{_fmt(r["winner_mape"])} | {dm_s} |')
    lines += ['', '**Conformal OOS-покрытие** (транскрипция из DS_REPORT_*.md; calib-набор мал n=2–6 '
              '→ шумно):', '',
              '| Отрасль | Модель | Покрытие | Ширина | Примечание | Источник |',
              '|---|---|---|---|---|---|']
    for ind in INDUSTRIES:
        c = CONFORMAL[ind]
        lines.append(f'| {RU[ind]} | {c["model"]} | {c["coverage"]} ({c["pct"]}%) | {c["width"]} | '
                     f'{c["note"]} | {c["src"]} |')
    lines.append('')
    p = out / 'summary_table.md'
    p.write_text('\n'.join(lines), encoding='utf-8')
    return p


def run(out_dir: Path = None, metrics_dir: Path = METRICS_DIR) -> dict:
    out = Path(out_dir) if out_dir is not None else OUT
    out.mkdir(parents=True, exist_ok=True)
    metrics = load_metrics(metrics_dir)
    if not metrics:
        print(f'Нет *_metrics.json в {metrics_dir}. Сначала: python osl_walkforward.py --industry <X>.')
        return {}
    rows = assemble(metrics)
    artifacts = {
        'mape': plot_mape(rows, out),
        'conformal': plot_conformal(out),
        'table': write_table(rows, out),
        'rows': rows,
    }
    return artifacts


def main():
    ap = argparse.ArgumentParser(description='Кросс-отраслевой DS-синтез (графики + таблица)')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    art = run(out_dir=args.out)
    if not art:
        return
    out = Path(args.out) if args.out is not None else OUT
    print('=' * 70)
    print(f'  DS-синтез: {len([k for k in art if k != "rows"])} артефакта → {out}')
    print('=' * 70)
    for r in art['rows']:
        print(f'  {RU[r["industry"]]:<12} N={r["panel_n"]:<3} победитель={r["winner"]:<14} '
              f'MAPE={r["winner_mape"]:.2f}%  (struct={_fmt(r["structural"])}, '
              f'persist={_fmt(r["persistence"])})')


if __name__ == '__main__':
    main()
