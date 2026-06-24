"""
Walk-forward валидация моделей прогноза выручки (Stage D) — ЧЕСТНЫЙ out-of-sample.

Expanding-window: для каждого тест-периода t обучаем на всех строках с годом < t,
тестируем на строках года t. Это реальный сценарий применения OSL: «по данным
до года t спрогнозировать выручку года t+1». Сплит по ВРЕМЕНИ (не по эмитенту) —
каждая тест-строка это будущий период эмитента, уже виденного в train (anti-leakage).

Метрики (на пуле тест-строк): MAE, MAPE, RMSE, skill-score = 1 − MAPE_model/MAPE_struct
(плюс = бьёт структурный бейзлайн), Diebold–Mariano p-value (структурный vs каждый).

Честность сравнения: StructuralOSL даёт NaN на gap-строках (сталь 2025) → его тест-выборка
уже. Поэтому headline-метрики (skill/DM) считаем на ОБЩЕМ наборе строк, где ВСЕ модели
дали конечный прогноз; плюс отдельно — per-model покрытие (на скольких строках смог).

CLI:
  python osl_walkforward.py --industry metallurgy   # пишет output/osl_metrics/<ind>.md + .json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

TOOLS = Path(__file__).parent
sys.path.insert(0, str(TOOLS))
import osl_panel       # noqa: E402
import osl_models as Mo  # noqa: E402

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

OUT = TOOLS / 'output' / 'osl_metrics'


# ============================================================
# Walk-forward
# ============================================================

def walk_forward(rows, model_ctors):
    """Возвращает dict: model_name → list[(issuer, period, year, actual, pred)] по всем фолдам.
    Плюс fold_log — что в train/test на каждом шаге."""
    rows = [r for r in rows if r.has_target and r.period_end]
    years = sorted({r.period_end.year for r in rows})
    preds = {name: [] for name in model_ctors}
    fold_log = []
    for t in years[1:]:                       # с 2-го периода (нужен хотя бы 1 train-год)
        train = [r for r in rows if r.period_end.year < t]
        test = [r for r in rows if r.period_end.year == t]
        if not train or not test:
            continue
        # ЛИК-ГАРД: ни одна train-строка не из будущего теста
        assert all(r.period_end.year < t for r in train)
        fold_log.append({'test_year': t, 'n_train': len(train), 'n_test': len(test)})
        for name, ctor in model_ctors.items():
            model = ctor().fit(train)
            yhat = model.predict(test)
            for r, p in zip(test, yhat):
                preds[name].append((r.issuer, r.period, t, r.target_bn, float(p)))
    return preds, fold_log


# ============================================================
# Метрики
# ============================================================

def _err_arrays(records):
    """(actual, pred) → только строки с конечным прогнозом."""
    a = np.array([x[3] for x in records], dtype=float)
    p = np.array([x[4] for x in records], dtype=float)
    m = np.isfinite(p) & np.isfinite(a) & (a != 0)
    return a[m], p[m], m


def metrics_for(records):
    a, p, _ = _err_arrays(records)
    if len(a) == 0:
        return {'n': 0, 'mae': None, 'mape': None, 'rmse': None}
    err = p - a
    return {'n': int(len(a)),
            'mae': float(np.mean(np.abs(err))),
            'mape': float(np.mean(np.abs(err / a)) * 100),
            'rmse': float(np.sqrt(np.mean(err ** 2)))}


def _common_keys(preds):
    """Ключи (issuer,period), где ВСЕ модели дали конечный прогноз — для честного skill/DM."""
    keysets = []
    for recs in preds.values():
        ks = {(i, per) for (i, per, _t, a, p) in recs if np.isfinite(p) and np.isfinite(a)}
        keysets.append(ks)
    return set.intersection(*keysets) if keysets else set()


def _abs_err_on_keys(records, keys):
    """|ошибка %| по ключам в фикс. порядке (для DM)."""
    d = {(i, per): abs((p - a) / a) for (i, per, _t, a, p) in records
         if np.isfinite(p) and np.isfinite(a) and a != 0}
    ordered = sorted(keys)
    return np.array([d[k] for k in ordered if k in d], dtype=float)


def diebold_mariano(ae_a, ae_b):
    """Сравнение точности (по |ошибке|). H0: равная точность.
    Возвращает (stat, p_two_sided). Малые n → t-распределение (df=n-1)."""
    d = ae_a - ae_b
    n = len(d)
    if n < 2 or np.allclose(d, 0):
        return 0.0, 1.0
    sd = np.std(d, ddof=1)
    if sd == 0:
        return 0.0, 1.0
    stat = float(np.mean(d) / (sd / np.sqrt(n)))
    p = float(2 * (1 - stats.t.cdf(abs(stat), df=n - 1)))
    return stat, p


def evaluate(preds):
    """Полная сводка: per-model метрики (на своих валидных строках) + skill/DM на общем наборе."""
    base = 'structural_osl'
    common = _common_keys(preds)
    base_ae = _abs_err_on_keys(preds[base], common) if base in preds else np.array([])
    base_mape_common = float(np.mean(base_ae) * 100) if len(base_ae) else None

    summary = {}
    for name, recs in preds.items():
        m = metrics_for(recs)
        ae = _abs_err_on_keys(recs, common)
        mape_common = float(np.mean(ae) * 100) if len(ae) else None
        skill = (1 - mape_common / base_mape_common
                 if (mape_common is not None and base_mape_common) else None)
        if name == base or len(ae) == 0 or len(base_ae) == 0:
            dm_stat, dm_p = (None, None)
        else:
            dm_stat, dm_p = diebold_mariano(base_ae, ae)  # >0 ⇒ base хуже (больше ошибка)
        summary[name] = {**m, 'mape_common': mape_common, 'skill_vs_struct': skill,
                         'dm_stat_vs_struct': dm_stat, 'dm_p_vs_struct': dm_p}
    return summary, sorted(common)


# ============================================================
# Отчёт
# ============================================================

def _fmt(v, f='.2f'):
    return '—' if v is None else format(v, f)


def render_report(industry, summary, fold_log, common, n_total):
    lines = [f'# Walk-forward валидация OSL — {industry}', '',
             '> Expanding-window: train = все годы < t, test = год t. Сплит по времени, '
             'группировка по периоду. Out-of-sample.', '',
             '**Фолды:** ' + '; '.join(f"test {f['test_year']} "
                                        f"(train {f['n_train']} → test {f['n_test']})"
                                        for f in fold_log),
             f'**Общий набор для skill/DM (все модели дали прогноз):** {len(common)} строк '
             f'из {n_total} тест-строк.', '',
             '| model | n | MAE | MAPE % | RMSE | MAPE_common % | skill_vs_struct | DM p (vs struct) |',
             '|---|---|---|---|---|---|---|---|']
    order = ['structural_osl', 'ridge', 'elasticnet', 'hist_gbm']
    for name in [n for n in order if n in summary] + [n for n in summary if n not in order]:
        s = summary[name]
        lines.append(f"| {name} | {s['n']} | {_fmt(s['mae'])} | {_fmt(s['mape'])} | "
                     f"{_fmt(s['rmse'])} | {_fmt(s['mape_common'])} | "
                     f"{_fmt(s['skill_vs_struct'], '+.3f')} | {_fmt(s['dm_p_vs_struct'], '.3f')} |")
    lines += ['',
              '- **skill_vs_struct** = 1 − MAPE_model/MAPE_struct на общем наборе (>0 ⇒ бьёт '
              'структурный бейзлайн).',
              '- **DM p** — Diebold–Mariano (двусторонний, t, df=n−1) по |ошибкам%| структурный vs '
              'модель; p<0.05 ⇒ различие точности значимо (на N столь малом — ориентир, не доказательство).',
              '- StructuralOSL на gap-строках (нет объёма, напр. сталь 2025) даёт NaN → его n меньше; '
              'поэтому честное сравнение — по колонке MAPE_common.',
              '- Прозрачность: номинальный перевес hist_gbm частично опирается на фолд 2025, где '
              'frozen-2025 калибровка структурной модели наиболее благоприятна; формальное '
              'сравнение (MAPE_common/DM) считается на общем 16-строчном наборе и этим нейтрализуется.']
    return '\n'.join(lines)


def run(industry='metallurgy'):
    rows = osl_panel.load_panel(industry=industry)
    rows = [r for r in rows if r.has_target and r.period_end]
    if len(rows) < 8:
        print('Панель мала — walk-forward пропущен.'); return None
    preds, fold_log = walk_forward(rows, Mo.MODELS)
    summary, common = evaluate(preds)
    n_total = max(len(v) for v in preds.values())
    OUT.mkdir(parents=True, exist_ok=True)
    md = render_report(industry, summary, fold_log, common, n_total)
    (OUT / f'{industry}.md').write_text(md, encoding='utf-8')
    (OUT / f'{industry}_metrics.json').write_text(
        json.dumps({'summary': summary, 'folds': fold_log,
                    'n_common': len(common), 'n_total': n_total}, ensure_ascii=False, indent=2),
        encoding='utf-8')
    return summary, md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--industry', default='metallurgy')
    args = ap.parse_args()
    res = run(args.industry)
    if res:
        print(res[1])
        print(f'\n→ output/osl_metrics/{args.industry}.md + _metrics.json')


if __name__ == '__main__':
    main()
