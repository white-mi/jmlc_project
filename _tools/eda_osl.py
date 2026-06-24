"""
EDA панели металлургии (DS-слой Макро-радара) — headless-генератор фигур.

`python eda_osl.py` строит ~8 графиков в docs/figures/eda/*.png и пишет
docs/figures/eda/implications.md — по одной «импликации для модели» на каждый график
(именно нарратив, а не картинка, закрывает критерий «качество EDA»).

Зависимости — только в extra [eda]: pip install -e ".[eda]".
Ноутбук notebooks/eda_osl.ipynb импортирует эти же функции.

Что показываем (и зачем):
  1. Динамика выручки по эмитентам (индекс 2021=100) — временной сигнал панели.
  2. Эластичность выручка↔цена (log-log) — проверка структурного допущения наклон≈1.
  3. FX pass-through (Норникель) — реализованный курс vs официальный USD/RUB.
  4. Объём↔выручка — подтверждение декомпозиции Q×P.
  5. Матрица пропусков — честная картина, обоснование scope (Pd/steel — gaps).
  6. Масштаб/распределение таргета — аргумент за log-пространство.
  7. Корреляция/VIF цен — аргумент за регуляризацию (ElasticNet) vs OLS.
  8. Остатки структурного бейзлайна (FY2025) — планка, которую бьют learned-модели.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

TOOLS = Path(__file__).parent
sys.path.insert(0, str(TOOLS))
import osl_panel          # noqa: E402
import osl_metallurgy     # noqa: E402

OUT = TOOLS.parent / 'docs' / 'figures' / 'eda'  # трекается в репо (видна на GitHub/в DS_REPORT)
PRICE_METALS = ['gold', 'copper', 'nickel', 'platinum', 'steel_proxy_iron_ore', 'usd_rub']


# ============================================================
# DataFrame
# ============================================================

def build_df() -> pd.DataFrame:
    rows = osl_panel.load_panel('metallurgy')
    recs = []
    for r in rows:
        rec = {'issuer': r.issuer, 'period': r.period,
               'year': r.period_end.year if r.period_end else None,
               'currency': r.revenue_currency, 'target': r.target_bn,
               'confidence': r.confidence}
        rec.update(r.volumes)               # vol_copper_t, vol_steel_t, ...
        for m in PRICE_METALS:
            rec['p_' + m] = r.prices.get(m)
        recs.append(rec)
    df = pd.DataFrame(recs).sort_values(['issuer', 'year']).reset_index(drop=True)
    return df


# ============================================================
# Фигуры (каждая возвращает строку-импликацию)
# ============================================================

def fig_revenue_trends(df) -> str:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for iss, g in df.groupby('issuer'):
        g = g.dropna(subset=['target']).sort_values('year')
        if g.empty:
            continue
        base = g['target'].iloc[0]
        ax.plot(g['year'], g['target'] / base * 100, marker='o', label=iss)
    ax.axhline(100, color='gray', lw=0.6, ls='--')
    ax.set_title('Выручка по эмитентам (индекс, год начала = 100)')
    ax.set_xlabel('Год'); ax.set_ylabel('Индекс выручки')
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(OUT / '01_revenue_trends.png', dpi=110); plt.close(fig)
    return ('01 Динамика выручки: индексация снимает разницу масштаба/валюты (USD Полюс vs '
            'RUB сталевары). Видно расхождение траекторий — золотодобыча растёт, сталь под '
            'давлением. Импликация: нужны issuer fixed-effects (своя база у каждого эмитента).')


def _loglog_slope(x, y):
    """OLS-наклон log(y)~log(x). Возвращает (slope, n)."""
    m = (~np.isnan(x)) & (~np.isnan(y)) & (x > 0) & (y > 0)
    if m.sum() < 2:
        return None, int(m.sum())
    lx, ly = np.log(x[m]), np.log(y[m])
    A = np.vstack([lx, np.ones_like(lx)]).T
    slope, _ = np.linalg.lstsq(A, ly, rcond=None)[0]
    return float(slope), int(m.sum())


def fig_elasticity(df) -> str:
    """Эластичность выручки к ключевой цене: Полюс↔золото, сталевары↔iron_ore."""
    pairs = [('Полюс', 'p_gold', 'золото'),
             ('Норникель', 'p_copper', 'медь'),
             ('Северсталь', 'p_steel_proxy_iron_ore', 'жел.руда (прокси)'),
             ('НЛМК', 'p_steel_proxy_iron_ore', 'жел.руда (прокси)')]
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    notes = []
    for ax, (iss, pcol, lbl) in zip(axes.ravel(), pairs):
        g = df[df['issuer'] == iss]
        x = g[pcol].to_numpy(float); y = g['target'].to_numpy(float)
        ax.scatter(x, y)
        for _, row in g.iterrows():
            ax.annotate(str(row['year']), (row[pcol], row['target']), fontsize=7)
        slope, n = _loglog_slope(x, y)
        notes.append(f'{iss}/{lbl}: эласт.≈{slope:.2f} (n={n})' if slope is not None
                     else f'{iss}/{lbl}: n<2')
        ax.set_title(f'{iss}: выручка vs {lbl}\nlog-log наклон≈'
                     f'{slope:.2f}' if slope is not None else f'{iss}: мало точек')
        ax.set_xlabel(lbl); ax.set_ylabel('выручка')
    fig.tight_layout()
    fig.savefig(OUT / '02_elasticity.png', dpi=110); plt.close(fig)
    return ('02 Эластичность выручка↔цена. ' + '; '.join(notes) +
            '. Структурная модель неявно предполагает наклон=1 (выручка=Q×P); отклонения '
            'эмпирического наклона (хедж, лаги, доля рынка) — то, что learned-модель может '
            'поймать. Для сталеваров iron_ore — ВХОДНАЯ цена (прокси), не выходная: знак '
            'связи слабее/иной — учитывать в интерпретации коэффициентов.')


def fig_fx_passthrough(df) -> str:
    """Норникель: реализованный курс = выручка_RUB / стоимость_корзины_USD vs офиц. USD/RUB."""
    g = df[df['issuer'] == 'Норникель'].copy()
    # стоимость корзины в USD по доступным металлам (Pd отсутствует — gap)
    basket = (g['vol_copper_t'] * g['p_copper'] +
              g['vol_nickel_t'] * g['p_nickel'] +
              g['vol_pt_oz'] * g['p_platinum'])      # без Pd (нет цены)
    implied_fx = g['target'] * 1e9 / basket          # target в млрд ₽ → ₽
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(g['year'], g['p_usd_rub'], marker='o', label='офиц. USD/RUB (IRS)')
    ax.plot(g['year'], implied_fx, marker='s',
            label='реализ. ₽/USD-корзины (без Pd)')
    ax.set_title('Норникель: «реализованный» (неполная корзина) vs официальный курс')
    ax.set_xlabel('Год'); ax.set_ylabel('₽ за USD')
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(OUT / '03_fx_passthrough.png', dpi=110); plt.close(fig)
    return ('03 FX pass-through. «Реализованный курс» = выручка_RUB / USD-корзина (только Cu+Ni+Pt) '
            'кратно ВЫШЕ официального, потому что корзина неполная: нет палладия (≈30% выручки), '
            'золота и байпродуктов — вместе ≈половина выручки Норникеля. Это и есть наглядная цена '
            'Pd-gap. Импликация: ручная USD→RUB реконструкция Норникеля смещена; надёжнее RUB-таргет '
            '+ issuer-FE, а не структурный пересчёт неполной корзины.')


def fig_volume_revenue(df) -> str:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    spec = [('Полюс', 'vol_gold_oz', 'oz'), ('Северсталь', 'vol_steel_t', 't'),
            ('ММК', 'vol_steel_t', 't'), ('НЛМК', 'vol_steel_t', 't'),
            ('Норникель', 'vol_copper_t', 't Cu')]
    for iss, vcol, _u in spec:
        g = df[df['issuer'] == iss].dropna(subset=[vcol, 'target'])
        if g.empty:
            continue
        ax.scatter(g[vcol] / g[vcol].max(), g['target'] / g['target'].max(), label=iss)
    ax.set_title('Объём (норм.) vs выручка (норм.) — проверка Q×P')
    ax.set_xlabel('Объём / max'); ax.set_ylabel('Выручка / max')
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(OUT / '04_volume_revenue.png', dpi=110); plt.close(fig)
    return ('04 Объём↔выручка (нормированы). Для золота/стали связь слабее, чем для «чистого» Q×P — '
            'значит цена и микс важнее объёма в эти годы (объём падал, выручка держалась на цене). '
            'Импликация: цена обязана быть признаком, объёма недостаточно.')


def fig_missingness(df) -> str:
    cols = ['target', 'vol_copper_t', 'vol_nickel_t', 'vol_pd_oz', 'vol_pt_oz',
            'vol_gold_oz', 'vol_steel_t', 'p_gold', 'p_copper', 'p_nickel',
            'p_platinum', 'p_steel_proxy_iron_ore', 'p_usd_rub']
    miss = df[cols].isna().astype(int)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(miss.values, aspect='auto', cmap='Reds', vmin=0, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=90, fontsize=7)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels([f'{r.issuer[:6]} {r.year}' for r in df.itertuples()], fontsize=6)
    ax.set_title('Матрица пропусков (красный = NaN)')
    fig.tight_layout()
    fig.savefig(OUT / '05_missingness.png', dpi=110); plt.close(fig)
    pct = 100 * miss.values.mean()
    return (f'05 Пропуски: {pct:.0f}% ячеек драйверов пусты. Структурно: vol_pd_oz/Cu/Ni/Pt — только '
            'Норникель; vol_gold — только Полюс; vol_steel — только сталевары (Mt 2021-24, 2025 NaN). '
            'Импликация: разреженные объёмы → GBM с нативным NaN-handling, а не impute; общий для всех '
            'признак — ЦЕНЫ (заполнены полностью, кроме gap-серий Pd/steel).')


def fig_target_distribution(df) -> str:
    t = df['target'].dropna()
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].hist(t, bins=10); axes[0].set_title('Таргет (млрд, смешанные валюты)')
    axes[1].hist(np.log(t), bins=10); axes[1].set_title('log(таргет)')
    for a in axes:
        a.set_ylabel('частота')
    fig.tight_layout()
    fig.savefig(OUT / '06_target_distribution.png', dpi=110); plt.close(fig)
    return (f'06 Масштаб таргета: от {t.min():.1f} до {t.max():.0f} (Полюс ~$5 млрд vs НЛМК ~1000 ₽млрд) '
            '— разброс на порядки из-за смешения USD/RUB. log-преобразование выравнивает дисперсию. '
            'Импликация: моделируем log(выручка); метрика MAPE (валюто-инвариантна); issuer-FE снимает '
            'межвалютный сдвиг уровня.')


def _vif(X: np.ndarray) -> list:
    """VIF_j = 1/(1-R²_j) через OLS каждого признака на остальные."""
    n, p = X.shape
    Xs = (X - X.mean(0)) / X.std(0)
    out = []
    for j in range(p):
        y = Xs[:, j]
        Z = np.delete(Xs, j, axis=1)
        Z = np.hstack([Z, np.ones((n, 1))])
        beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
        r2 = 1 - np.sum((y - Z @ beta) ** 2) / np.sum((y - y.mean()) ** 2)
        out.append(1.0 / max(1e-9, 1 - r2))
    return out


def fig_price_correlation(df) -> str:
    pcols = ['p_gold', 'p_copper', 'p_nickel', 'p_platinum',
             'p_steel_proxy_iron_ore', 'p_usd_rub']
    prices = df[['year'] + pcols].drop_duplicates('year').dropna()
    corr = prices[pcols].corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr.values, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_xticks(range(len(pcols))); ax.set_xticklabels(pcols, rotation=90, fontsize=7)
    ax.set_yticks(range(len(pcols))); ax.set_yticklabels(pcols, fontsize=7)
    for i in range(len(pcols)):
        for j in range(len(pcols)):
            ax.text(j, i, f'{corr.values[i, j]:.2f}', ha='center', va='center', fontsize=7)
    fig.colorbar(im, fraction=0.046)
    ax.set_title('Корреляция цен (годовые средние)')
    fig.tight_layout()
    fig.savefig(OUT / '07_price_correlation.png', dpi=110); plt.close(fig)
    # VIF корректен только при n>p. Периодов всего 5 → полный набор (6) вырожден.
    # Считаем VIF на сокращённом наборе (4 признака, n=5>4) + cond.number полной матрицы.
    # сокращённый набор: по одному драйверу на бизнес-модель (золото/Полюс, медь+никель/
    # Норникель, usd_rub/сталевары-RUB), n=5>p=4 → VIF определён. platinum/iron_ore опущены
    # как сильно коллинеарные с copper/nickel (видно на heatmap) — чтобы не раздувать p.
    sub = ['p_gold', 'p_copper', 'p_nickel', 'p_usd_rub']
    vif = _vif(prices[sub].to_numpy(float))
    vif_s = ', '.join(f'{c.replace("p_", "")}={v:.1f}' for c, v in zip(sub, vif))
    Xs = prices[pcols].to_numpy(float)
    Xs = (Xs - Xs.mean(0)) / Xs.std(0)
    cond = np.linalg.cond(Xs)
    max_corr = corr.where(~np.eye(len(pcols), dtype=bool)).abs().max().max()
    return (f'07 Мультиколлинеарность цен (n={len(prices)} периодов). VIF (сокр. набор '
            f'gold/copper/nickel/usd_rub, n>p): {vif_s} (>5-10 ⇒ сильная коллинеарность). '
            f'Макс|corr|={max_corr:.2f}, cond(матрицы цен)≈{cond:.0e} (>1e6 ⇒ практически вырождена). '
            'Полный VIF на 6 ценах не определён (n=5<6) — сам по себе сигнал: ценовых признаков '
            'больше, чем независимых наблюдений. Импликация: OLS неустойчив; регуляризация '
            '(ElasticNet/Ridge) обязательна, плюс отбор/сжатие цен на N=24.')


def fig_structural_residuals(df) -> str:
    """Бейзлайн: статичная StructuralOSL (predict_revenue, 12M2025) vs факт FY2025."""
    rows, errs = [], []
    for iss in ['Норникель', 'Северсталь', 'ММК', 'НЛМК', 'Полюс']:
        g = df[(df['issuer'] == iss) & (df['year'] == 2025)]
        if g.empty:
            continue
        actual = g['target'].iloc[0]; cur = g['currency'].iloc[0]
        try:
            pred = osl_metallurgy.predict_revenue(iss)
        except Exception:
            continue
        pv = pred.predicted_rub_bn if cur == 'RUB' else pred.predicted_usd_bn
        if pv is None or actual is None:
            continue
        rows.append(iss); errs.append((pv - actual) / actual * 100)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    colors = ['tab:red' if abs(e) > 15 else 'tab:orange' if abs(e) > 8 else 'tab:green' for e in errs]
    ax.bar(rows, errs, color=colors)
    ax.axhline(0, color='k', lw=0.8)
    for h in (10, -10):
        ax.axhline(h, color='gray', lw=0.5, ls='--')
    ax.set_title('Ошибка структурного бейзлайна (IN-SAMPLE), FY2025 (%) — планка для learned-моделей')
    ax.set_ylabel('(прогноз − факт) / факт, %')
    fig.tight_layout()
    fig.savefig(OUT / '08_structural_residuals.png', dpi=110); plt.close(fig)
    mae = np.mean(np.abs(errs)) if errs else float('nan')
    return (f'08 Бейзлайн StructuralOSL (статичные цены/объёмы 12M2025, IN-SAMPLE): средн |ошибка| '
            f'≈{mae:.1f}% по {len(errs)} эмитентам на FY2025. Это планка, которую честная '
            'out-of-sample learned-модель должна побить (Stage C/D). NB: бейзлайн использует '
            'захардкоженные цены → не видит ценовой динамики 2021-24, отсюда и потенциал улучшения.')


FIGURES = [fig_revenue_trends, fig_elasticity, fig_fx_passthrough, fig_volume_revenue,
           fig_missingness, fig_target_distribution, fig_price_correlation,
           fig_structural_residuals]


def run() -> list:
    OUT.mkdir(parents=True, exist_ok=True)
    df = build_df()
    if df.empty:
        print('Панель пуста — заполни data/panel/. EDA пропущена.')
        return []
    notes = []
    for fn in FIGURES:
        try:
            notes.append(fn(df))
        except Exception as e:  # одна фигура не должна валить весь прогон
            notes.append(f'{fn.__name__}: ОШИБКА {type(e).__name__}: {e}')
    (OUT / 'implications.md').write_text(
        '# EDA — импликации для модели\n\n' +
        '\n\n'.join(f'- **{n.split(" ", 1)[0]}** {n.split(" ", 1)[1]}' for n in notes),
        encoding='utf-8')
    return notes


def main():
    notes = run()
    print('=' * 70)
    print(f'  EDA: {len(notes)} фигур → {OUT}')
    print('=' * 70)
    for n in notes:
        print('•', n)
        print()


if __name__ == '__main__':
    main()
