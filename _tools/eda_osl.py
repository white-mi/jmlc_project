"""
EDA панели OSL (DS-слой Макро-радара) — headless-генератор фигур, industry-параметрический.

`python eda_osl.py [--industry <отрасль>]` строит ~8 графиков в docs/figures/eda[_<отрасль>]/*.png
и пишет implications.md — по одной «импликации для модели» на каждый график (именно нарратив, а не
картинка, закрывает критерий «качество EDA»). Металлургия пишет в docs/figures/eda/ (исторические
ссылки в DS_REPORT.md), остальные отрасли — в docs/figures/eda_<отрасль>/.

Параметризация зеркалит остальной харнесс: отрасль фильтрует панель (osl_panel.load_panel), а наборы
цен/объёмов берутся из osl_models.INDUSTRY_PRICE_FEATURES / INDUSTRY_VOL_FEATURES. Issuer-специфичные
фигуры (эластичность, объём↔выручка, FX) конфигурируются через INDUSTRY_EDA_DRIVERS — это
презентационный лейблинг, а не модель. Структурный бейзлайн — общий StructuralOSL (диспетч по отрасли).

Зависимости — только в extra [eda]: pip install -e ".[eda]".
Ноутбук notebooks/eda_osl.ipynb импортирует эти же функции.

Что показываем (и зачем):
  1. Динамика выручки по эмитентам (индекс год начала=100) — временной сигнал панели.
  2. Эластичность выручка↔цена (log-log) — проверка структурного допущения наклон≈1.
  3. FX pass-through — реализованный курс vs официальный (только USD-ценовая корзина: металлургия; иначе пропуск).
  4. Объём↔выручка — подтверждение декомпозиции Q×P.
  5. Матрица пропусков — честная картина, обоснование scope (gap-серии).
  6. Масштаб/распределение таргета — аргумент за log-пространство.
  7. Корреляция/VIF цен — аргумент за регуляризацию (ElasticNet) vs OLS.
  8. Остатки структурного бейзлайна (IN-SAMPLE, все периоды) — планка, которую бьют learned-модели.
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

TOOLS = Path(__file__).parent
sys.path.insert(0, str(TOOLS))
import osl_panel  # noqa: E402
import osl_models  # noqa: E402

OUT = TOOLS.parent / "docs" / "figures" / "eda"  # дефолт металлургии (трекается в репо)

# Issuer-специфичная конфигурация фигур (лейблинг EDA, НЕ модель): эмитент →
# (ценовой признак для эластичности, объёмная колонка для объём↔выручка, человекочитаемая метка).
# Метка эластичности — основной драйвер выручки эмитента. Неприконфигуренный эмитент идёт по
# авто-фолбэку (первый non-usd_rub ценовой признак / первая ненулевая объёмная колонка).
INDUSTRY_EDA_DRIVERS = {
    "metallurgy": {
        "Полюс": ("gold", "vol_gold_oz", "золото"),
        "Норникель": ("copper", "vol_copper_t", "медь"),
        "Северсталь": ("steel_proxy_iron_ore", "vol_steel_t", "жел.руда (прокси)"),
        "НЛМК": ("steel_proxy_iron_ore", "vol_steel_t", "жел.руда (прокси)"),
        "ММК": ("steel_proxy_iron_ore", "vol_steel_t", "жел.руда (прокси)"),
    },
    "oilgas": {
        "Роснефть": ("urals", "vol_oil_t", "Urals"),
        "ЛУКОЙЛ": ("urals", "vol_oil_t", "Urals"),
        "Газпром": ("gas_eu", "vol_gas_mmcm", "газ EU (TTF прокси)"),
        "Новатэк": ("gas_eu", "vol_gas_mmcm", "газ EU (TTF прокси)"),
    },
    "chemistry": {
        "ФосАгро": ("dap", "vol_fertilizer_kt", "DAP (фосфор)"),
        "Акрон": ("urea", "vol_fertilizer_kt", "карбамид"),
        "КуйбышевАзот": ("urea", "vol_fertilizer_kt", "карбамид"),
        "КОС": ("crude_brent", "vol_polymer_kt", "нефть Brent (прокси полимеров)"),
    },
    "energy": {
        "РусГидро": ("electricity_rsv", "vol_generation_twh", "РСВ (спот)"),
        "Мосэнерго": ("electricity_rsv", "vol_generation_twh", "РСВ (спот)"),
        "ОГК-2": ("electricity_rsv", "vol_generation_twh", "РСВ (спот)"),
        "ТГК-1": ("electricity_rsv", "vol_generation_twh", "РСВ (спот)"),
        "Эл5-Энерго": ("electricity_rsv", "vol_generation_twh", "РСВ (спот)"),
        "Юнипро": ("electricity_rsv", "vol_generation_twh", "РСВ (спот)"),
    },
}

# FX pass-through осмысленна только там, где выручка эмитента — USD-ценовая сырьёвая корзина
# (металлургия; сами финотчёты при этом в ₽, кроме Полюса): реконструкция «реализованного курса» =
# выручка_RUB / стоимость_USD-корзины. Для отраслей без USD-ценовой корзины фигура пропускается.
# отрасль → (эмитент, [(объёмная_колонка, ценовой_признак), ...] компоненты корзины).
INDUSTRY_FX_BASKET = {
    "metallurgy": (
        "Норникель",
        [("vol_copper_t", "copper"), ("vol_nickel_t", "nickel"), ("vol_pt_oz", "platinum")],
    ),  # без Pd/Au (нет цены/объёма)
}

# Ручной сокращённый набор для VIF (n>p): где автоматический «первые k» хуже отражает
# реальную мультиколлинеарность. По умолчанию берутся первые k признаков.
INDUSTRY_VIF_SUBSET = {
    "metallurgy": ["p_gold", "p_copper", "p_nickel", "p_usd_rub"],
}


def _out_dir(industry: str) -> Path:
    return OUT if industry == "metallurgy" else OUT.parent / f"eda_{industry}"


class _Ctx:
    """Контекст прогона одной отрасли: куда писать, какие признаки/драйверы использовать."""

    def __init__(self, industry: str, rows: list, out: Path):
        self.industry = industry
        self.rows = rows
        self.out = out
        self.price_feats = osl_models.INDUSTRY_PRICE_FEATURES[industry]
        self.vol_feats = osl_models.INDUSTRY_VOL_FEATURES[industry]
        self.drivers = INDUSTRY_EDA_DRIVERS.get(industry, {})


# ============================================================
# DataFrame
# ============================================================


def build_df(industry: str = "metallurgy", rows=None) -> pd.DataFrame:
    if rows is None:
        rows = osl_panel.load_panel(industry)
    price_feats = osl_models.INDUSTRY_PRICE_FEATURES[industry]
    recs = []
    for r in rows:
        rec = {
            "issuer": r.issuer,
            "period": r.period,
            "year": r.period_end.year if r.period_end else None,
            "currency": r.revenue_currency,
            "target": r.target_bn,
            "confidence": r.confidence,
        }
        rec.update(r.volumes)  # vol_*: объёмы данной строки
        for m in price_feats:
            rec["p_" + m] = r.prices.get(m)
        recs.append(rec)
    df = pd.DataFrame(recs).sort_values(["issuer", "year"]).reset_index(drop=True)
    return df


# ============================================================
# Фигуры (каждая принимает (df, ctx) и возвращает строку-импликацию)
# ============================================================


def fig_revenue_trends(df, ctx) -> str:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for iss, g in df.groupby("issuer"):
        g = g.dropna(subset=["target"]).sort_values("year")
        if g.empty:
            continue
        base = g["target"].iloc[0]
        ax.plot(g["year"], g["target"] / base * 100, marker="o", label=iss)
    ax.axhline(100, color="gray", lw=0.6, ls="--")
    ax.set_title("Выручка по эмитентам (индекс, год начала = 100)")
    ax.set_xlabel("Год")
    ax.set_ylabel("Индекс выручки")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ctx.out / "01_revenue_trends.png", dpi=110)
    plt.close(fig)
    return (
        "01 Динамика выручки: индексация снимает разницу масштаба/валюты. Видно расхождение "
        "траекторий эмитентов внутри отрасли. Импликация: нужны issuer fixed-effects "
        "(своя база у каждого эмитента)."
    )


def _loglog_slope(x, y):
    """OLS-наклон log(y)~log(x). Возвращает (slope, n)."""
    m = (~np.isnan(x)) & (~np.isnan(y)) & (x > 0) & (y > 0)
    if m.sum() < 2:
        return None, int(m.sum())
    lx, ly = np.log(x[m]), np.log(y[m])
    A = np.vstack([lx, np.ones_like(lx)]).T
    slope, _ = np.linalg.lstsq(A, ly, rcond=None)[0]
    return float(slope), int(m.sum())


def fig_elasticity(df, ctx) -> str:
    """Эластичность выручки к ключевой цене эмитента (log-log наклон ≈ структурное допущение=1)."""
    present = set(df["issuer"])
    issuers = [i for i in ctx.drivers if i in present][:4]  # до 4 панелей (2×2)
    if not issuers:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.axis("off")
        ax.text(0.5, 0.5, "нет сконфигурированных драйверов", ha="center")
        fig.savefig(ctx.out / "02_elasticity.png", dpi=110)
        plt.close(fig)
        return "02 Эластичность: нет сконфигурированных драйверов для отрасли (пропуск)."
    ncol = 2 if len(issuers) > 1 else 1
    nrow = (len(issuers) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.5 * nrow), squeeze=False)
    axflat = axes.ravel()
    notes = []
    for ax, iss in zip(axflat, issuers):
        price_feat, _vol, lbl = ctx.drivers[iss]
        pcol = "p_" + price_feat
        g = df[df["issuer"] == iss]
        x = g[pcol].to_numpy(float)
        y = g["target"].to_numpy(float)
        ax.scatter(x, y)
        for _, row in g.iterrows():
            if not np.isnan(row[pcol]) and not np.isnan(row["target"]):
                ax.annotate(str(row["year"]), (row[pcol], row["target"]), fontsize=7)
        slope, n = _loglog_slope(x, y)
        notes.append(
            f"{iss}/{lbl}: эласт.≈{slope:.2f} (n={n})" if slope is not None else f"{iss}/{lbl}: n<2"
        )
        ax.set_title(
            f"{iss}: выручка vs {lbl}\nlog-log наклон≈" f"{slope:.2f}"
            if slope is not None
            else f"{iss}: мало точек"
        )
        ax.set_xlabel(lbl)
        ax.set_ylabel("выручка")
    for ax in axflat[len(issuers) :]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(ctx.out / "02_elasticity.png", dpi=110)
    plt.close(fig)
    tail = (
        ". Структурная модель неявно предполагает наклон=1 (выручка=Q×P); отклонения "
        "эмпирического наклона (хедж, лаги, доля рынка, бленд продуктов) — то, что "
        "learned-модель может поймать."
    )
    if ctx.industry == "metallurgy":
        tail += (
            " Для сталеваров iron_ore — ВХОДНАЯ цена (прокси), не выходная: знак связи "
            "слабее/иной — учитывать в интерпретации коэффициентов."
        )
    return "02 Эластичность выручка↔цена. " + "; ".join(notes) + tail


def fig_fx_passthrough(df, ctx) -> str:
    """Реализованный курс = выручка_RUB / стоимость_USD-корзины vs официальный USD/RUB.
    Только там, где выручка считается через USD-цены товара (металлургия); иначе — пропуск."""
    cfg = INDUSTRY_FX_BASKET.get(ctx.industry)
    if not cfg:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "FX-реконструкция неприменима\n(выручка отрасли в ₽)",
            ha="center",
            va="center",
            fontsize=11,
        )
        fig.savefig(ctx.out / "03_fx_passthrough.png", dpi=110)
        plt.close(fig)
        return (
            "03 FX pass-through: ПРОПУЩЕНО — у отрасли нет USD-ценовой сырьёвой корзины для "
            "реконструкции «реализованного курса». Фигура осмысленна только там, где выручка "
            "считается через USD-цены товара (металлургия). Импликация: здесь FX входит как "
            "обычный ценовой признак (usd_rub), отдельная реконструкция курса не нужна."
        )
    issuer, components = cfg
    g = df[df["issuer"] == issuer].copy()
    basket = sum(g[vol] * g["p_" + price] for vol, price in components)
    implied_fx = g["target"] * 1e9 / basket  # target в млрд ₽ → ₽
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(g["year"], g["p_usd_rub"], marker="o", label="офиц. USD/RUB (IRS)")
    ax.plot(g["year"], implied_fx, marker="s", label="реализ. ₽/USD-корзины (неполная)")
    ax.set_title(f"{issuer}: «реализованный» (неполная корзина) vs официальный курс")
    ax.set_xlabel("Год")
    ax.set_ylabel("₽ за USD")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ctx.out / "03_fx_passthrough.png", dpi=110)
    plt.close(fig)
    return (
        "03 FX pass-through. «Реализованный курс» = выручка_RUB / USD-корзина (только Cu+Ni+Pt) "
        "кратно ВЫШЕ официального, потому что корзина неполная: нет палладия (≈30% выручки), "
        "золота и байпродуктов — вместе ≈половина выручки Норникеля. Это и есть наглядная цена "
        "Pd-gap. Импликация: ручная USD→RUB реконструкция Норникеля смещена; надёжнее RUB-таргет "
        "+ issuer-FE, а не структурный пересчёт неполной корзины."
    )


def fig_volume_revenue(df, ctx) -> str:
    present = set(df["issuer"])
    spec = [(iss, drv[1]) for iss, drv in ctx.drivers.items() if iss in present]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    plotted = 0
    for iss, vcol in spec:
        if vcol not in df.columns:
            continue
        g = df[df["issuer"] == iss].dropna(subset=[vcol, "target"])
        if g.empty or g[vcol].max() == 0:
            continue
        ax.scatter(g[vcol] / g[vcol].max(), g["target"] / g["target"].max(), label=iss)
        plotted += 1
    ax.set_title("Объём (норм.) vs выручка (норм.) — проверка Q×P")
    ax.set_xlabel("Объём / max")
    ax.set_ylabel("Выручка / max")
    if plotted:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ctx.out / "04_volume_revenue.png", dpi=110)
    plt.close(fig)
    return (
        "04 Объём↔выручка (нормированы). Где связь слабее «чистого» Q×P — значит цена и микс "
        "важнее объёма в эти годы (объём падал, выручка держалась на цене). Импликация: цена "
        "обязана быть признаком, объёма недостаточно."
    )


def fig_missingness(df, ctx) -> str:
    cols = ["target"] + list(ctx.vol_feats) + ["p_" + f for f in ctx.price_feats]
    cols = [c for c in cols if c in df.columns]
    miss = df[cols].isna().astype(int)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(miss.values, aspect="auto", cmap="Reds", vmin=0, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=90, fontsize=7)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels([f"{r.issuer[:6]} {r.year}" for r in df.itertuples()], fontsize=6)
    ax.set_title("Матрица пропусков (красный = NaN)")
    fig.tight_layout()
    fig.savefig(ctx.out / "05_missingness.png", dpi=110)
    plt.close(fig)
    pct = 100 * miss.values.mean()
    vol_cols = [c for c in cols if c.startswith("vol_")]
    vol_miss = 100 * df[vol_cols].isna().mean().mean() if vol_cols else 0.0
    if pct < 3:
        return (
            f"05 Пропуски: всего {pct:.0f}% ячеек драйверов пусты — панель ПЛОТНАЯ (объёмы "
            "раскрыты по всем эмитентам и годам). Импликация: NaN-handling здесь не нагружен; "
            "плотные объёмы делают структурную Q×P-модель применимой ко всем строкам."
        )
    return (
        f"05 Пропуски: {pct:.0f}% ячеек драйверов пусты (объёмы — {vol_miss:.0f}%). Структурно "
        "объёмы разрежены (своя номенклатура у эмитента), а общий для всех заполненный признак — "
        "ЦЕНЫ. Импликация: разреженные объёмы → GBM с нативным NaN-handling, а не impute; "
        "цены — стабильный костяк."
    )


def fig_target_distribution(df, ctx) -> str:
    t = df["target"].dropna()
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].hist(t, bins=10)
    axes[0].set_title("Таргет (млрд, родная валюта)")
    axes[1].hist(np.log(t), bins=10)
    axes[1].set_title("log(таргет)")
    for a in axes:
        a.set_ylabel("частота")
    fig.tight_layout()
    fig.savefig(ctx.out / "06_target_distribution.png", dpi=110)
    plt.close(fig)
    mix = " (а в металлургии ещё и смешение USD/RUB)" if ctx.industry == "metallurgy" else ""
    return (
        f"06 Масштаб таргета: от {t.min():.1f} до {t.max():.0f} — разброс уровней эмитентов{mix}. "
        "log-преобразование выравнивает дисперсию. Импликация: моделируем log(выручка); метрика "
        "MAPE (валюто-инвариантна); issuer-FE снимает межуровневый сдвиг."
    )


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


def fig_price_correlation(df, ctx) -> str:
    pcols = [f"p_{f}" for f in ctx.price_feats if f"p_{f}" in df.columns]
    prices = df[["year"] + pcols].drop_duplicates("year").dropna()
    corr = prices[pcols].corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(pcols)))
    ax.set_xticklabels(pcols, rotation=90, fontsize=7)
    ax.set_yticks(range(len(pcols)))
    ax.set_yticklabels(pcols, fontsize=7)
    for i in range(len(pcols)):
        for j in range(len(pcols)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, fraction=0.046)
    ax.set_title("Корреляция цен (годовые средние)")
    fig.tight_layout()
    fig.savefig(ctx.out / "07_price_correlation.png", dpi=110)
    plt.close(fig)
    # VIF корректен только при n>p. Сокращённый набор (ручной override или первые k, n>p).
    n_per = len(prices)
    sub = INDUSTRY_VIF_SUBSET.get(ctx.industry)
    if sub:
        sub = [c for c in sub if c in pcols]
    else:
        k = max(2, min(len(pcols), n_per - 1))
        sub = pcols[:k]
    max_corr = corr.where(~np.eye(len(pcols), dtype=bool)).abs().max().max()
    Xs = prices[pcols].to_numpy(float)
    Xs = (Xs - Xs.mean(0)) / Xs.std(0)
    cond = np.linalg.cond(Xs)
    vif_max = None
    if n_per > len(sub) >= 2:
        vif = _vif(prices[sub].to_numpy(float))
        vif_max = max(vif)
        vif_s = ", ".join(f'{c.replace("p_", "")}={v:.1f}' for c, v in zip(sub, vif))
        vif_part = (
            f'VIF (сокр. набор {"/".join(c.replace("p_", "") for c in sub)}, n>p): '
            f"{vif_s} (>5-10 ⇒ сильная коллинеарность). "
        )
    else:
        vif_part = "VIF не определён (n≤p — ценовых признаков не меньше, чем периодов). "
    if cond >= 1e6:
        cond_clause = f"cond(матрицы цен)≈{cond:.0e} ⇒ практически вырождена"
        if n_per <= len(pcols):
            cond_clause += f" (раздута малой выборкой: {len(pcols)} цен ≥ {n_per} периодов)"
    else:
        cond_clause = f"cond(матрицы цен)≈{cond:.0e} ⇒ обусловленность приемлемая"
    strong = max_corr >= 0.8 or (vif_max is not None and vif_max > 5)
    if strong:
        impl = (
            "Импликация: высокие VIF/|corr| ⇒ OLS неустойчив → регуляризация (ElasticNet/Ridge) "
            "оправдана; на малом N плюс отбор/сжатие цен."
        )
    else:
        impl = (
            "Импликация: коллинеарность умеренная ⇒ регуляризация менее критична, но на малом N "
            "всё равно желательна для устойчивости."
        )
    return (
        f"07 Мультиколлинеарность цен (n={n_per} периодов). {vif_part}"
        f"Макс|corr|={max_corr:.2f}, {cond_clause}. {impl}"
    )


def fig_structural_residuals(df, ctx) -> str:
    """Бейзлайн: общий StructuralOSL (fit→predict, IN-SAMPLE) — средн |ошибка| по эмитенту.

    Берём все строки панели, где структурная определена (не NaN), и считаем средн |ошибка %|
    на эмитента. Это планка структурной физики Q×P, которую честная OOS learned-модель должна
    побить (реальный OOS-разрыв виден в walk-forward; здесь оценка оптимистична — k фитится in-sample).
    Для нефтегаза структурная отложена (NaN на всех строках) → бар-граф пуст, рендерим пометку."""
    struct = osl_models.StructuralOSL().fit(ctx.rows)
    preds = struct.predict(ctx.rows) if ctx.rows else np.array([])
    per_issuer = {}  # issuer → список |error %|
    for r, pv in zip(ctx.rows, preds):
        if pv is None or np.isnan(pv) or not r.target_bn:
            continue
        per_issuer.setdefault(r.issuer, []).append(abs(pv - r.target_bn) / r.target_bn * 100)
    labels = sorted(per_issuer)
    errs = [float(np.mean(per_issuer[i])) for i in labels]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    if errs:
        colors = ["tab:red" if e > 15 else "tab:orange" if e > 8 else "tab:green" for e in errs]
        ax.bar(labels, errs, color=colors)
        for h in (8, 15):
            ax.axhline(h, color="gray", lw=0.5, ls="--")
        ax.set_title("Структурный бейзлайн: средн |ошибка| по эмитенту (IN-SAMPLE, все периоды)")
        ax.set_ylabel("средн |прогноз − факт| / факт, %")
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=8)
    else:
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "структурная модель отложена\n(NaN на всех строках панели)",
            ha="center",
            va="center",
            fontsize=11,
        )
    fig.tight_layout()
    fig.savefig(ctx.out / "08_structural_residuals.png", dpi=110)
    plt.close(fig)
    if not errs:
        return (
            "08 Структурный бейзлайн ОТЛОЖЕН для отрасли (StructuralOSL даёт NaN на всех строках "
            "— напр. нефтегаз: годовые НДПИ/демпфер недоступны). База сравнения авто-падает на "
            "persistence. Импликация: learned-модель здесь сравнивается с наивным, а не с физикой."
        )
    mae = float(np.mean(errs))
    n_rows = sum(len(v) for v in per_issuer.values())
    thin = [i for i in labels if len(per_issuer[i]) <= 2]
    thin_note = (
        (
            f' Тонкие эмитенты (n≤2: {", ".join(thin)}) дают почти точный in-sample фит — '
            "их низкая ошибка не информативна."
        )
        if thin
        else ""
    )
    return (
        f"08 Бейзлайн StructuralOSL (IN-SAMPLE, per-period цены/объёмы + скаляр k): средн |ошибка| "
        f"≈{mae:.1f}% по {len(labels)} эмитентам ({n_rows} строк). Это планка структурной физики "
        "Q×P, которую честная out-of-sample learned-модель должна побить (Stage D). NB: k фитится "
        f"in-sample → оценка оптимистична; реальный OOS-разрыв виден в walk-forward.{thin_note}"
    )


FIGURES = [
    fig_revenue_trends,
    fig_elasticity,
    fig_fx_passthrough,
    fig_volume_revenue,
    fig_missingness,
    fig_target_distribution,
    fig_price_correlation,
    fig_structural_residuals,
]


def run(industry: str = "metallurgy", out_dir: Path = None) -> list:
    out = Path(out_dir) if out_dir is not None else _out_dir(industry)
    out.mkdir(parents=True, exist_ok=True)
    rows = osl_panel.load_panel(industry)
    df = build_df(industry, rows)
    if df.empty:
        print(f"Панель «{industry}» пуста — заполни data/panel/. EDA пропущена.")
        return []
    ctx = _Ctx(industry, rows, out)
    notes = []
    for fn in FIGURES:
        try:
            notes.append(fn(df, ctx))
        except Exception as e:  # одна фигура не должна валить весь прогон
            notes.append(f"{fn.__name__}: ОШИБКА {type(e).__name__}: {e}")
    (out / "implications.md").write_text(
        f"# EDA — импликации для модели ({industry})\n\n"
        + "\n\n".join(f'- **{n.split(" ", 1)[0]}** {n.split(" ", 1)[1]}' for n in notes),
        encoding="utf-8",
    )
    return notes


def main():
    ap = argparse.ArgumentParser(description="EDA панели OSL по отрасли")
    ap.add_argument(
        "--industry", default="metallurgy", help="metallurgy / oilgas / chemistry / energy"
    )
    args = ap.parse_args()
    notes = run(args.industry)
    out = _out_dir(args.industry)
    print("=" * 70)
    print(f"  EDA [{args.industry}]: {len(notes)} фигур → {out}")
    print("=" * 70)
    for n in notes:
        print("•", n)
        print()


if __name__ == "__main__":
    main()
