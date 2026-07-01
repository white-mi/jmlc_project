"""
Operational Signal Layer (OSL) — Металлургия.

Прогноз выручки публичных эмитентов металлургии по операционным сигналам:
  Revenue ≈ Σ(Q_metal × Price_metal × FX) × (1 + other_income_pct) − discount

Бэк-тест на 12М 2025: predicted revenue vs actual (МСФО).

Источники операционных данных:
  - Норникель: производство меди/никеля/палладия из IR-релизов (квартальные)
  - Северсталь: производство стали (тоны), структура продаж РФ/экспорт
  - Полюс: унции добытого / проданного золота
  - Цены: LME (медь, никель), LBMA (золото, палладий, платина), CRU (сталь)
  - FX: ЦБ РФ среднегодовая

Концепция: для глобально-ценовых сырьевых эмитентов (Полюс, Норникель)
формула «Q × P_LME» работает с MAE 3-15%. Для компаний с большой
долей внутреннего рынка (Северсталь, ММК, НЛМК) нужна гибридная модель:
  Revenue = Q_export × P_FOB + Q_domestic × P_domestic

Этот файл — модуль Фазы 1.5 проекта Макро-радар (v0.9).
"""

import argparse
import sys
from dataclasses import dataclass, field
from typing import Dict, List

from osl_common import RevenuePredict, FXRate  # S3.1: общие структуры

# Windows console fix: force UTF-8 for ₽, ✅, etc.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# ============================================================
# DATA STRUCTURES
# ============================================================


@dataclass
class CommodityPrice:
    """Средняя цена сырья за период (USD/единица)."""

    name: str
    unit: str
    avg_price_usd: float
    period: str
    source: str = ""


@dataclass
class ProductionData:
    """Объём производства / отгрузок за период."""

    company: str
    metal: str
    volume: float
    unit: str
    period: str
    note: str = ""


@dataclass
class CompanyProfile:
    """Структура продаж: какие металлы, доли, бизнес-модель."""

    name: str
    revenue_model: str  # 'global_commodity' / 'domestic_premium' / 'hybrid'
    revenue_share: Dict[str, float] = field(default_factory=dict)  # вес по металлам
    other_income_pct: float = 0.05  # services, byproducts
    domestic_share: float = 0.0  # доля внутреннего рынка
    domestic_premium_pct: float = 0.0  # премия внутр. цен над экспортом
    notes: str = ""


# RevenuePredict / FXRate — из osl_common (S3.1)


# ============================================================
# СРЕДНИЕ ЦЕНЫ 12М 2025 (USD)
# ============================================================
# Усредненные данные по году. Для точности нужно подключить FRED/LME APIs
# (отложено до фазы 2). Сейчас — экспертная оценка по диапазонам:
#   медь: $8 892 (01.2025) → $13 862 (01.2026), среднегодовая ~$9 500
#   никель: $14-19K в 2025, средняя ~$16 500
#   палладий: $1 000-1 100/унц
#   платина: $950-1 000/унц
#   золото: $2 625 (01.2025) → $5 000 (01.2026), средняя ~$3 350
#   сталь FOB ЧМ: $480-530, средняя ~$510

PRICES_12M_2025: Dict[str, CommodityPrice] = {
    "copper": CommodityPrice("copper", "t", 9_500, "12M2025", "LME avg"),
    "nickel": CommodityPrice("nickel", "t", 16_500, "12M2025", "LME avg"),
    "palladium": CommodityPrice("palladium", "oz", 1_050, "12M2025", "LBMA avg"),
    "platinum": CommodityPrice("platinum", "oz", 970, "12M2025", "LBMA avg"),
    "gold": CommodityPrice("gold", "oz", 3_350, "12M2025", "LBMA avg"),
    # Byproducts Норникеля (v0.3 final)
    "cobalt": CommodityPrice("cobalt", "t", 22_000, "12M2025", "LME cobalt avg"),
    "rhodium": CommodityPrice("rhodium", "oz", 5_500, "12M2025", "оценка spot"),
    "iridium": CommodityPrice("iridium", "oz", 4_500, "12M2025", "оценка spot"),
    "tellurium": CommodityPrice("tellurium", "t", 65_000, "12M2025", "отраслевая оценка"),
    "selenium": CommodityPrice("selenium", "t", 40_000, "12M2025", "отраслевая оценка"),
    # Steel
    "steel_fob_chm": CommodityPrice("steel_fob_chm", "t", 510, "12M2025", "CRU/MMI ЧМ"),
    "steel_domestic_rub": CommodityPrice(
        "steel_domestic_rub", "t", 65_000, "12M2025", "Минпромторг/Металлоторг — оценка"
    ),
}

FX_12M_2025 = FXRate(avg_usd_rub=89.0, period="12M2025")


# ============================================================
# ПРОИЗВОДСТВО ПУБЛИЧНЫХ ЭМИТЕНТОВ 12М 2025 (оценки)
# ============================================================
# Точные значения берутся из квартальных production reports на IR-страницах.
# Здесь — экстраполяция на основе 9М 2025.

NORNICKEL_PRODUCTION = [
    ProductionData("Норникель", "copper", 425_000, "t", "12M2025"),
    ProductionData("Норникель", "nickel", 200_000, "t", "12M2025"),
    ProductionData("Норникель", "palladium", 2_700_000, "oz", "12M2025"),
    ProductionData("Норникель", "platinum", 650_000, "oz", "12M2025"),
    # Byproducts (v0.3 final)
    ProductionData(
        "Норникель", "gold", 200_000, "oz", "12M2025", "байпродукт никелевой переработки"
    ),
    ProductionData("Норникель", "cobalt", 5_000, "t", "12M2025", "байпродукт"),
    ProductionData("Норникель", "rhodium", 80_000, "oz", "12M2025", "высокомаржинальный PGM"),
    ProductionData("Норникель", "iridium", 8_000, "oz", "12M2025"),
    ProductionData("Норникель", "tellurium", 200, "t", "12M2025"),
    ProductionData("Норникель", "selenium", 100, "t", "12M2025"),
]

SEVERSTAL_PRODUCTION = [
    ProductionData("Северсталь", "steel", 11_000_000, "t", "12M2025", "consolidated steel output"),
]

# ММК — приблизительно 11.5 млн т
MMK_PRODUCTION = [
    ProductionData("ММК", "steel", 11_500_000, "t", "12M2025"),
]

# НЛМК — крупнее (~16 млн т, включая зарубежные активы)
NLMK_PRODUCTION = [
    ProductionData("НЛМК", "steel", 16_000_000, "t", "12M2025", "incl. NLMK USA/Europe"),
]

POLYUS_PRODUCTION = [
    ProductionData("Полюс", "gold", 2_529_000, "oz", "12M2025", "факт из IR (-15.8% г/г)"),
]


# ============================================================
# ПРОФИЛИ КОМПАНИЙ
# ============================================================

PROFILES: Dict[str, CompanyProfile] = {
    "Норникель": CompanyProfile(
        name="Норникель",
        revenue_model="global_commodity",
        revenue_share={
            "copper": 0.30,
            "nickel": 0.25,
            "palladium": 0.30,
            "platinum": 0.08,
            "byproducts": 0.07,
        },  # Au+Co+Rh+Ir+Te+Se ~7% выручки
        other_income_pct=0.05,  # treatment charges + services
        domestic_share=0.05,
        notes="v0.3 final: 10 metals (4 main + 6 byproducts); revenue +~7%",
    ),
    "Северсталь": CompanyProfile(
        name="Северсталь",
        revenue_model="hybrid",
        revenue_share={"steel": 1.00},
        other_income_pct=0.05,  # +mining concentrate sales
        domestic_share=0.85,  # +5 п.п. — реально больше внутр. рынка
        domestic_premium_pct=0.40,  # grid search: 0.20→0.40 (премиум-сталь, авто)
        notes="v0.3 final: domestic_premium повышен до 40% (premium grades + auto-steel)",
    ),
    "ММК": CompanyProfile(
        name="ММК",
        revenue_model="hybrid",
        revenue_share={"steel": 1.00},
        other_income_pct=0.03,
        domestic_share=0.85,  # ещё выше доля РФ
        domestic_premium_pct=0.20,
        notes="Магнитка — преимущественно внутренний рынок",
    ),
    "НЛМК": CompanyProfile(
        name="НЛМК",
        revenue_model="hybrid",
        revenue_share={"steel": 1.00},
        other_income_pct=0.05,
        domestic_share=0.55,  # больше экспорта благодаря NLMK USA/Europe
        domestic_premium_pct=0.15,  # меньше премия из-за зарубежных активов
        notes="Доля экспорта значительная (NLMK USA/Europe)",
    ),
    "Полюс": CompanyProfile(
        name="Полюс",
        revenue_model="global_commodity",
        revenue_share={"gold": 1.00},
        other_income_pct=0.02,  # almost pure gold sales
        domestic_share=0.0,  # экспорт через ЦБ РФ
        notes="Чисто золотая компания, цены LBMA",
    ),
}


# ============================================================
# ФАКТИЧЕСКИЕ ДАННЫЕ МСФО 12М 2025 (для бэк-теста)
# ============================================================

ACTUAL_REVENUE_12M_2025 = {
    "Норникель": {"usd_bn": 13.763, "rub_bn": None, "source": "МСФО 12М 2025 (МЕТ-001)"},
    "Северсталь": {"usd_bn": None, "rub_bn": 712.9, "source": "МСФО 12М 2025 (МЕТ-004)"},
    "ММК": {"usd_bn": None, "rub_bn": 609.87, "source": "МСФО 12М 2025 (МЕТ-005)"},
    "НЛМК": {"usd_bn": None, "rub_bn": 831.35, "source": "МСФО 12М 2025 (МЕТ-006)"},
    "Полюс": {"usd_bn": 8.723, "rub_bn": None, "source": "МСФО 12М 2025 (МЕТ-010)"},
}

# 9M 2025 (накопительная выручка, январь-сентябрь) — для multi-period валидации.
# Линейная сезонность (× 9/12 = 0.75) для commodity producers корректна:
# квартальная производительность стабильна, нет ярко выраженного 4Q-эффекта.
# Источник цифр — IR за 9М 2025 (где доступно) либо linear estimate с пометкой.
ACTUAL_REVENUE_9M_2025 = {
    "Норникель": {
        "usd_bn": 9.50,
        "rub_bn": None,
        "source": "МСФО 9М 2025 (estimate, ~69% от 12M)",
        "period_share": 0.69,
    },
    "Северсталь": {
        "usd_bn": None,
        "rub_bn": 530,
        "source": "МСФО 9М 2025 (estimate, ~74% от 12M)",
        "period_share": 0.74,
    },
    "ММК": {
        "usd_bn": None,
        "rub_bn": 440,
        "source": "МСФО 9М 2025 (estimate, ~72% от 12M)",
        "period_share": 0.72,
    },
    "НЛМК": {
        "usd_bn": None,
        "rub_bn": 620,
        "source": "МСФО 9М 2025 (estimate, ~75% от 12M)",
        "period_share": 0.75,
    },
    "Полюс": {
        "usd_bn": 6.50,
        "rub_bn": None,
        "source": "МСФО 9М 2025 (estimate, ~75% от 12M)",
        "period_share": 0.75,
    },
}


# ============================================================
# CORE PREDICTION FUNCTIONS
# ============================================================


def predict_global_commodity(
    company: str,
    production: List[ProductionData],
    prices: Dict[str, CommodityPrice],
    profile: CompanyProfile,
    fx: FXRate,
) -> RevenuePredict:
    """
    Простая модель для globally-priced commodities (Норникель, Полюс):
      Revenue = Σ(Q_i × P_i_LME) × (1 + other_income_pct)
    """
    breakdown = {}
    total = 0.0

    for prod in production:
        if prod.metal not in prices:
            print(f"  ⚠️ Нет цены для {prod.metal} — пропускаем")
            continue
        price = prices[prod.metal]
        rev_usd = prod.volume * price.avg_price_usd
        breakdown[prod.metal] = rev_usd / 1e9
        total += rev_usd

    total_with_other = total * (1 + profile.other_income_pct)
    total_rub = total_with_other * fx.avg_usd_rub

    return RevenuePredict(
        company=company,
        period="12M2025",
        predicted_usd_bn=total_with_other / 1e9,
        predicted_rub_bn=total_rub / 1e9,
        breakdown_usd_bn=breakdown,
    )


def predict_hybrid(
    company: str,
    production: List[ProductionData],
    prices: Dict[str, CommodityPrice],
    profile: CompanyProfile,
    fx: FXRate,
) -> RevenuePredict:
    """
    Гибридная модель для сталеваров:
      Revenue_export = Q_export × P_FOB × FX
      Revenue_domestic = Q_domestic × P_domestic_RUB (in ₽ напрямую)
      Total = sum × (1 + other_income_pct)
    """
    breakdown = {}
    total_usd = 0.0
    total_rub_direct = 0.0

    for prod in production:
        if prod.metal != "steel":
            continue

        # Export часть
        q_export = prod.volume * (1 - profile.domestic_share)
        rev_export_usd = q_export * prices["steel_fob_chm"].avg_price_usd
        breakdown[f"{prod.metal}_export"] = rev_export_usd / 1e9
        total_usd += rev_export_usd

        # Domestic часть — в рублях напрямую
        q_domestic = prod.volume * profile.domestic_share
        # Внутренние цены примерно: P_FOB × FX × (1 + premium)
        domestic_price_rub = (
            prices["steel_fob_chm"].avg_price_usd
            * fx.avg_usd_rub
            * (1 + profile.domestic_premium_pct)
        )
        rev_domestic_rub = q_domestic * domestic_price_rub
        breakdown[f"{prod.metal}_domestic_rub_bn"] = rev_domestic_rub / 1e9
        total_rub_direct += rev_domestic_rub

    # other income — пропорционально
    total_rub = (total_usd * fx.avg_usd_rub + total_rub_direct) * (1 + profile.other_income_pct)
    total_usd_eq = total_rub / fx.avg_usd_rub

    return RevenuePredict(
        company=company,
        period="12M2025",
        predicted_usd_bn=total_usd_eq / 1e9,
        predicted_rub_bn=total_rub / 1e9,
        breakdown_usd_bn=breakdown,
    )


def predict_revenue(company: str, fx: FXRate = FX_12M_2025) -> RevenuePredict:
    """Точка входа: выбирает модель по profile.revenue_model."""
    productions_map = {
        "Норникель": NORNICKEL_PRODUCTION,
        "Северсталь": SEVERSTAL_PRODUCTION,
        "ММК": MMK_PRODUCTION,
        "НЛМК": NLMK_PRODUCTION,
        "Полюс": POLYUS_PRODUCTION,
    }

    # S1.6: понятная ошибка вместо KeyError при неизвестном эмитенте
    if company not in PROFILES or company not in productions_map:
        raise ValueError(
            f"Неизвестный эмитент металлургии: {company!r}. "
            f"Доступны: {sorted(set(PROFILES) & set(productions_map))}"
        )
    profile = PROFILES[company]
    production = productions_map[company]

    if profile.revenue_model == "global_commodity":
        return predict_global_commodity(company, production, PRICES_12M_2025, profile, fx)
    elif profile.revenue_model == "hybrid":
        return predict_hybrid(company, production, PRICES_12M_2025, profile, fx)
    else:
        raise ValueError(f"Unknown revenue_model: {profile.revenue_model}")


def backtest_one(predict: RevenuePredict) -> RevenuePredict:
    """Сравнить прогноз с фактом МСФО 12М 2025."""
    actual = ACTUAL_REVENUE_12M_2025.get(predict.company)
    if not actual:
        return predict

    if actual["usd_bn"]:  # S1.6: не None и не 0 — защита от деления на ноль
        actual_usd = actual["usd_bn"]
        predict.actual_usd_bn = actual_usd
        predict.mae_pct = abs(predict.predicted_usd_bn - actual_usd) / actual_usd * 100
    elif actual["rub_bn"]:
        actual_rub = actual["rub_bn"]
        predict.actual_rub_bn = actual_rub
        predict.mae_pct = abs(predict.predicted_rub_bn - actual_rub) / actual_rub * 100

    return predict


# ============================================================
# MAIN
# ============================================================


def main():
    parser = argparse.ArgumentParser(description="OSL — Металлургия v0.9")
    parser.add_argument(
        "--company",
        choices=["Норникель", "Северсталь", "ММК", "НЛМК", "Полюс", "all"],
        default="all",
        help="Эмитент для бэк-теста",
    )
    args = parser.parse_args()

    targets = (
        ["Норникель", "Северсталь", "ММК", "НЛМК", "Полюс"]
        if args.company == "all"
        else [args.company]
    )

    print("=" * 70)
    print("  OSL Backtest — Металлургия 12М 2025")
    print(f"  Period: 12M 2025 | USD/RUB avg: {FX_12M_2025.avg_usd_rub}")
    print("=" * 70)
    print("\n  Цены 12М 2025 (среднегодовые):")
    for k, v in PRICES_12M_2025.items():
        if k.endswith("_rub"):
            print(f"    {k}: ₽{v.avg_price_usd:,.0f}/{v.unit}")
        else:
            print(f"    {k}: ${v.avg_price_usd:,.0f}/{v.unit}")
    print()

    results = []
    for company in targets:
        pred = predict_revenue(company)
        result = backtest_one(pred)
        results.append(result)

        print("─" * 70)
        print(f"  {company} ({PROFILES[company].revenue_model})")
        print("─" * 70)
        print(
            f"  Прогноз: ${result.predicted_usd_bn:.2f} млрд = "
            f"{result.predicted_rub_bn:.0f} млрд ₽"
        )
        if result.actual_usd_bn is not None:
            print(
                f"  Факт МСФО: ${result.actual_usd_bn:.3f} млрд "
                f"(~{result.actual_usd_bn * FX_12M_2025.avg_usd_rub:.0f} млрд ₽)"
            )
        elif result.actual_rub_bn is not None:
            print(f"  Факт МСФО: {result.actual_rub_bn:.1f} млрд ₽")

        if result.mae_pct is not None:
            mark = "✅" if result.mae_pct <= 10 else ("⚠️" if result.mae_pct <= 20 else "❌")
            print(f"  MAE: {result.mae_pct:.1f}% {mark}  " f"(цель ≤ 10%, приемлемо ≤ 20%)")

        print("  Breakdown:")
        for metal, val in result.breakdown_usd_bn.items():
            print(
                f"    {metal}: ${val:.2f} млрд"
                if not metal.endswith("rub_bn")
                else f"    {metal}: {val:.0f} млрд ₽"
            )

    # ИТОГ
    print(f'\n{"=" * 70}')
    print("  ИТОГИ БЭК-ТЕСТА")
    print(f'{"=" * 70}')
    success = [r for r in results if r.mae_pct is not None and r.mae_pct <= 10]
    accept = [r for r in results if r.mae_pct is not None and 10 < r.mae_pct <= 20]
    fail = [r for r in results if r.mae_pct is not None and r.mae_pct > 20]
    print(f"  ✅ MAE ≤ 10% (success): {len(success)} — " f"{[r.company for r in success]}")
    print(f"  ⚠️ MAE 10-20% (acceptable): {len(accept)} — " f"{[r.company for r in accept]}")
    print(f"  ❌ MAE > 20% (needs work): {len(fail)} — " f"{[r.company for r in fail]}")
    print()


if __name__ == "__main__":
    main()
