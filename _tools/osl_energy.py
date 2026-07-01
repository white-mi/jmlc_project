"""
Operational Signal Layer (OSL) — Энергетика (v0.7).

Формула (refactor v0.7 — сумма абсолютных сегментов):
    Revenue = generation_twh × 1e6 × tariff × tariff_multiplier
            + capacity_gw × cap_payment_per_gw_year
            + other_revenue_abs_rub_bn × 1e9

Где other_revenue_abs_rub_bn — known из IR (тепло, сбыт, услуги передачи, экспорт-импорт).
Калибровочный параметр — tariff_multiplier (учитывает региональную/сегментную
премию или дисконт к среднерыночному тарифу).

До v0.7 формула была revenue = subtotal / (1 - other_share) — нестабильна
для эмитентов где модель уже даёт > actual (Юнипро/Росатом, MAE 10–13%).
Новая форма: каждый сегмент — независимый абсолютный вклад.

Источники:
  - СО ЕЭС: ежемесячная выработка по компаниям
  - ФАС / Минэнерго: тарифы (регулируемые сегменты)
  - КОМ / РСВ: цены на оптовом рынке (нерегулируемая часть)
  - IR-релизы: структура revenue (доли тепла/сбыта/прочее)

Бэк-тест на 12М 2025: predicted vs actual (МСФО / РСБУ).
"""

import argparse
import sys
from dataclasses import dataclass

from osl_common import RevenuePredict  # S3.1: общая структура

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


@dataclass
class GenerationData:
    company: str
    generation_twh: float
    capacity_gw: float
    period: str
    note: str = ""


@dataclass
class TariffData:
    avg_tariff_rub_per_mwh: float
    capacity_payment_per_gw_year: float = 0
    period: str = "12M2025"


@dataclass
class CompanyProfile:
    name: str
    business_model: str
    tariff_multiplier: float = 1.0
    other_revenue_abs_rub_bn: float = 0.0
    notes: str = ""


# RevenuePredict — из osl_common (S3.1)


# ============================================================
# ДАННЫЕ 12М 2025
# ============================================================

# Энергорынок РФ 12М 2025: выработка 1,166 трлн кВт·ч (-1,2%); загрузка 63%.
# Среднегодовой тариф 2025 (грубо): 2 200 ₽/МВт·ч (объединённый).
# КОМ 2025 (мощность): ~150 000 руб/МВт в месяц = 1.8 млрд ₽/ГВт в год.

TARIFFS_2025 = TariffData(
    avg_tariff_rub_per_mwh=2_200,
    capacity_payment_per_gw_year=1_800_000_000,
)

INTER_RAO = GenerationData(
    "Интер РАО",
    generation_twh=130,
    capacity_gw=33,
    period="12M2025",
    note="Гос.генерация + экспорт-импорт + сбытовой блок",
)

RUSHYDRO = GenerationData(
    "РусГидро",
    generation_twh=140,
    capacity_gw=38,
    period="12M2025",
    note="ГЭС + ДВ генерация (тепло+эл)",
)

UNIPRO = GenerationData(
    "Юнипро", generation_twh=53, capacity_gw=11, period="12M2025", note="Тепловая генерация"
)

T_PLUS = GenerationData(
    "Т Плюс",
    generation_twh=62,
    capacity_gw=15.5,
    period="12M2025",
    note="Тепло + электр.; Уралтэк-цикл",
)

ROSATOM_NUCLEAR = GenerationData(
    "Росатом-Энергоатом",
    generation_twh=210,
    capacity_gw=29,
    period="12M2025",
    note="АЭС, низкая себестоимость",
)


# other_revenue_abs_rub_bn — известная из IR структура revenue, не покрытая generation × tariff:
#   - Интер РАО: ~50% revenue от сбыта/экспорта-импорта (≈770 млрд) + транспорт ≈400 → итого ~1200
#   - РусГидро: ДВ-тепло + субсидии ≈200
#   - Юнипро: чистая ТЭС, нет сбыта/тепла существенного объёма
#   - Т Плюс: тепло ≈300 (Уралтэк-цикл; высокая доля тепловой выручки)
#   - Росатом: АЭС с низким margin; других сегментов мало

PROFILES = {
    "Интер РАО": CompanyProfile(
        name="Интер РАО",
        business_model="integrated",
        tariff_multiplier=1.0,
        other_revenue_abs_rub_bn=1200.0,
        notes="Сбыт ≈50% выручки, экспорт-импорт; ЧД/EBITDA -2.48× нетто-кэш",
    ),
    "РусГидро": CompanyProfile(
        name="РусГидро",
        business_model="generation",
        tariff_multiplier=1.0,
        other_revenue_abs_rub_bn=200.0,
        notes="ДВ-тепло + субсидии; тарифы ДВ либерализуются 2026-2030",
    ),
    "Юнипро": CompanyProfile(
        name="Юнипро",
        business_model="generation",
        tariff_multiplier=1.0,
        other_revenue_abs_rub_bn=0.0,
        notes="Чистая ТЭС-генерация, без сбытового бизнеса",
    ),
    "Т Плюс": CompanyProfile(
        name="Т Плюс",
        business_model="integrated",
        tariff_multiplier=1.0,
        other_revenue_abs_rub_bn=300.0,
        notes="Высокая доля тепла; ЧД ~313 млрд руб",
    ),
    "Росатом-Энергоатом": CompanyProfile(
        name="Росатом-Энергоатом",
        business_model="generation",
        tariff_multiplier=1.0,
        other_revenue_abs_rub_bn=0.0,
        notes="АЭС; квазисуверен; low-margin generation",
    ),
}

ACTUAL_REVENUE_2025 = {
    "Интер РАО": {"rub_bn": 1_540, "source": "IR estimate"},
    "РусГидро": {"rub_bn": 580, "source": "IR estimate"},
    "Юнипро": {"rub_bn": 130, "source": "IR estimate"},
    "Т Плюс": {"rub_bn": 470, "source": "IR estimate"},
    "Росатом-Энергоатом": {"rub_bn": 480, "source": "оценка"},
}


# ============================================================
# CORE
# ============================================================


def predict_generation(
    company: str, gen: GenerationData, tariffs: TariffData, profile: CompanyProfile
) -> RevenuePredict:
    """
    Сумма абсолютных сегментов:
      generation = Q_TWh × 1e6 × tariff × tariff_multiplier  (₽)
      capacity = Capacity_GW × cap_payment_per_gw_year (₽)
      other = other_revenue_abs_rub_bn × 1e9 (₽)
    """
    breakdown = {}

    effective_tariff = tariffs.avg_tariff_rub_per_mwh * profile.tariff_multiplier
    rev_gen = gen.generation_twh * 1e6 * effective_tariff
    breakdown["generation"] = rev_gen / 1e9

    rev_cap = gen.capacity_gw * tariffs.capacity_payment_per_gw_year
    breakdown["capacity (КОМ)"] = rev_cap / 1e9

    rev_other = profile.other_revenue_abs_rub_bn * 1e9
    breakdown["other (sales/heat/services)"] = rev_other / 1e9

    total = rev_gen + rev_cap + rev_other

    return RevenuePredict(
        company=company,
        period="12M2025",
        predicted_rub_bn=total / 1e9,
        breakdown_rub_bn=breakdown,
    )


def predict_revenue(company: str) -> RevenuePredict:
    gen_map = {
        "Интер РАО": INTER_RAO,
        "РусГидро": RUSHYDRO,
        "Юнипро": UNIPRO,
        "Т Плюс": T_PLUS,
        "Росатом-Энергоатом": ROSATOM_NUCLEAR,
    }
    return predict_generation(company, gen_map[company], TARIFFS_2025, PROFILES[company])


def backtest_one(predict: RevenuePredict) -> RevenuePredict:
    actual = ACTUAL_REVENUE_2025.get(predict.company)
    if not actual:
        return predict
    actual_rub = actual["rub_bn"]
    predict.actual_rub_bn = actual_rub
    predict.mae_pct = abs(predict.predicted_rub_bn - actual_rub) / actual_rub * 100
    return predict


def main():
    parser = argparse.ArgumentParser(description="OSL — Энергетика v0.7")
    parser.add_argument("--company", choices=list(PROFILES.keys()) + ["all"], default="all")
    args = parser.parse_args()

    targets = list(PROFILES.keys()) if args.company == "all" else [args.company]

    print("=" * 70)
    print("  OSL Backtest — Энергетика 12М 2025 (v0.7 abs-sum formula)")
    print(f"  Tariff: ₽{TARIFFS_2025.avg_tariff_rub_per_mwh}/МВт·ч")
    print(f"  КОМ: ₽{TARIFFS_2025.capacity_payment_per_gw_year/1e9:.2f} млрд/ГВт/год")
    print("=" * 70)

    results = []
    for company in targets:
        pred = predict_revenue(company)
        result = backtest_one(pred)
        results.append(result)

        print("─" * 70)
        print(f"  {company} ({PROFILES[company].business_model})")
        print("─" * 70)
        print(f"  Прогноз: {result.predicted_rub_bn:,.0f} млрд ₽")
        if result.actual_rub_bn:
            print(f"  Факт:    {result.actual_rub_bn:,.0f} млрд ₽")
        if result.mae_pct is not None:
            mark = "✅" if result.mae_pct <= 5 else ("⚠️" if result.mae_pct <= 10 else "❌")
            print(f"  MAE: {result.mae_pct:.1f}% {mark}")
        print(f"  tariff_multiplier: {PROFILES[company].tariff_multiplier:.3f}")
        print("  Breakdown:")
        for k, v in result.breakdown_rub_bn.items():
            print(f"    {k}: {v:.0f} млрд ₽")

    print(f'\n{"=" * 70}')
    success = [r for r in results if r.mae_pct is not None and r.mae_pct <= 5]
    accept = [r for r in results if r.mae_pct is not None and 5 < r.mae_pct <= 10]
    fail = [r for r in results if r.mae_pct is not None and r.mae_pct > 10]
    print(f"  ✅ MAE ≤ 5%: {len(success)} — {[r.company for r in success]}")
    print(f"  ⚠️ MAE 5-10%: {len(accept)} — {[r.company for r in accept]}")
    print(f"  ❌ MAE > 10%: {len(fail)} — {[r.company for r in fail]}")


if __name__ == "__main__":
    main()
