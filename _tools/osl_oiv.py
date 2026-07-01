"""
Operational Signal Layer (OSL) — ОИВ (регионы).

Формула:
  Доходы региона ≈ налог_на_прибыль_донорских_отраслей + НДФЛ + НДПИ_доля
                  + межбюджетные трансферты (МБТ) + прочие
                  − региональные расходы

Здесь фокус — регионы-доноры с сильной НГ-зависимостью:
  ХМАО-Югра, Тюменская обл, Сахалинская, Татарстан

Источники:
  - Минфин субъектов РФ — бюджет на 2025-2026
  - Минэнерго — добыча нефти/газа по регионам
  - Минфин РФ — структура НГ-доходов

Бэк-тест на 12М 2025: predicted vs actual бюджет регионов.
"""

import argparse
import sys
from dataclasses import dataclass

from osl_common import RevenuePredict  # S3.1: общая структура

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


@dataclass
class RegionProfile:
    name: str
    region_type: str  # 'oil_donor' / 'gas_donor' / 'industrial' / 'recipient' / 'metropolis'
    oil_production_mt: float = 0  # млн т нефти
    gas_production_bcm: float = 0  # млрд куб.м газа
    own_revenue_share: float = 0.0  # доля собственных доходов
    federal_transfer_share: float = 0.0  # доля МБТ
    notes: str = ""


# RevenuePredict — из osl_common (S3.1)


# ============================================================
# КОНТЕКСТ 12М 2025
# ============================================================

# Brent average 2025: $78
# Urals average 2025: $55
# USD/RUB avg: 89

# НДПИ нефть (упрощённо): ~$22/барр × курс на тонну = $22 × 7.33 × 89 = ~14 350 ₽/т
# Хотя НДПИ платится в фед.бюджет, регионы получают свою долю через:
#   - налог на прибыль нефтекомпаний (отдаётся в регион размещения / добычи)
#   - НДФЛ нефтяников (где работают)

OIL_PROFIT_TAX_PER_TON_RUB = 4_500  # ₽/т (оценка налога на прибыль с тонны добычи нефти)
GAS_PROFIT_TAX_PER_BCM_RUB = 850_000_000  # ₽/млрд м³ (оценка для газа)


# ============================================================
# РЕГИОНЫ
# ============================================================

PROFILES = {
    "ХМАО-Югра": RegionProfile(
        name="ХМАО-Югра",
        region_type="oil_donor",
        oil_production_mt=200,  # ~200 млн т нефти
        own_revenue_share=0.85,
        federal_transfer_share=0.05,
        notes="Главный нефтяной регион РФ; 70% бюджета от нефти",
    ),
    "Тюменская обл.": RegionProfile(
        name="Тюменская обл.",
        region_type="oil_donor",
        oil_production_mt=12,  # сама Тюмень — мало; основная добыча в ХМАО/ЯНАО
        own_revenue_share=0.80,
        federal_transfer_share=0.10,
        notes="Тюмень собирает с ХМАО+ЯНАО; перераспределение через сложную модель",
    ),
    "ЯНАО": RegionProfile(
        name="ЯНАО",
        region_type="gas_donor",
        gas_production_bcm=540,  # ~540 млрд м³ газа (главный газовый регион)
        oil_production_mt=22,
        own_revenue_share=0.85,
        federal_transfer_share=0.05,
        notes="Главный газовый регион (Газпром + Новатэк)",
    ),
    "Татарстан": RegionProfile(
        name="Татарстан",
        region_type="oil_donor",
        oil_production_mt=33,  # Татнефть + ТАИФ
        own_revenue_share=0.80,
        federal_transfer_share=0.05,
        notes="Татнефть; нефтехимия; промышленный регион",
    ),
    "Сахалинская обл.": RegionProfile(
        name="Сахалинская обл.",
        region_type="oil_donor",
        oil_production_mt=15,
        gas_production_bcm=33,  # Сахалин-1, Сахалин-2
        own_revenue_share=0.75,
        federal_transfer_share=0.10,
        notes="Сахалин-1, Сахалин-2; СРП-доля; 58.5% СД от НГ",
    ),
}


# Фактические бюджеты регионов 2025 (СД = собственные доходы из открытых источников)
ACTUAL_BUDGET_2025 = {
    "ХМАО-Югра": {"rub_bn": 360, "source": "Минфин ХМАО / открытый бюджет"},
    "Тюменская обл.": {"rub_bn": 240, "source": "Минфин Тюмени"},
    "ЯНАО": {"rub_bn": 295, "source": "Минфин ЯНАО"},
    "Татарстан": {"rub_bn": 480, "source": "Минфин Татарстана"},
    "Сахалинская обл.": {"rub_bn": 172, "source": "СД 2025 = 171.8 млрд"},
}


# ============================================================
# CORE
# ============================================================


def predict_oil_donor(name: str, profile: RegionProfile) -> RevenuePredict:
    """
    Нефтяной регион:
      Revenue ≈ Q_oil × profit_tax_per_ton + другие_налоги + МБТ
    """
    breakdown = {}

    # Налог на прибыль нефтянок в регионе
    rev_oil = profile.oil_production_mt * 1e6 * OIL_PROFIT_TAX_PER_TON_RUB
    breakdown["oil_profit_tax"] = rev_oil / 1e9

    # Газ (если есть)
    rev_gas = profile.gas_production_bcm * GAS_PROFIT_TAX_PER_BCM_RUB
    breakdown["gas_profit_tax"] = rev_gas / 1e9

    # НДФЛ + другие региональные налоги (приблизительно 30-40% от нефтегазовых = размером с
    # отрасль; промышленность, сельское хозяйство, торговля)
    other_own = (rev_oil + rev_gas) * 0.4
    breakdown["ndfl_other_taxes"] = other_own / 1e9

    # МБТ (федеральные трансферты)
    own_revenue = rev_oil + rev_gas + other_own
    if profile.federal_transfer_share > 0 and profile.own_revenue_share > 0:
        target_total = own_revenue / profile.own_revenue_share
        mbt = target_total * profile.federal_transfer_share
    else:
        mbt = 0
        target_total = own_revenue
    breakdown["federal_transfers"] = mbt / 1e9

    return RevenuePredict(
        company=name,
        period="12M2025",
        predicted_rub_bn=target_total / 1e9,
        breakdown_rub_bn=breakdown,
    )


def predict_revenue(region: str) -> RevenuePredict:
    profile = PROFILES[region]
    return predict_oil_donor(region, profile)


def backtest_one(predict: RevenuePredict) -> RevenuePredict:
    actual = ACTUAL_BUDGET_2025.get(predict.company)
    if not actual:
        return predict
    actual_rub = actual["rub_bn"]
    predict.actual_rub_bn = actual_rub
    predict.mae_pct = abs(predict.predicted_rub_bn - actual_rub) / actual_rub * 100
    return predict


def main():
    parser = argparse.ArgumentParser(description="OSL — ОИВ v0.9")
    parser.add_argument("--region", choices=list(PROFILES.keys()) + ["all"], default="all")
    args = parser.parse_args()

    targets = list(PROFILES.keys()) if args.region == "all" else [args.region]

    print("=" * 70)
    print("  OSL Backtest — ОИВ (регионы) 12М 2025")
    print(f"  Налог на прибыль нефти: {OIL_PROFIT_TAX_PER_TON_RUB:,} ₽/т")
    print(f"  Налог на прибыль газа: {GAS_PROFIT_TAX_PER_BCM_RUB/1e9:.2f} млрд ₽/млрд м³")
    print("=" * 70)

    results = []
    for region in targets:
        pred = predict_revenue(region)
        result = backtest_one(pred)
        results.append(result)

        print("─" * 70)
        print(f"  {region} ({PROFILES[region].region_type})")
        print("─" * 70)
        print(f"  Прогноз: {result.predicted_rub_bn:,.0f} млрд ₽")
        if result.actual_rub_bn:
            print(f"  Факт СД: {result.actual_rub_bn:,.0f} млрд ₽")
        if result.mae_pct is not None:
            mark = "✅" if result.mae_pct <= 10 else ("⚠️" if result.mae_pct <= 20 else "❌")
            print(f"  MAE: {result.mae_pct:.1f}% {mark}")
        print("  Breakdown:")
        for k, v in result.breakdown_rub_bn.items():
            print(f"    {k}: {v:.0f} млрд ₽")

    print(f'\n{"=" * 70}')
    success = [r for r in results if r.mae_pct is not None and r.mae_pct <= 10]
    accept = [r for r in results if r.mae_pct is not None and 10 < r.mae_pct <= 20]
    fail = [r for r in results if r.mae_pct is not None and r.mae_pct > 20]
    print(f"  ✅ MAE ≤ 10%: {len(success)} — {[r.company for r in success]}")
    print(f"  ⚠️ MAE 10-20%: {len(accept)} — {[r.company for r in accept]}")
    print(f"  ❌ MAE > 20%: {len(fail)} — {[r.company for r in fail]}")


if __name__ == "__main__":
    main()
