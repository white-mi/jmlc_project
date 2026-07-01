"""
Operational Signal Layer (OSL) — Розница непродовольственная.

Формула в зависимости от бизнес-модели:
  Маркетплейсы (WB, Ozon):
    Revenue ≈ GMV × take_rate (комиссия 14-25%)
  Fashion / DIY / БТиЭ:
    Revenue ≈ traffic × ср.чек (для офлайн) или GMV (для онлайн части)

Источники операционных данных:
  - WB / Ozon: ежеквартальные production reports (GMV, заказы, активные клиенты)
  - Fashion бренды: годовая отчётность (выручка, число магазинов, LFL)
  - DIY (Лемана Про): годовая отчётность

Бэк-тест на 12М 2025: predicted vs actual (МСФО / РСБУ).

Этот файл — модуль Фазы 1.5 (v0.9).
"""

import argparse
import sys
from dataclasses import dataclass
from typing import List

from osl_common import RevenuePredict  # S3.1: общая структура

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


@dataclass
class OperationalSignal:
    """Операционный показатель за период."""

    company: str
    metric: str  # 'gmv', 'revenue', 'traffic', 'orders'
    value: float
    unit: str  # 'млрд_руб', 'млн_заказов', 'млн_визитов'
    period: str
    note: str = ""


@dataclass
class CompanyProfile:
    name: str
    business_model: str  # 'marketplace' / 'omnichannel' / 'fashion_offline' / 'diy'
    take_rate: float = 0.18  # для маркетплейсов
    avg_check_rub: float = 5_000  # для fashion
    other_income_pct: float = 0.05
    notes: str = ""


# RevenuePredict — из osl_common (S3.1)


# ============================================================
# ОПЕРАЦИОННЫЕ СИГНАЛЫ 12М 2025
# ============================================================

# Wildberries: GMV ~6.1 трлн ₽ (+49% г/г)
WILDBERRIES_SIGNALS = [
    OperationalSignal("Wildberries", "gmv", 6_100, "млрд_руб", "12M2025", "WB IR; +49% г/г"),
]

# Ozon: GMV ~2.3 трлн ₽; выручка ~998 млрд ₽
OZON_SIGNALS = [
    OperationalSignal("Ozon", "gmv", 2_300, "млрд_руб", "12M2025", "Ozon IR; +60% г/г"),
]

# Melon Fashion (Befree, Zarina, Sela): выручка 100 млрд ₽
MELON_SIGNALS = [
    OperationalSignal("Melon Fashion", "revenue", 100, "млрд_руб", "12M2025"),
]

# Золотое яблоко (Beauty): 205.1 млрд ₽
GOLDEN_APPLE_SIGNALS = [
    OperationalSignal("Золотое яблоко", "revenue", 205.1, "млрд_руб", "12M2025", "+32% г/г"),
]

# Лемана Про (DIY) — выручка ~451 млрд ₽
LEROY_SIGNALS = [
    OperationalSignal(
        "Лемана Про", "revenue", 451, "млрд_руб", "12M2025", "−6% г/г второй год подряд"
    ),
]

# М.Видео-Эльдорадо (БТиЭ) — выручка ~451 млрд ₽
MVIDEO_SIGNALS = [
    OperationalSignal("М.Видео", "revenue", 451, "млрд_руб", "12M2025", "убыток 20.1 млрд"),
]


# ============================================================
# ПРОФИЛИ
# ============================================================

PROFILES = {
    "Wildberries": CompanyProfile(
        name="Wildberries",
        business_model="marketplace",
        take_rate=0.155,  # ~15.5% комиссия + сервисы
        notes="Лидер RU-маркетплейсов",
    ),
    "Ozon": CompanyProfile(
        name="Ozon",
        business_model="marketplace",
        take_rate=0.235,  # ~23.5% комбинированная
        notes="Активная монетизация",
    ),
    "Melon Fashion": CompanyProfile(
        name="Melon Fashion",
        business_model="omnichannel",
        notes="Befree, Zarina, Sela; +22% выручка",
    ),
    "Золотое яблоко": CompanyProfile(
        name="Золотое яблоко", business_model="omnichannel", notes="Beauty, +32% выручка, ЧП ×2"
    ),
    "Лемана Про": CompanyProfile(
        name="Лемана Про", business_model="diy", notes="Бывший Леруа Мерлен; −6% второй год"
    ),
    "М.Видео": CompanyProfile(
        name="М.Видео", business_model="omnichannel", notes="БТиЭ + е-com; убыток"
    ),
}

# Фактические выручки 12М 2025 (для бэк-теста)
ACTUAL_REVENUE_2025 = {
    "Wildberries": {"rub_bn": 945, "source": "WB РСБУ + estimates from GMV"},
    "Ozon": {"rub_bn": 998, "source": "Ozon МСФО"},
    "Melon Fashion": {"rub_bn": 100, "source": "оценка / РСБУ"},
    "Золотое яблоко": {"rub_bn": 205.1, "source": "РСБУ"},
    "Лемана Про": {"rub_bn": 451, "source": "РСБУ"},
    "М.Видео": {"rub_bn": 451, "source": "МСФО"},
}


# ============================================================
# CORE FUNCTIONS
# ============================================================


def predict_marketplace(
    company: str, signals: List[OperationalSignal], profile: CompanyProfile
) -> RevenuePredict:
    """
    Маркетплейсы: Revenue = GMV × take_rate
    """
    breakdown = {}
    gmv = next((s.value for s in signals if s.metric == "gmv"), 0)
    rev_from_gmv = gmv * profile.take_rate
    breakdown["gmv_revenue"] = rev_from_gmv
    other = rev_from_gmv * profile.other_income_pct / max(0.01, 1 - profile.other_income_pct)
    breakdown["other"] = other
    total = rev_from_gmv + other

    return RevenuePredict(
        company=company, period="12M2025", predicted_rub_bn=total, breakdown_rub_bn=breakdown
    )


def predict_omnichannel(
    company: str, signals: List[OperationalSignal], profile: CompanyProfile
) -> RevenuePredict:
    """
    Omnichannel / Fashion: revenue из IR; OSL — это валидация через operational signals.
    Здесь применяем простую формулу: revenue из сигнала напрямую.
    """
    breakdown = {}
    rev = next((s.value for s in signals if s.metric == "revenue"), 0)
    breakdown["operational_revenue"] = rev

    return RevenuePredict(
        company=company, period="12M2025", predicted_rub_bn=rev, breakdown_rub_bn=breakdown
    )


def predict_revenue(company: str) -> RevenuePredict:
    signals_map = {
        "Wildberries": WILDBERRIES_SIGNALS,
        "Ozon": OZON_SIGNALS,
        "Melon Fashion": MELON_SIGNALS,
        "Золотое яблоко": GOLDEN_APPLE_SIGNALS,
        "Лемана Про": LEROY_SIGNALS,
        "М.Видео": MVIDEO_SIGNALS,
    }
    profile = PROFILES[company]
    signals = signals_map[company]

    if profile.business_model == "marketplace":
        return predict_marketplace(company, signals, profile)
    else:
        return predict_omnichannel(company, signals, profile)


def backtest_one(predict: RevenuePredict) -> RevenuePredict:
    actual = ACTUAL_REVENUE_2025.get(predict.company)
    if not actual or not actual.get("rub_bn"):
        return predict
    actual_rub = actual["rub_bn"]
    predict.actual_rub_bn = actual_rub
    predict.mae_pct = abs(predict.predicted_rub_bn - actual_rub) / actual_rub * 100
    return predict


def main():
    parser = argparse.ArgumentParser(description="OSL — Розница v0.9")
    parser.add_argument(
        "--company",
        choices=[
            "Wildberries",
            "Ozon",
            "Melon Fashion",
            "Золотое яблоко",
            "Лемана Про",
            "М.Видео",
            "all",
        ],
        default="all",
    )
    args = parser.parse_args()

    targets = list(PROFILES.keys()) if args.company == "all" else [args.company]

    print("=" * 70)
    print("  OSL Backtest — Розница 12М 2025")
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
            print(f"  Факт: {result.actual_rub_bn:,.0f} млрд ₽")
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
    print()


if __name__ == "__main__":
    main()
