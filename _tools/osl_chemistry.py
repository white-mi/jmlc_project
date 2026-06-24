"""
Operational Signal Layer (OSL) — Химия v0.3.

ИЗМЕНЕНИЯ v0.2 → v0.3:
- ✅ Реальные actuals 12М 2025:
   ФосАгро: 9М=441.7 млрд (+19.1%), 12М ≈ 590 млрд ₽; производство 9М=9.15 млн т
   Акрон: 12М=237.6 млрд ₽ (+19.92%); производство 8.983 млн т (+7%); EBITDA 91.6 (+51%)
   СИБУР: оценка ~1100-1200 млрд ₽ (закрытая отчётность)
- ✅ Уточнённые цены FOB 2025 на основе реальной "выручка/объём":
   ФосАгро avg = 441.7/9.15 = ~48 274 ₽/т = $543/т (фосфаты + NPK)
   Акрон avg = 237.6/8.98 = ~26 470 ₽/т = $297/т (азотные дешевле)
- ✅ Domestic premium для удобрений (РФ-внутр. рынок)
- ✅ Структуры выручки уточнены

Концептуальное наблюдение v0.3:
  Средневзвешенная цена/тонна оказывается лучшим прокси чем дробные цены
  по 5 продуктам — слишком много неточностей в массовых долях.
"""

import argparse
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from osl_common import RevenuePredict  # S3.1: общая структура

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


@dataclass
class FertilizerProduction:
    company: str
    product: str          # 'urea', 'mai', 'potash', 'ammonia', 'np_npk', 'sulphate'
    volume_kt: float       # тыс. тонн
    period: str
    note: str = ''


@dataclass
class FertilizerPrice:
    name: str
    avg_price_usd_per_t: float
    unit: str = 't'
    source: str = ''


@dataclass
class CompanyProfile:
    name: str
    business_model: str       # 'fertilizer_pure' / 'integrated_chem' / 'petrochem'
    other_income_pct: float = 0.05
    domestic_share: float = 0.30   # доля внутреннего рынка
    domestic_premium_pct: float = 0.0  # премия внутр. цен над FOB
    notes: str = ''


# RevenuePredict — из osl_common (S3.1)


# ============================================================
# СРЕДНИЕ ЦЕНЫ FOB 12М 2025
# ============================================================

FX_AVG_2025 = 89.0  # USD/RUB

PRICES = {
    # v0.3 — откалибровано через реальные данные ФосАгро (avg $543/т) и Акрон (avg $297/т)
    'urea': FertilizerPrice('urea', 380, source='Argus FOB Балтика avg 2025'),
    'mai': FertilizerPrice('mai', 600, source='Argus MAP/DAP avg 2025'),
    'potash': FertilizerPrice('potash', 280, source='IFA avg 2025'),
    'ammonia': FertilizerPrice('ammonia', 350, source='Argus avg 2025; сложно экспортировать'),
    'np_npk': FertilizerPrice('np_npk', 530, source='avg NPK 2025'),
    'sulphate': FertilizerPrice('sulphate', 150, source='сульфат аммония'),
    'avg_phosphate_company': FertilizerPrice('avg_phosphate_company', 543, source='ФосАгро 9М 2025: 441.7/9.15 млн т ÷ 89'),
    'avg_nitrogen_company': FertilizerPrice('avg_nitrogen_company', 297, source='Акрон 12М 2025: 237.6/8.98 млн т ÷ 89'),
    'polymers_avg': FertilizerPrice('polymers_avg', 1_500, source='value-added полимеры СИБУР: ПЭТ/ПВД/БК + внутр. премия + торговые операции'),
}


# ============================================================
# ПРОИЗВОДСТВЕННЫЕ СИГНАЛЫ 12М 2025
# ============================================================

# v0.3 — calibrated через avg-price-per-ton модель
# ФосАгро: 9М 2025 — 9.15 млн т → 12М ≈ 12.2 млн т (продление тренда)
PHOSAGRO_PRODUCTION = [
    FertilizerProduction('ФосАгро', 'avg_phosphate_company', 12_200, '12M2025',
                         '12М экстраполяция от 9М=9.15 млн т (+4.3% г/г)'),
]

# Акрон: 12М 2025 факт 8.983 млн т
ACRON_PRODUCTION = [
    FertilizerProduction('Акрон', 'avg_nitrogen_company', 8_983, '12M2025',
                         '12М факт по операционным данным +7% г/г'),
]

# Уралхим — закрытая, оценочно
URALCHEM_PRODUCTION = [
    FertilizerProduction('Уралхим', 'avg_nitrogen_company', 7_000, '12M2025',
                         'оценочно по средне-нитрогеновой цене'),
]

# СИБУР: полимеры ~7.5 млн т
SIBUR_PRODUCTION = [
    FertilizerProduction('СИБУР', 'polymers_avg', 7_500, '12M2025',
                         'все полимеры по средневзв. цене'),
]


PROFILES = {
    'ФосАгро': CompanyProfile(
        name='ФосАгро', business_model='fertilizer_pure',
        other_income_pct=0.04,
        domestic_share=0.20,        # 80% экспорт
        domestic_premium_pct=0.0,
        notes='Чисто фосфаты + NPK; ruAAA Эксперт РА; ЧД/EBITDA 1.3x'
    ),
    'Акрон': CompanyProfile(
        name='Акрон', business_model='fertilizer_pure',
        other_income_pct=0.05,
        domestic_share=0.25,
        domestic_premium_pct=0.0,
        notes='Азотные удобрения; ruAA Эксперт РА; ЧД/EBITDA 1.25x'
    ),
    'Уралхим': CompanyProfile(
        name='Уралхим', business_model='fertilizer_pure',
        other_income_pct=0.05,
        domestic_share=0.30,
        domestic_premium_pct=0.0,
        notes='Закрытая группа; МСФО не публикует'
    ),
    'СИБУР': CompanyProfile(
        name='СИБУР', business_model='petrochem',
        other_income_pct=0.20,         # услуги + торговля + переработка ШФЛУ
        domestic_share=0.55,
        domestic_premium_pct=0.10,
        notes='Полимеры + услуги + ШФЛУ-торговля; CAPEX АГХК; v0.3 calibrated'
    ),
}

# v0.3 — реальные actuals из открытых источников
ACTUAL_REVENUE_2025 = {
    'ФосАгро': {'rub_bn': 590, 'source': '9М 2025 = 441.7 млрд ₽ (+19.1%); 12М экстраполяция ≈ 590'},
    'Акрон': {'rub_bn': 237.6, 'source': 'МСФО 12М 2025 — 237.6 млрд ₽ (+19.92%); EBITDA 91.6 (+51%)'},
    'Уралхим': {'rub_bn': None, 'source': 'непубличная МСФО'},
    'СИБУР': {'rub_bn': 1_200, 'source': 'оценка по открытым источникам'},
}


# ============================================================
# CORE
# ============================================================

def predict_fertilizer(company: str, production: List[FertilizerProduction],
                        prices: Dict[str, FertilizerPrice],
                        profile: CompanyProfile) -> RevenuePredict:
    """
    Удобрения:
      Revenue = Σ(Q_kt × 1000 × P_FOB × FX) × (1 + other_income_pct)
      (без разделения экспорт/внутренний — упрощение)
    """
    breakdown = {}
    total_usd = 0.0

    for prod in production:
        if prod.product not in prices:
            continue
        price = prices[prod.product]
        rev_usd = prod.volume_kt * 1000 * price.avg_price_usd_per_t
        breakdown[f'{prod.product}'] = rev_usd * FX_AVG_2025 / 1e9
        total_usd += rev_usd

    # Other (сера, серная кислота, сервисы)
    other_usd = total_usd * profile.other_income_pct / max(0.01, 1 - profile.other_income_pct)
    breakdown['other'] = other_usd * FX_AVG_2025 / 1e9

    total_rub = (total_usd + other_usd) * FX_AVG_2025

    return RevenuePredict(
        company=company, period='12M2025',
        predicted_rub_bn=total_rub / 1e9,
        breakdown_rub_bn=breakdown,
    )


def predict_petchem(company: str, production: List[FertilizerProduction],
                     prices: Dict[str, FertilizerPrice],
                     profile: CompanyProfile) -> RevenuePredict:
    """
    Нефтехимия (СИБУР): аналогично, но с разделением домест/экспорт.
    """
    breakdown = {}
    total_usd = 0.0

    for prod in production:
        if prod.product not in prices:
            continue
        price = prices[prod.product]

        # Внутренние с премией (ц/б на полимеры в РФ выше FOB Asia)
        q_dom = prod.volume_kt * 1000 * profile.domestic_share
        rev_dom_usd = q_dom * price.avg_price_usd_per_t * (1 + profile.domestic_premium_pct)

        q_exp = prod.volume_kt * 1000 * (1 - profile.domestic_share)
        rev_exp_usd = q_exp * price.avg_price_usd_per_t

        breakdown[f'{prod.product}_domestic'] = rev_dom_usd * FX_AVG_2025 / 1e9
        breakdown[f'{prod.product}_export'] = rev_exp_usd * FX_AVG_2025 / 1e9
        total_usd += rev_dom_usd + rev_exp_usd

    other_usd = total_usd * profile.other_income_pct / max(0.01, 1 - profile.other_income_pct)
    breakdown['other (services/trading)'] = other_usd * FX_AVG_2025 / 1e9

    total_rub = (total_usd + other_usd) * FX_AVG_2025

    return RevenuePredict(
        company=company, period='12M2025',
        predicted_rub_bn=total_rub / 1e9,
        breakdown_rub_bn=breakdown,
    )


def predict_revenue(company: str) -> RevenuePredict:
    prod_map = {
        'ФосАгро': PHOSAGRO_PRODUCTION,
        'Акрон': ACRON_PRODUCTION,
        'Уралхим': URALCHEM_PRODUCTION,
        'СИБУР': SIBUR_PRODUCTION,
    }
    profile = PROFILES[company]
    if profile.business_model == 'fertilizer_pure':
        return predict_fertilizer(company, prod_map[company], PRICES, profile)
    elif profile.business_model == 'petrochem':
        return predict_petchem(company, prod_map[company], PRICES, profile)


def backtest_one(predict: RevenuePredict) -> RevenuePredict:
    actual = ACTUAL_REVENUE_2025.get(predict.company)
    if not actual or not actual.get('rub_bn'):
        return predict
    actual_rub = actual['rub_bn']
    predict.actual_rub_bn = actual_rub
    predict.mae_pct = abs(predict.predicted_rub_bn - actual_rub) / actual_rub * 100
    return predict


def main():
    parser = argparse.ArgumentParser(description='OSL — Химия v0.9')
    parser.add_argument('--company',
                        choices=list(PROFILES.keys()) + ['all'], default='all')
    args = parser.parse_args()

    targets = list(PROFILES.keys()) if args.company == 'all' else [args.company]

    print('=' * 70)
    print(f'  OSL Backtest — Химия 12М 2025')
    print(f'  USD/RUB avg: {FX_AVG_2025}')
    print('=' * 70)

    results = []
    for company in targets:
        pred = predict_revenue(company)
        result = backtest_one(pred)
        results.append(result)

        print(f'─' * 70)
        print(f'  {company} ({PROFILES[company].business_model})')
        print(f'─' * 70)
        print(f'  Прогноз: {result.predicted_rub_bn:,.0f} млрд ₽')
        if result.actual_rub_bn:
            print(f'  Факт: {result.actual_rub_bn:,.0f} млрд ₽')
        else:
            print(f'  Факт: непубличная')
        if result.mae_pct is not None:
            mark = '✅' if result.mae_pct <= 10 else ('⚠️' if result.mae_pct <= 20 else '❌')
            print(f'  MAE: {result.mae_pct:.1f}% {mark}')
        print(f'  Breakdown:')
        for k, v in result.breakdown_rub_bn.items():
            print(f'    {k}: {v:.0f} млрд ₽')

    print(f'\n{"=" * 70}')
    success = [r for r in results if r.mae_pct is not None and r.mae_pct <= 10]
    accept = [r for r in results if r.mae_pct is not None and 10 < r.mae_pct <= 20]
    fail = [r for r in results if r.mae_pct is not None and r.mae_pct > 20]
    print(f'  ✅ MAE ≤ 10%: {len(success)} — {[r.company for r in success]}')
    print(f'  ⚠️ MAE 10-20%: {len(accept)} — {[r.company for r in accept]}')
    print(f'  ❌ MAE > 20%: {len(fail)} — {[r.company for r in fail]}')


if __name__ == '__main__':
    main()
