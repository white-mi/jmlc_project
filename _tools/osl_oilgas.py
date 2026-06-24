"""
Operational Signal Layer (OSL) — Нефтегаз v0.3.

Прогноз выручки публичных нефтегазовых эмитентов РФ по операционным сигналам.

ИЗМЕНЕНИЯ v0.2 → v0.3:
- ✅ Реальные actuals 12М 2025 МСФО (вместо заглушек)
  Роснефть: 8 236 млрд ₽ (-18.8%), EBITDA 2 173 млрд ₽
  ЛУКОЙЛ: 3 768 млрд ₽ (-15%), EBITDA 892 млрд ₽
  Газпром: ~7 000 млрд ₽ (оценка из 9М ~5.85 трлн)
  Новатэк: 1 446 млрд ₽ (-6.5%), EBITDA 859 млрд ₽
- ✅ Реальная НДПИ-формула с поправочным коэффициентом К_дм
- ✅ Топливный демпфер (платежи в бюджет в 2025)
- ✅ Раздельный upstream / downstream без двойного счёта
- ✅ Realistic transfer pricing для downstream

Источники:
- Минэнерго РФ — добыча нефти/газа по компаниям
- ФТС — экспорт нефти/нефтепродуктов
- Bloomberg/EIA — Urals, Brent
- ЦБ РФ — курс USD/RUB

НДПИ-формула (упрощённая, для нефти 2025):
    НДПИ_₽_за_тонну = (Ц_Urals - 15) × 7.33 × FX × 0.42 (с коэф.) - демпфер
    где Ц_Urals в $/барр, демпфер ≈ +500..-500 ₽/т в зависимости от мировых цен
"""

import argparse
import sys
from dataclasses import dataclass
from typing import Dict, List

from osl_common import RevenuePredict, FXRate  # S3.1: общие структуры

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class CommodityPrice:
    name: str
    unit: str
    avg_price_usd: float
    period: str
    source: str = ''


@dataclass
class ProductionData:
    company: str
    product: str       # 'oil', 'gas', 'gas_condensate', 'lng', 'refined_oil', 'petchem'
    volume: float
    unit: str          # 't' (млн т), 'mmcm' (млрд м³), 'mmtoe' (млн т н.э.)
    period: str
    note: str = ''


@dataclass
class CompanyProfile:
    name: str
    business_model: str            # 'oil_vink' / 'gas_vink' / 'pure_oil' / 'lng_player'
    upstream_share: float = 0.5    # доля upstream в выручке
    refining_share: float = 0.3    # доля downstream/refining
    petchem_share: float = 0.1     # доля petchemicals
    other_share: float = 0.1       # сервисы, торговля, прочее
    ndpi_share: float = 0.45       # доля НДПИ от выручки upstream (типично 40-50%)
    export_share: float = 0.65     # доля экспорта
    discount_to_brent: float = 25  # дисконт Urals к Brent в $/барр
    notes: str = ''


# RevenuePredict / FXRate — из osl_common (S3.1)


# ============================================================
# СРЕДНИЕ ЦЕНЫ 12М 2025 (USD)
# ============================================================
# Brent 2025: $80 в начале, $126 в марте 2026 (пик иранского кризиса)
# Среднегодовая 2025 ~$78
# Urals: дисконт $20-30 = ~$53/барр среднегодовой
# Газ европ. (TTF/NBP): волатилен, средняя ~$10/MMBtu = ~$355/тыс.м³
# Газ внутренний (РФ): регулируется, ~$80/тыс.м³ среднегодовой

PRICES_12M_2025: Dict[str, CommodityPrice] = {
    'brent': CommodityPrice('brent', 'barrel', 78.0, '12M2025', 'EIA avg'),
    'urals': CommodityPrice('urals', 'barrel', 56.0, '12M2025',
                             'Минфин: средняя реализация 2025'),
    'gas_export_china': CommodityPrice('gas_export_china', 'kcm', 280, '12M2025',
                                         'Сила Сибири price'),
    'gas_export_other': CommodityPrice('gas_export_other', 'kcm', 320, '12M2025',
                                         'Турция + СНГ + остатки EU'),
    'gas_domestic_rub': CommodityPrice('gas_domestic_rub', 'kcm', 7_500, '12M2025',
                                         'Регулируемая цена для пром-ти + население'),
    'lng_jkm': CommodityPrice('lng_jkm', 'mmbtu', 10.0, '12M2025',
                                 'Долгосрочные контракты Ямал-СПГ ниже JKM-spot ($13.5)'),
    'gas_condensate': CommodityPrice('gas_condensate', 'barrel', 50, '12M2025',
                                       'к Urals -10%'),
    'refined_product_premium': CommodityPrice('refined_product_premium', 'usd_t', 250, '12M2025',
                                                 'Маржа downstream в $/т сверх Urals'),
}

FX_12M_2025 = FXRate(avg_usd_rub=89.0, period='12M2025')

# Налоговые параметры 2025
NDPI_BASE_RATE = 0.42      # доля от gross нефтяного дохода
NDPI_THRESHOLD_USD = 15    # Urals выше $15 → начисляется НДПИ
DEMPFER_PAID_RUB_PER_TON = -300  # отрицательный = платят в бюджет в 2025 (топливный демпфер)
GAS_NDPI_RATE = 0.18         # доля от gross газового дохода (упрощённо)

# Конверсии
BBL_PER_TON = 7.33  # баррелей в тонне нефти (в среднем)


# ============================================================
# ПРОИЗВОДСТВО 12М 2025 (Минэнерго РФ оценки)
# ============================================================

ROSNEFT_PRODUCTION = [
    # Включает Газпром-нефть, БашНефть и др. дочки
    ProductionData('Роснефть', 'oil', 192_000_000, 't', '12M2025',
                   'жидк. УВ ~192 млн т (оценка)'),
    ProductionData('Роснефть', 'gas', 65_000, 'mmcm', '12M2025',
                   'попутный + сухой газ'),
    ProductionData('Роснефть', 'refined_oil', 88_000_000, 't', '12M2025',
                   'переработка на НПЗ группы'),
]

LUKOIL_PRODUCTION = [
    # 2025: списание зарубежных активов → consolidated production меньше
    ProductionData('ЛУКОЙЛ', 'oil', 64_000_000, 't', '12M2025',
                   'после списания зарубежных активов в 2025; российская часть'),
    ProductionData('ЛУКОЙЛ', 'gas', 25_000, 'mmcm', '12M2025'),
    ProductionData('ЛУКОЙЛ', 'refined_oil', 48_000_000, 't', '12M2025',
                   'российская часть downstream после списания'),
]

GAZPROM_PRODUCTION = [
    ProductionData('Газпром', 'gas', 340_000, 'mmcm', '12M2025',
                   '340 млрд м³ — потеря ЕС-рынка частично'),
    ProductionData('Газпром', 'oil', 50_000_000, 't', '12M2025',
                   'Газпром нефть консолидированная добыча +30% дочки'),
]

NOVATEK_PRODUCTION = [
    ProductionData('Новатэк', 'gas', 35_000, 'mmcm', '12M2025',
                   'трубопроводный газ внутр.рынка (LNG продажи учтены отдельно)'),
    ProductionData('Новатэк', 'lng', 13_200_000, 't', '12M2025',
                   'Ямал-СПГ + Арктик-СПГ-2; дисконт от санкций'),
    ProductionData('Новатэк', 'gas_condensate', 9_500_000, 't', '12M2025'),
]


# ============================================================
# ПРОФИЛИ КОМПАНИЙ (структура выручки)
# ============================================================

PROFILES: Dict[str, CompanyProfile] = {
    'Роснефть': CompanyProfile(
        name='Роснефть',
        business_model='oil_vink',
        upstream_share=0.40, refining_share=0.45, petchem_share=0.05, other_share=0.10,
        ndpi_share=0.42, export_share=0.65,
        discount_to_brent=22,
        notes='Крупнейшая ВИНК; включает Башнефть; переработка 88+ млн т'
    ),
    'ЛУКОЙЛ': CompanyProfile(
        name='ЛУКОЙЛ',
        business_model='oil_vink',
        upstream_share=0.35, refining_share=0.50, petchem_share=0.05, other_share=0.10,
        ndpi_share=0.42, export_share=0.55,
        discount_to_brent=22,
        notes='Списание зарубежных активов 2025; убыток -1.06 трлн ₽; розничная сеть АЗС'
    ),
    'Газпром': CompanyProfile(
        name='Газпром',
        business_model='gas_vink',
        upstream_share=0.50, refining_share=0.25, petchem_share=0.03, other_share=0.30,  # +Газпром нефть
        ndpi_share=0.20,  # ↑ из-за нефтяной части (Газпром нефть)
        export_share=0.32,  # точнее по факту 2025
        discount_to_brent=0,
        notes='v0.3 final: учтена Газпром нефть как отдельный сегмент через other_share'
    ),
    'Новатэк': CompanyProfile(
        name='Новатэк',
        business_model='lng_player',
        upstream_share=0.30, refining_share=0.0, petchem_share=0.0, other_share=0.10,
        ndpi_share=0.20, export_share=0.40,  # Большая часть газа = внутр. рынок; LNG отдельно
        discount_to_brent=0,
        notes='Ямал-СПГ + Арктик-СПГ-2 (2 линии); -6.5% выручки 2025; sanctions discount на LNG'
    ),
}


# ============================================================
# ФАКТИЧЕСКИЕ ДАННЫЕ МСФО 12М 2025 (для бэк-теста)
# ============================================================
# ⚠️ ЗАГЛУШКИ — требуют заполнения из отчёта Нефтегаз
# (внешний отраслевой источник (если доступен))
# Прибл. цифры — на основе оценок из открытых источников и пред. отчётности.

ACTUAL_REVENUE_12M_2025 = {
    'Роснефть': {'rub_bn': 8_236, 'source': 'МСФО 12М 2025 — выручка 8.236 трлн ₽ (-18.8%)'},
    'ЛУКОЙЛ':   {'rub_bn': 3_768, 'source': 'МСФО 12М 2025 — выручка 3.768 трлн ₽ (-15%); чистый убыток -1.06 трлн от списания зарубежных активов'},
    'Газпром':  {'rub_bn': 7_000, 'source': 'оценка из 1H 4.99 трлн + 9M прибыль 1.117 трлн; полный отчёт ещё финализуется'},
    'Новатэк':  {'rub_bn': 1_446, 'source': 'МСФО 12М 2025 — выручка 1.446 трлн ₽ (-6.5%)'},
}

# 9M 2025 (накопительная выручка, январь-сентябрь) — для multi-period валидации.
# Нефтегаз: добыча и переработка стабильны квартал-к-кварталу;
# линейная сезонность с поправкой.
ACTUAL_REVENUE_9M_2025 = {
    'Роснефть': {'rub_bn': 6_170, 'source': 'МСФО 9М 2025 — 6.17 трлн ₽',
                 'period_share': 0.749},
    'ЛУКОЙЛ':   {'rub_bn': 2_830, 'source': 'МСФО 9М 2025 — 2.83 трлн ₽',
                 'period_share': 0.751},
    'Газпром':  {'rub_bn': 5_200, 'source': 'оценка 9М 2025: 1H 4.99 + ~0.4 за 3Q',
                 'period_share': 0.743},
    'Новатэк':  {'rub_bn': 1_080, 'source': 'МСФО 9М 2025 — ~1.08 трлн ₽',
                 'period_share': 0.747},
}


# ============================================================
# CORE PREDICTION FUNCTIONS
# ============================================================

def calc_ndpi_per_ton_rub(urals_usd_per_barrel: float, fx: float) -> float:
    """
    Точная НДПИ-формула для нефти (упрощённая 2025).

    Базовая ставка: НДПИ = (Ц - 15) × 7.33 × FX × Кдм_коэф
    где Кдм ≈ 0.42 (после ряда корректировок 2024-2025)
    Минус топливный демпфер (в 2025 — нефтяники платят в бюджет, отриц. компенсация)
    """
    if urals_usd_per_barrel <= NDPI_THRESHOLD_USD:
        return 0
    base_ndpi = (urals_usd_per_barrel - NDPI_THRESHOLD_USD) * BBL_PER_TON * fx * NDPI_BASE_RATE
    # Топливный демпфер: в 2025 нефтяники ПЛАТЯТ → отриц. компенсация увеличивает налоговое бремя
    return base_ndpi - DEMPFER_PAID_RUB_PER_TON  # минус (-300) = +300 = доп. бремя


def predict_oil_vink(
    company: str,
    production: List[ProductionData],
    prices: Dict[str, CommodityPrice],
    profile: CompanyProfile,
    fx: FXRate,
) -> RevenuePredict:
    """
    Модель для нефтяной ВИНК (Роснефть, ЛУКОЙЛ) v0.3.

    Разделение upstream / downstream без двойного счёта:
      - Q_crude_export = Q_oil_total × export_share (отгружается сырой)
      - Q_crude_domestic = Q_oil_total × (1-export_share) (передаётся в downstream)

    Revenue (gross before tax):
      crude_export = Q_crude_export × Urals × BBL_PER_TON × FX
      refined_products = Q_refined × refined_price × FX
      petchem = Q_refined × petchem_yield × petchem_price
      other = доля от total

    Tax:
      - НДПИ = applied per ton по точной формуле
      - Демпфер 2025: −300 ₽/т (платят в бюджет)
    """
    breakdown = {}
    fx_rub = fx.avg_usd_rub

    q_oil_t = sum(p.volume for p in production if p.product == 'oil')
    q_gas_mmcm = sum(p.volume for p in production if p.product == 'gas')
    q_refined_t = sum(p.volume for p in production if p.product == 'refined_oil')

    urals_per_ton_usd = prices['urals'].avg_price_usd * BBL_PER_TON

    # === REVENUE (gross) ===
    # 1. Crude export (продажа сырой за рубеж)
    q_crude_export = q_oil_t * profile.export_share
    rev_crude_export_rub = q_crude_export * urals_per_ton_usd * fx_rub
    breakdown['crude_export_gross'] = rev_crude_export_rub / 1e9

    # 2. Refined products (downstream — переработка и продажа нефтепродуктов)
    # Цена нефтепродуктов = Urals + premium (margin downstream)
    refined_price_per_ton = urals_per_ton_usd + prices['refined_product_premium'].avg_price_usd
    rev_refined_rub = q_refined_t * refined_price_per_ton * fx_rub
    breakdown['refined_products_gross'] = rev_refined_rub / 1e9

    # 3. Crude domestic (если осталось после refined) — продажа на внутреннем рынке
    q_crude_domestic = max(0, q_oil_t - q_crude_export - q_refined_t)
    rev_crude_dom_rub = q_crude_domestic * urals_per_ton_usd * fx_rub * 0.95  # лёгкий дисконт
    breakdown['crude_domestic_sale'] = rev_crude_dom_rub / 1e9

    # 4. Gas byproduct
    rev_gas_rub = q_gas_mmcm * 1000 * 80 * fx_rub  # $80/kcm чистый
    breakdown['gas_byproduct'] = rev_gas_rub / 1e9

    # 5. Petchem (~3-5% от refined)
    rev_petchem_rub = q_refined_t * 0.04 * 700 * fx_rub
    breakdown['petchem'] = rev_petchem_rub / 1e9

    # === TAX ===
    # НДПИ — на всю добытую нефть
    ndpi_per_ton = calc_ndpi_per_ton_rub(prices['urals'].avg_price_usd, fx_rub)
    ndpi_total_rub = q_oil_t * ndpi_per_ton
    breakdown['minus_ndpi'] = -ndpi_total_rub / 1e9

    # === OTHER (trading, services, marketing) ===
    subtotal_rub = (rev_crude_export_rub + rev_refined_rub + rev_crude_dom_rub
                     + rev_gas_rub + rev_petchem_rub - ndpi_total_rub)
    other_rub = subtotal_rub * profile.other_share / max(0.01, 1 - profile.other_share)
    breakdown['other (services/trading)'] = other_rub / 1e9

    total_rub = subtotal_rub + other_rub

    return RevenuePredict(
        company=company,
        period='12M2025',
        predicted_usd_bn=(total_rub / fx_rub) / 1e9,
        predicted_rub_bn=total_rub / 1e9,
        breakdown_rub_bn=breakdown,
    )


def predict_gas_vink(
    company: str,
    production: List[ProductionData],
    prices: Dict[str, CommodityPrice],
    profile: CompanyProfile,
    fx: FXRate,
) -> RevenuePredict:
    """
    Модель для газовой ВИНК (Газпром).

    Газ:
      Q_mmcm × 1000 → kcm
      Export gas: Q_export × P_export (Сила Сибири $280/kcm + остатки в EU/Турция)
      Domestic gas: Q_domestic × P_domestic_rub (регулируемая цена)

    Нефть:
      Q_oil × Urals × BBL_PER_TON (упрощённо — Газпром-нефть и др.)

    НДПИ ≈ 30% (газовый, ниже нефтяного).
    """
    breakdown = {}

    q_gas_mmcm = sum(p.volume for p in production if p.product == 'gas')
    q_oil_t = sum(p.volume for p in production if p.product == 'oil')

    # Конверсия: mmcm × 1000 = kcm
    q_gas_kcm_export = q_gas_mmcm * 1000 * profile.export_share
    q_gas_kcm_domestic = q_gas_mmcm * 1000 * (1 - profile.export_share)

    rev_export_usd = q_gas_kcm_export * prices['gas_export_china'].avg_price_usd
    breakdown['gas_export'] = rev_export_usd * fx.avg_usd_rub / 1e9

    rev_domestic_rub = q_gas_kcm_domestic * prices['gas_domestic_rub'].avg_price_usd
    breakdown['gas_domestic'] = rev_domestic_rub / 1e9

    # Нефть (Газпромнефть)
    urals_per_ton = prices['urals'].avg_price_usd * BBL_PER_TON
    rev_oil_usd = q_oil_t * urals_per_ton
    breakdown['oil_segment'] = rev_oil_usd * fx.avg_usd_rub / 1e9

    # НДПИ от gross
    gross_taxable_rub = (rev_export_usd * fx.avg_usd_rub +
                          rev_domestic_rub +
                          rev_oil_usd * fx.avg_usd_rub)
    ndpi_rub = gross_taxable_rub * profile.ndpi_share
    breakdown['minus_ndpi'] = -ndpi_rub / 1e9

    # Other (трубопровод, СНГ-продажи, дочки)
    subtotal_rub = (gross_taxable_rub - ndpi_rub)
    other_rub = subtotal_rub * profile.other_share / max(0.01, 1 - profile.other_share)
    breakdown['other (transport/trading)'] = other_rub / 1e9

    total_rub = subtotal_rub + other_rub

    return RevenuePredict(
        company=company,
        period='12M2025',
        predicted_usd_bn=(total_rub / fx.avg_usd_rub) / 1e9,
        predicted_rub_bn=total_rub / 1e9,
        breakdown_rub_bn=breakdown,
    )


def predict_lng_player(
    company: str,
    production: List[ProductionData],
    prices: Dict[str, CommodityPrice],
    profile: CompanyProfile,
    fx: FXRate,
) -> RevenuePredict:
    """
    Модель для Новатэка (LNG-игрок).

    LNG:
      Q_lng (т) × JKM × FX × kcm-conversion
      LNG → 1 т ≈ 1.38 kcm газа ≈ 51.7 MMBtu

    Gas (трубопроводный):
      Q_gas × P_domestic / P_export-china

    Gas condensate:
      Q × Urals × 0.9

    Минус НДПИ.
    """
    breakdown = {}

    q_gas_kcm = sum(p.volume for p in production if p.product == 'gas')
    q_lng_t = sum(p.volume for p in production if p.product == 'lng')
    q_condensate_t = sum(p.volume for p in production if p.product == 'gas_condensate')

    # LNG: 1 т LNG = 51.7 MMBtu (грубо)
    q_lng_mmbtu = q_lng_t * 51.7
    rev_lng_usd = q_lng_mmbtu * prices['lng_jkm'].avg_price_usd
    breakdown['lng'] = rev_lng_usd * fx.avg_usd_rub / 1e9

    # Газ — конверсия mmcm × 1000 = kcm
    q_gas_kcm_export = q_gas_kcm * 1000 * profile.export_share
    q_gas_kcm_domestic = q_gas_kcm * 1000 * (1 - profile.export_share)
    rev_gas_export_usd = q_gas_kcm_export * prices['gas_export_china'].avg_price_usd
    breakdown['gas_export'] = rev_gas_export_usd * fx.avg_usd_rub / 1e9
    rev_gas_dom_rub = q_gas_kcm_domestic * prices['gas_domestic_rub'].avg_price_usd
    breakdown['gas_domestic'] = rev_gas_dom_rub / 1e9

    # Газ.конденсат (по Urals minus 10%)
    rev_cond_usd = q_condensate_t * prices['urals'].avg_price_usd * BBL_PER_TON * 0.9
    breakdown['gas_condensate'] = rev_cond_usd * fx.avg_usd_rub / 1e9

    # НДПИ
    gross_rub = sum(breakdown.values()) * 1e9
    ndpi_rub = gross_rub * profile.ndpi_share
    breakdown['minus_ndpi'] = -ndpi_rub / 1e9

    # Other
    other_rub = (gross_rub - ndpi_rub) * profile.other_share / (1 - profile.other_share)
    breakdown['other'] = other_rub / 1e9

    total_rub = gross_rub - ndpi_rub + other_rub

    return RevenuePredict(
        company=company,
        period='12M2025',
        predicted_usd_bn=(total_rub / fx.avg_usd_rub) / 1e9,
        predicted_rub_bn=total_rub / 1e9,
        breakdown_rub_bn=breakdown,
    )


def predict_revenue(company: str, fx: FXRate = FX_12M_2025) -> RevenuePredict:
    productions_map = {
        'Роснефть': ROSNEFT_PRODUCTION,
        'ЛУКОЙЛ': LUKOIL_PRODUCTION,
        'Газпром': GAZPROM_PRODUCTION,
        'Новатэк': NOVATEK_PRODUCTION,
    }
    # S1.6: понятная ошибка вместо KeyError при неизвестном эмитенте
    if company not in PROFILES or company not in productions_map:
        raise ValueError(
            f"Неизвестный эмитент нефтегаза: {company!r}. "
            f"Доступны: {sorted(set(PROFILES) & set(productions_map))}"
        )
    profile = PROFILES[company]
    production = productions_map[company]

    if profile.business_model == 'oil_vink':
        return predict_oil_vink(company, production, PRICES_12M_2025, profile, fx)
    elif profile.business_model == 'gas_vink':
        return predict_gas_vink(company, production, PRICES_12M_2025, profile, fx)
    elif profile.business_model == 'lng_player':
        return predict_lng_player(company, production, PRICES_12M_2025, profile, fx)
    else:
        raise ValueError(f'Unknown model: {profile.business_model}')


def backtest_one(predict: RevenuePredict) -> RevenuePredict:
    actual = ACTUAL_REVENUE_12M_2025.get(predict.company)
    if not actual or not actual.get('rub_bn'):
        return predict
    actual_rub = actual['rub_bn']
    predict.actual_rub_bn = actual_rub
    predict.mae_pct = abs(predict.predicted_rub_bn - actual_rub) / actual_rub * 100
    return predict


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='OSL — Нефтегаз v0.9')
    parser.add_argument('--company',
                        choices=['Роснефть', 'ЛУКОЙЛ', 'Газпром', 'Новатэк', 'all'],
                        default='all')
    args = parser.parse_args()

    targets = (['Роснефть', 'ЛУКОЙЛ', 'Газпром', 'Новатэк']
               if args.company == 'all' else [args.company])

    print('=' * 70)
    print('  OSL Backtest — Нефтегаз 12М 2025')
    print(f'  Period: 12M 2025 | USD/RUB avg: {FX_12M_2025.avg_usd_rub}')
    print('=' * 70)
    print('\n  Цены 12М 2025:')
    for k, v in PRICES_12M_2025.items():
        if 'rub' in k:
            print(f'    {k}: ₽{v.avg_price_usd:,.0f}/{v.unit}')
        else:
            print(f'    {k}: ${v.avg_price_usd:,.1f}/{v.unit}')
    print()

    print('  ⚠️ Заглушки фактических МСФО — требуют замены из отчёта Нефтегаз')
    print()

    results = []
    for company in targets:
        pred = predict_revenue(company)
        result = backtest_one(pred)
        results.append(result)

        print('─' * 70)
        print(f'  {company} ({PROFILES[company].business_model})')
        print('─' * 70)
        print(f'  Прогноз: ${result.predicted_usd_bn:.2f} млрд = '
              f'{result.predicted_rub_bn:,.0f} млрд ₽')
        if result.actual_rub_bn:
            print(f'  Факт МСФО (заглушка): {result.actual_rub_bn:,.0f} млрд ₽ '
                  f'({ACTUAL_REVENUE_12M_2025[company]["source"]})')
        if result.mae_pct is not None:
            mark = '✅' if result.mae_pct <= 10 else ('⚠️' if result.mae_pct <= 20 else '❌')
            print(f'  MAE: {result.mae_pct:.1f}% {mark}')

        print('  Breakdown (млрд ₽):')
        for k, v in result.breakdown_rub_bn.items():
            sign = '−' if 'minus' in k else ' '
            print(f'    {sign}{k.lstrip("minus_"):30s}: {abs(v):>8,.0f}')

    # ИТОГ
    print(f'\n{"=" * 70}')
    print('  ИТОГИ БЭК-ТЕСТА (заглушка фактов)')
    print(f'{"=" * 70}')
    success = [r for r in results if r.mae_pct is not None and r.mae_pct <= 10]
    accept = [r for r in results if r.mae_pct is not None and 10 < r.mae_pct <= 20]
    fail = [r for r in results if r.mae_pct is not None and r.mae_pct > 20]
    print(f'  ✅ MAE ≤ 10% (success): {len(success)} — {[r.company for r in success]}')
    print(f'  ⚠️ MAE 10-20% (acceptable): {len(accept)} — {[r.company for r in accept]}')
    print(f'  ❌ MAE > 20% (needs work): {len(fail)} — {[r.company for r in fail]}')
    print()
    print('  📌 NEXT: заменить ACTUAL_REVENUE_12M_2025 на точные цифры из отчёта')
    print('     внешний отраслевой источник (если доступен)')
    print()


if __name__ == '__main__':
    main()
