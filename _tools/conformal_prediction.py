"""
OSL v0.4 — Inductive Conformal Prediction для калиброванных интервалов.

Концепция:
  При прогнозе ΔRevenue / Revenue точечная оценка недостаточна. Нужен интервал
  с заданным уровнем confidence (90%): "Revenue лежит в [low, high] с вероятностью 90%".

Подход (т.к. исторических данных мало):
  Perturbation-based conformal prediction:
    1. Прогон OSL с base параметрами → predicted_base
    2. Генерация N=200 perturbed-вариантов:
       - Цены ±5% (uncertainty в FOB / LME / LBMA / Urals)
       - Объёмы ±3% (uncertainty в production reports)
       - FX ±2% (USD/RUB volatility)
       - Other_share ±20% относительно (структурная неопределённость)
    3. Получаем распределение predicted values
    4. Quantiles 5% и 95% → 90% prediction interval

Validation против факта:
  Для эмитентов где известен actual: проверяем содержит ли interval actual value.

  ⚠️ ВАЖНО (S2.3 / находка F5): это IN-SAMPLE проверка. Параметры OSL калиброваны
  на тех же ACTUAL_REVENUE_*_2025, против которых строится «покрытие», поэтому
  высокий % покрытия (≈96%) отражает остроту/согласованность интервалов, а НЕ
  обобщающую способность (out-of-sample). Настоящий temporal hold-out (калибровка
  на 9M → проверка на отложенном 12M) требует НЕЗАВИСИМЫХ 9M-actuals из IR —
  текущие 9M выведены из 12M через period_share (annualized-9M ≡ 12M). См. план
  S4.2 и skip-тест tests/test_conformal.py::test_holdout_coverage_metallurgy.

Источник методологии:
  arXiv:2508.14078 — Out-of-Sample Hydrocarbon Production Forecasting
  с Inductive Conformal Prediction
"""

import sys
import argparse
import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# v0.6: автоматически применить сохранённые калибровки при импорте.
# S(v0.9 loop): лог идёт в stderr, НЕ в stdout — иначе ломается контракт
# `run_pipeline.py --json` (import-time print попадал в JSON-вывод).
try:
    from osl_calibrator import apply_all_calibrations

    _calib_applied = apply_all_calibrations(verbose=False)
    _total = sum(v for v in _calib_applied.values() if isinstance(v, int))
    print(f"  ✅ Auto-applied {_total} calibrations from calibration/*.json", file=sys.stderr)
except Exception as _e:
    print(f"  ⚠️ Calibrations not applied: {_e}", file=sys.stderr)


@dataclass
class PredictionInterval:
    """Прогноз с интервалом."""

    company: str
    predicted_base: float
    predicted_low: float  # 5% quantile
    predicted_high: float  # 95% quantile
    interval_width_pct: float
    actual: Optional[float] = None
    actual_in_interval: Optional[bool] = None
    coverage_metric: Optional[str] = None  # "BELOW" / "INSIDE" / "ABOVE"


def _finalize_interval(
    company: str, base_pred: float, predictions, actual: Optional[float], conf: float = 0.90
) -> PredictionInterval:
    """Общий «хвост» всех make_interval_*: квантильный интервал из распределения
    perturbed-прогнозов → PredictionInterval. Раньше дублировался ~6 раз
    (v0.9 loop: дедупликация при сохранении бэспоук perturbation-логики на модуль)."""
    if predictions is None or len(predictions) == 0:
        return PredictionInterval(company, base_pred, base_pred, base_pred, 0.0, actual, None, None)
    arr = np.array(predictions)
    alpha = 1 - conf
    low = float(np.quantile(arr, alpha / 2))
    high = float(np.quantile(arr, 1 - alpha / 2))
    width_pct = (high - low) / base_pred * 100 if base_pred else 0.0
    in_int = cov = None
    if actual is not None:
        in_int = low <= actual <= high
        cov = "BELOW" if actual < low else ("ABOVE" if actual > high else "INSIDE")
    return PredictionInterval(company, base_pred, low, high, width_pct, actual, in_int, cov)


def perturb_params(
    base_params: dict,
    perturbation_config: dict = None,
    rng: np.random.Generator = None,
) -> dict:
    """
    Сгенерировать perturbed-копию параметров.

    perturbation_config = {
      'prices': {'std_pct': 0.05},       # ±5%
      'volumes': {'std_pct': 0.03},      # ±3%
      'fx': {'std_pct': 0.02},           # ±2%
      'other_share': {'std_relative': 0.20},  # ±20%
    }
    """
    if rng is None:
        rng = np.random.default_rng()
    if perturbation_config is None:
        perturbation_config = {
            "prices": {"std_pct": 0.05},
            "volumes": {"std_pct": 0.03},
            "fx": {"std_pct": 0.02},
            "other_share": {"std_relative": 0.20},
        }

    perturbed = {}
    for key, val in base_params.items():
        if "price" in key.lower():
            std = perturbation_config["prices"]["std_pct"]
            perturbed[key] = val * (1 + rng.normal(0, std))
        elif "volume" in key.lower() or "production" in key.lower():
            std = perturbation_config["volumes"]["std_pct"]
            perturbed[key] = val * (1 + rng.normal(0, std))
        elif key == "fx" or key == "usd_rub":
            std = perturbation_config["fx"]["std_pct"]
            perturbed[key] = val * (1 + rng.normal(0, std))
        elif key == "other_share":
            std = perturbation_config["other_share"]["std_relative"]
            perturbed[key] = max(0.0, min(0.5, val * (1 + rng.normal(0, std))))
        else:
            perturbed[key] = val
    return perturbed


def conformal_predict(
    predictor_fn: Callable[[dict], float],
    base_params: dict,
    n_simulations: int = 200,
    confidence_level: float = 0.90,
    perturbation_config: dict = None,
    seed: int = 42,
) -> dict:
    """
    Запустить N симуляций и вернуть base + low/high интервал.

    Args:
      predictor_fn: функция (params: dict) -> float (predicted revenue)
      base_params: базовые параметры
      n_simulations: количество perturbed-прогонов
      confidence_level: уровень confidence (0.90 = 90% interval)
      perturbation_config: см. perturb_params
      seed: для воспроизводимости

    Returns:
      {
        'predicted_base': base prediction,
        'predicted_low': lower bound,
        'predicted_high': upper bound,
        'mean': среднее симуляций,
        'std': std симуляций,
        'all_predictions': список всех симуляций
      }
    """
    rng = np.random.default_rng(seed)
    base_pred = predictor_fn(base_params)

    predictions = []
    for _ in range(n_simulations):
        perturbed = perturb_params(base_params, perturbation_config, rng)
        try:
            pred = predictor_fn(perturbed)
            if pred is not None and not np.isnan(pred) and not np.isinf(pred):
                predictions.append(pred)
        except Exception:
            continue

    if not predictions:
        return {
            "predicted_base": base_pred,
            "predicted_low": base_pred,
            "predicted_high": base_pred,
            "mean": base_pred,
            "std": 0,
            "all_predictions": [],
        }

    predictions = np.array(predictions)
    alpha = 1 - confidence_level
    low_q = alpha / 2
    high_q = 1 - alpha / 2
    low = float(np.quantile(predictions, low_q))
    high = float(np.quantile(predictions, high_q))

    return {
        "predicted_base": base_pred,
        "predicted_low": low,
        "predicted_high": high,
        "mean": float(np.mean(predictions)),
        "std": float(np.std(predictions)),
        "n_simulations": len(predictions),
        "confidence_level": confidence_level,
    }


def make_interval_from_metallurgy(
    company: str, n_sim: int = 200, conf: float = 0.90
) -> PredictionInterval:
    """Wrapper для osl_metallurgy.predict_revenue с conformal."""
    from osl_metallurgy import (
        predict_revenue,
        FX_12M_2025,
        PRICES_12M_2025,
        ACTUAL_REVENUE_12M_2025,
    )

    # Базовые параметры — extract from globals (упрощённо)
    base_params = {
        "fx": FX_12M_2025.avg_usd_rub,
        "price_copper": PRICES_12M_2025["copper"].avg_price_usd,
        "price_nickel": PRICES_12M_2025["nickel"].avg_price_usd,
        "price_gold": PRICES_12M_2025["gold"].avg_price_usd,
        "price_steel_fob": PRICES_12M_2025["steel_fob_chm"].avg_price_usd,
    }

    def predictor_fn(params):
        # Применяем perturbation: модифицируем глобальные prices
        # (для чистоты — было бы лучше передавать как аргументы predict_revenue)
        original_fx = FX_12M_2025.avg_usd_rub
        original_copper = PRICES_12M_2025["copper"].avg_price_usd
        original_nickel = PRICES_12M_2025["nickel"].avg_price_usd
        original_gold = PRICES_12M_2025["gold"].avg_price_usd
        original_steel = PRICES_12M_2025["steel_fob_chm"].avg_price_usd

        FX_12M_2025.avg_usd_rub = params["fx"]
        PRICES_12M_2025["copper"].avg_price_usd = params["price_copper"]
        PRICES_12M_2025["nickel"].avg_price_usd = params["price_nickel"]
        PRICES_12M_2025["gold"].avg_price_usd = params["price_gold"]
        PRICES_12M_2025["steel_fob_chm"].avg_price_usd = params["price_steel_fob"]

        try:
            pred = predict_revenue(company)
            result = pred.predicted_rub_bn
        finally:
            # Restore
            FX_12M_2025.avg_usd_rub = original_fx
            PRICES_12M_2025["copper"].avg_price_usd = original_copper
            PRICES_12M_2025["nickel"].avg_price_usd = original_nickel
            PRICES_12M_2025["gold"].avg_price_usd = original_gold
            PRICES_12M_2025["steel_fob_chm"].avg_price_usd = original_steel

        return result

    result = conformal_predict(predictor_fn, base_params, n_sim, conf)

    # Validation against actual
    actual_data = ACTUAL_REVENUE_12M_2025.get(company)
    actual = None
    in_interval = None
    coverage = None
    if actual_data:
        if actual_data.get("rub_bn"):
            actual = actual_data["rub_bn"]
        elif actual_data.get("usd_bn"):
            actual = actual_data["usd_bn"] * FX_12M_2025.avg_usd_rub

        if actual is not None:
            in_interval = result["predicted_low"] <= actual <= result["predicted_high"]
            if actual < result["predicted_low"]:
                coverage = "BELOW"
            elif actual > result["predicted_high"]:
                coverage = "ABOVE"
            else:
                coverage = "INSIDE"

    width_pct = (
        (result["predicted_high"] - result["predicted_low"]) / result["predicted_base"] * 100
    )

    return PredictionInterval(
        company=company,
        predicted_base=result["predicted_base"],
        predicted_low=result["predicted_low"],
        predicted_high=result["predicted_high"],
        interval_width_pct=width_pct,
        actual=actual,
        actual_in_interval=in_interval,
        coverage_metric=coverage,
    )


def make_interval_generic(
    company: str,
    osl_module_name: str,
    n_sim: int = 200,
    conf: float = 0.90,
) -> PredictionInterval:
    """
    Generic conformal wrapper для любого OSL-модуля.

    Подход: перехватываем поля цен/курса, применимые для отрасли (PRICES/FX),
    сохраняем оригиналы, применяем шум в каждой симуляции, откатываем после.
    actual выводится из ACTUAL_* модуля автоматически.

    (v0.9 loop: убран мёртвый параметр perturbation_fields — он не использовался,
    а вызывающие передавали туда число actual, которое молча игнорировалось.)
    """
    import importlib

    osl_module = importlib.import_module(osl_module_name)

    # Найти actuals
    actual = None
    actual_dict = (
        getattr(osl_module, "ACTUAL_REVENUE_2025", None)
        or getattr(osl_module, "ACTUAL_REVENUE_12M_2025", None)
        or getattr(osl_module, "ACTUAL_BUDGET_2025", None)
    )
    if actual_dict and company in actual_dict:
        rec = actual_dict[company]
        if rec.get("rub_bn"):
            actual = rec["rub_bn"]
        elif rec.get("usd_bn"):
            fx = getattr(osl_module, "FX_12M_2025", None) or getattr(
                osl_module, "FX_AVG_2025", 89.0
            )
            if hasattr(fx, "avg_usd_rub"):
                fx_val = fx.avg_usd_rub
            else:
                fx_val = float(fx)
            actual = rec["usd_bn"] * fx_val

    # base prediction
    base_pred_obj = osl_module.predict_revenue(company)
    base_pred = base_pred_obj.predicted_rub_bn

    # PRICES global (or PRICES_12M_2025)
    PRICES = getattr(osl_module, "PRICES", None) or getattr(osl_module, "PRICES_12M_2025", None)
    FX = getattr(osl_module, "FX_12M_2025", None) or getattr(osl_module, "FX_AVG_2025", None)

    # Save originals
    original_prices = {}
    if PRICES:
        for k, v in PRICES.items():
            if hasattr(v, "avg_price_usd"):
                original_prices[k] = v.avg_price_usd
            elif hasattr(v, "avg_price_usd_per_t"):
                original_prices[k] = v.avg_price_usd_per_t
    original_fx = None
    if FX is not None:
        if hasattr(FX, "avg_usd_rub"):
            original_fx = FX.avg_usd_rub
        elif isinstance(FX, (int, float)):
            original_fx = float(FX)

    rng = np.random.default_rng(42)
    predictions = []
    for _ in range(n_sim):
        # Perturb
        if PRICES:
            for k, v in PRICES.items():
                std = 0.05  # 5% на цены
                if hasattr(v, "avg_price_usd"):
                    v.avg_price_usd = original_prices[k] * (1 + rng.normal(0, std))
                elif hasattr(v, "avg_price_usd_per_t"):
                    v.avg_price_usd_per_t = original_prices[k] * (1 + rng.normal(0, std))
        if FX is not None and original_fx and hasattr(FX, "avg_usd_rub"):
            FX.avg_usd_rub = original_fx * (1 + rng.normal(0, 0.02))
        try:
            pred = osl_module.predict_revenue(company)
            if pred and pred.predicted_rub_bn and not np.isnan(pred.predicted_rub_bn):
                predictions.append(pred.predicted_rub_bn)
        except Exception:
            pass
    # Restore
    if PRICES:
        for k, v in PRICES.items():
            if hasattr(v, "avg_price_usd"):
                v.avg_price_usd = original_prices[k]
            elif hasattr(v, "avg_price_usd_per_t"):
                v.avg_price_usd_per_t = original_prices[k]
    if FX is not None and original_fx and hasattr(FX, "avg_usd_rub"):
        FX.avg_usd_rub = original_fx

    return _finalize_interval(company, base_pred, predictions, actual, conf)


def make_interval_retail(company: str, n_sim: int = 200, conf: float = 0.90) -> PredictionInterval:
    """Conformal для osl_retail. Perturbation на take_rate (для marketplaces) и GMV/revenue."""
    import osl_retail as m

    actual = m.ACTUAL_REVENUE_2025.get(company, {}).get("rub_bn")
    base_pred = m.predict_revenue(company).predicted_rub_bn

    profile = m.PROFILES[company]
    original_take_rate = profile.take_rate

    # Find signals (volumes)
    signals_map = {
        "Wildberries": m.WILDBERRIES_SIGNALS,
        "Ozon": m.OZON_SIGNALS,
        "Melon Fashion": m.MELON_SIGNALS,
        "Золотое яблоко": m.GOLDEN_APPLE_SIGNALS,
        "Лемана Про": m.LEROY_SIGNALS,
        "М.Видео": m.MVIDEO_SIGNALS,
    }
    signals = signals_map.get(company, [])
    original_values = [s.value for s in signals]

    rng = np.random.default_rng(42)
    predictions = []
    for _ in range(n_sim):
        profile.take_rate = original_take_rate * (1 + rng.normal(0, 0.10))  # take_rate ±10%
        for s, ov in zip(signals, original_values):
            s.value = ov * (1 + rng.normal(0, 0.05))  # GMV/revenue ±5%
        try:
            pred = m.predict_revenue(company).predicted_rub_bn
            if pred and not np.isnan(pred):
                predictions.append(pred)
        except Exception:
            pass

    profile.take_rate = original_take_rate
    for s, ov in zip(signals, original_values):
        s.value = ov

    return _finalize_interval(company, base_pred, predictions, actual, conf)


def make_interval_energy(company: str, n_sim: int = 200, conf: float = 0.90) -> PredictionInterval:
    """Conformal для osl_energy (v0.7). Perturbation на tariff_multiplier
    + other_revenue_abs_rub_bn + tariff/capacity rates + production."""
    import osl_energy as m

    actual = m.ACTUAL_REVENUE_2025.get(company, {}).get("rub_bn")
    base_pred = m.predict_revenue(company).predicted_rub_bn

    orig_tariff = m.TARIFFS_2025.avg_tariff_rub_per_mwh
    orig_capacity = m.TARIFFS_2025.capacity_payment_per_gw_year

    gen_map = {
        "Интер РАО": m.INTER_RAO,
        "РусГидро": m.RUSHYDRO,
        "Юнипро": m.UNIPRO,
        "Т Плюс": m.T_PLUS,
        "Росатом-Энергоатом": m.ROSATOM_NUCLEAR,
    }
    gen = gen_map.get(company)
    orig_gen_twh = gen.generation_twh if gen else 0
    orig_cap_gw = gen.capacity_gw if gen else 0

    profile = m.PROFILES[company]
    orig_mult = profile.tariff_multiplier
    orig_other_abs = profile.other_revenue_abs_rub_bn

    rng = np.random.default_rng(42)
    predictions = []
    for _ in range(n_sim):
        m.TARIFFS_2025.avg_tariff_rub_per_mwh = orig_tariff * (1 + rng.normal(0, 0.05))
        m.TARIFFS_2025.capacity_payment_per_gw_year = orig_capacity * (1 + rng.normal(0, 0.05))
        if gen:
            gen.generation_twh = orig_gen_twh * (1 + rng.normal(0, 0.03))
            gen.capacity_gw = orig_cap_gw * (1 + rng.normal(0, 0.02))
        profile.tariff_multiplier = orig_mult * (1 + rng.normal(0, 0.05))
        # other_revenue_abs может быть 0 — пертурбация в абсолютных рублях относительно base
        # для компаний с other_abs=0 даём малую пертурбацию ±20 млрд (чтобы было какое-то rev_other)
        if orig_other_abs > 0:
            profile.other_revenue_abs_rub_bn = max(0.0, orig_other_abs * (1 + rng.normal(0, 0.10)))
        else:
            profile.other_revenue_abs_rub_bn = max(0.0, rng.normal(0, 20))
        try:
            pred = m.predict_revenue(company).predicted_rub_bn
            if pred and not np.isnan(pred):
                predictions.append(pred)
        except Exception:
            pass

    m.TARIFFS_2025.avg_tariff_rub_per_mwh = orig_tariff
    m.TARIFFS_2025.capacity_payment_per_gw_year = orig_capacity
    if gen:
        gen.generation_twh = orig_gen_twh
        gen.capacity_gw = orig_cap_gw
    profile.tariff_multiplier = orig_mult
    profile.other_revenue_abs_rub_bn = orig_other_abs

    return _finalize_interval(company, base_pred, predictions, actual, conf)


def make_interval_oiv(region: str, n_sim: int = 200, conf: float = 0.90) -> PredictionInterval:
    """Conformal для osl_oiv. Perturbation на налоговые константы и production."""
    import osl_oiv as m

    actual = m.ACTUAL_BUDGET_2025.get(region, {}).get("rub_bn")
    base_pred = m.predict_revenue(region).predicted_rub_bn

    orig_oil_tax = m.OIL_PROFIT_TAX_PER_TON_RUB
    orig_gas_tax = m.GAS_PROFIT_TAX_PER_BCM_RUB
    profile = m.PROFILES[region]
    orig_oil_prod = profile.oil_production_mt
    orig_gas_prod = profile.gas_production_bcm
    orig_fed_share = profile.federal_transfer_share

    rng = np.random.default_rng(42)
    predictions = []
    for _ in range(n_sim):
        m.OIL_PROFIT_TAX_PER_TON_RUB = orig_oil_tax * (1 + rng.normal(0, 0.10))  # tax rates ±10%
        m.GAS_PROFIT_TAX_PER_BCM_RUB = orig_gas_tax * (1 + rng.normal(0, 0.10))
        profile.oil_production_mt = orig_oil_prod * (1 + rng.normal(0, 0.05))  # production ±5%
        profile.gas_production_bcm = orig_gas_prod * (1 + rng.normal(0, 0.05))
        profile.federal_transfer_share = max(
            0.0, min(0.5, orig_fed_share * (1 + rng.normal(0, 0.30)))
        )
        try:
            pred = m.predict_revenue(region).predicted_rub_bn
            if pred and not np.isnan(pred):
                predictions.append(pred)
        except Exception:
            pass

    # Restore
    m.OIL_PROFIT_TAX_PER_TON_RUB = orig_oil_tax
    m.GAS_PROFIT_TAX_PER_BCM_RUB = orig_gas_tax
    profile.oil_production_mt = orig_oil_prod
    profile.gas_production_bcm = orig_gas_prod
    profile.federal_transfer_share = orig_fed_share

    return _finalize_interval(region, base_pred, predictions, actual, conf)


def make_interval_pharma(company: str, n_sim: int = 200, conf: float = 0.90) -> PredictionInterval:
    """Conformal для osl_pharma. Perturbation на market_share + размер рынка."""
    import osl_pharma as m

    actual = m.ACTUAL_REVENUE_2025.get(company, {}).get("rub_bn")
    base_pred = m.predict_revenue(company).predicted_rub_bn

    profile = m.PROFILES[company]
    orig = {
        "market_share_total": profile.market_share_total,
        "market_share_retail": profile.market_share_retail,
        "market_share_gov": profile.market_share_gov,
    }
    market_orig = dict(m.PHARMA_MARKET_2025)

    rng = np.random.default_rng(42)
    predictions = []
    for _ in range(n_sim):
        # Market shares ±5% relative
        profile.market_share_total = orig["market_share_total"] * (1 + rng.normal(0, 0.05))
        profile.market_share_retail = orig["market_share_retail"] * (1 + rng.normal(0, 0.05))
        profile.market_share_gov = orig["market_share_gov"] * (1 + rng.normal(0, 0.05))
        # Рынок ±3%
        for k in ["total_rub_bn", "commercial_retail_rub_bn", "gov_segment_rub_bn"]:
            if k in m.PHARMA_MARKET_2025:
                m.PHARMA_MARKET_2025[k] = market_orig[k] * (1 + rng.normal(0, 0.03))
        try:
            pred = m.predict_revenue(company).predicted_rub_bn
            if pred and not np.isnan(pred):
                predictions.append(pred)
        except Exception:
            pass

    # Restore
    for k, v in orig.items():
        setattr(profile, k, v)
    for k, v in market_orig.items():
        m.PHARMA_MARKET_2025[k] = v

    return _finalize_interval(company, base_pred, predictions, actual, conf)


def main():
    parser = argparse.ArgumentParser(description="OSL v0.4 — Conformal Prediction All Industries")
    parser.add_argument("--n-sim", type=int, default=200)
    parser.add_argument("--confidence", type=float, default=0.90)
    parser.add_argument(
        "--industry",
        default="all",
        choices=["metallurgy", "oilgas", "chemistry", "retail", "energy", "pharma", "oiv", "all"],
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  OSL v0.4 — Conformal Prediction (all industries)")
    print(f"  N simulations: {args.n_sim}, confidence: {args.confidence}")
    print("=" * 70)

    INDUSTRY_COMPANIES = {
        "metallurgy": ("osl_metallurgy", ["Норникель", "Северсталь", "ММК", "НЛМК", "Полюс"]),
        "oilgas": ("osl_oilgas", ["Роснефть", "ЛУКОЙЛ", "Газпром", "Новатэк"]),
        "chemistry": ("osl_chemistry", ["ФосАгро", "Акрон", "СИБУР"]),
        "retail": ("osl_retail", ["Wildberries", "Ozon", "М.Видео"]),
        "energy": (
            "osl_energy",
            ["Интер РАО", "РусГидро", "Юнипро", "Т Плюс", "Росатом-Энергоатом"],
        ),
        "pharma": ("osl_pharma", ["Пульс", "Протек", "Катрен"]),
        "oiv": (
            "osl_oiv",
            ["ХМАО-Югра", "Тюменская обл.", "ЯНАО", "Татарстан", "Сахалинская обл."],
        ),
    }

    targets = list(INDUSTRY_COMPANIES.keys()) if args.industry == "all" else [args.industry]
    all_results = []

    for industry in targets:
        module_name, companies = INDUSTRY_COMPANIES[industry]
        print(f"\n{'─' * 70}")
        print(f"  {industry.upper()} ({module_name})")
        print(f"{'─' * 70}")

        for c in companies:
            try:
                if industry == "metallurgy":
                    result = make_interval_from_metallurgy(c, args.n_sim, args.confidence)
                elif industry == "retail":
                    result = make_interval_retail(c, args.n_sim, args.confidence)
                elif industry == "pharma":
                    result = make_interval_pharma(c, args.n_sim, args.confidence)
                elif industry == "energy":
                    result = make_interval_energy(c, args.n_sim, args.confidence)
                elif industry == "oiv":
                    result = make_interval_oiv(c, args.n_sim, args.confidence)
                else:
                    result = make_interval_generic(c, module_name, args.n_sim, args.confidence)
                all_results.append((industry, result))

                actual_str = f"факт={result.actual:,.0f}" if result.actual else "факт=н/д"
                mark = (
                    "✅"
                    if result.actual_in_interval
                    else ("❌" if result.coverage_metric in ("BELOW", "ABOVE") else "—")
                )
                print(
                    f"  {c:25s} | base={result.predicted_base:>7,.0f} | "
                    f"[{result.predicted_low:>7,.0f}; {result.predicted_high:>7,.0f}] | "
                    f"±{result.interval_width_pct/2:>4.1f}% | {actual_str} | {result.coverage_metric or 'н/д':<6} {mark}"
                )
            except Exception as e:
                print(f"  {c}: ERROR {e}")

    # Summary
    valid = [(ind, r) for ind, r in all_results if r.actual is not None]

    # Считаем только эмитентов с **non-zero interval** — значит perturbation действительно работает
    perturbed = [(ind, r) for ind, r in valid if r.interval_width_pct > 0.5]
    inside_perturbed = [(ind, r) for ind, r in perturbed if r.actual_in_interval]
    perturbed_rate = len(inside_perturbed) / len(perturbed) if perturbed else 0

    print(f"\n{'=' * 70}")
    print("  ИТОГИ CONFORMAL PREDICTION (все отрасли)")
    print(f"{'=' * 70}")
    print(f"  Эмитентов с actual: {len(valid)}")
    print(f"  Эмитентов с РАБОТАЮЩИМ perturbation (width>0.5%): {len(perturbed)}")
    print(
        f"  ✅ INSIDE 90% interval (на perturbed): {len(inside_perturbed)} ({perturbed_rate*100:.0f}%)"
    )
    print(f"  Цель покрытия: {args.confidence*100:.0f}%")
    print()
    print("  ⚠️ Эмитенты с width=0% — Conformal wrapper НЕ работает")
    print("     (модули energy/pharma/oiv/retail используют параметры вне PRICES_12M_2025)")
    print("     → требуют индивидуальной адаптации в v0.5")
    print()

    by_industry = {}
    for ind, r in valid:
        by_industry.setdefault(ind, {"total": 0, "inside": 0, "perturbed": 0})
        by_industry[ind]["total"] += 1
        if r.interval_width_pct > 0.5:
            by_industry[ind]["perturbed"] += 1
        if r.actual_in_interval and r.interval_width_pct > 0.5:
            by_industry[ind]["inside"] += 1

    print("  Покрытие по отраслям (только эмитенты с работающим perturbation):")
    for ind, stats in by_industry.items():
        if stats["perturbed"] == 0:
            print(f"    {ind:15s}: 0/0 perturbation NOT working — требует v0.5 рефакторинга")
        else:
            rate = stats["inside"] / stats["perturbed"] * 100
            print(
                f"    {ind:15s}: {stats['inside']}/{stats['perturbed']} (perturbed) = {rate:.0f}% покрытие"
            )


if __name__ == "__main__":
    main()
