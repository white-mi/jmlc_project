"""
OSL Calibrator v0.6 — auto-tune параметров OSL-моделей.

Цель: обеспечить **бесшовную регулярную калибровку** моделей по мере появления
новых actuals (новый квартал, новые цены). Без этого модели «дрейфуют»
и калибровка превращается в ручной hack каждые 3 месяца.

Архитектура:
1. tune_param(predict_fn, target_actual, param_setter, search_range)
   - scipy.optimize.minimize_scalar (Brent's method)
   - Минимизирует |predicted - actual| / actual
   - Возвращает оптимальный параметр + достигнутый MAE

2. calibrate_module(module_name)
   - Прогонит все эмитенты модуля через auto-tune
   - Сохранит лучшие параметры в _tools/calibration/<module>_calibrated.json
   - При следующем запуске OSL — load_calibrated() применит автоматически

3. drift_check()
   - Сравнивает текущий MAE с baseline (saved при последней калибровке)
   - Если MAE вырос >5 п.п. — флаг "calibration needed"
"""

import json
import sys
from pathlib import Path
from typing import Callable, Dict
import importlib

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

CALIBRATION_DIR = Path(__file__).parent / "calibration"
CALIBRATION_DIR.mkdir(exist_ok=True)


def tune_param(
    predict_fn: Callable[[float], float],
    target: float,
    param_min: float,
    param_max: float,
    n_steps: int = 30,
) -> tuple[float, float]:
    """
    Grid search + binary refine.
    Возвращает (best_param, best_mae_pct).
    """
    # Grid search
    step = (param_max - param_min) / n_steps
    best_param = param_min
    best_err = float('inf')
    for i in range(n_steps + 1):
        p = param_min + i * step
        try:
            pred = predict_fn(p)
            if pred is None or pred <= 0:
                continue
            err = abs(pred - target) / target
            if err < best_err:
                best_err = err
                best_param = p
        except Exception:
            continue

    # Refine: binary search вокруг best
    for _ in range(8):
        candidates = [best_param * 0.97, best_param * 1.03]
        for c in candidates:
            try:
                pred = predict_fn(c)
                if pred is None or pred <= 0:
                    continue
                err = abs(pred - target) / target
                if err < best_err:
                    best_err = err
                    best_param = c
            except Exception:
                continue

    return best_param, best_err * 100


def tune_multi_period(
    predict_fn: Callable[[float], float],
    targets: Dict[str, float],
    period_shares: Dict[str, float],
    param_min: float,
    param_max: float,
    n_steps: int = 30,
) -> tuple[float, dict]:
    """
    Multi-period калибровка: минимизирует среднюю относительную ошибку
    по нескольким временным периодам. Защищает от overfitting к одной точке.

    Аргументы:
        predict_fn: возвращает прогноз 12M-выручки при заданном параметре
        targets: {'12M': actual_12M, '9M': actual_9M, ...}
        period_shares: {'12M': 1.0, '9M': 0.75, ...} — доля периода от 12M
        param_min/max: границы поиска
        n_steps: шагов grid search

    Возвращает (best_param, metrics_dict с MAE по каждому периоду + средний MAE).
    """
    def aggregate_loss(p):
        try:
            pred_12m = predict_fn(p)
            if pred_12m is None or pred_12m <= 0:
                return float('inf')
            total_loss = 0.0
            count = 0
            for period, target in targets.items():
                share = period_shares.get(period, 1.0)
                pred_for_period = pred_12m * share
                err = abs(pred_for_period - target) / target
                total_loss += err
                count += 1
            return total_loss / count if count > 0 else float('inf')
        except Exception:
            return float('inf')

    step = (param_max - param_min) / n_steps
    best_param = param_min
    best_err = float('inf')
    for i in range(n_steps + 1):
        p = param_min + i * step
        err = aggregate_loss(p)
        if err < best_err:
            best_err = err
            best_param = p

    for _ in range(8):
        for c in [best_param * 0.97, best_param * 1.03]:
            err = aggregate_loss(c)
            if err < best_err:
                best_err = err
                best_param = c

    # Считаем MAE по каждому периоду отдельно для отчёта
    pred_12m = predict_fn(best_param)
    metrics = {'best_avg_mae_pct': round(best_err * 100, 2)}
    for period, target in targets.items():
        share = period_shares.get(period, 1.0)
        pred_for_period = pred_12m * share
        metrics[f'mae_{period}_pct'] = round(abs(pred_for_period - target) / target * 100, 2)

    return best_param, metrics


def tune_multi_param(
    predict_fn: Callable[[tuple], float],
    target: float,
    bounds: list,
    seed: int = 42,
    maxiter: int = 100,
) -> tuple[tuple, float]:
    """
    Multi-parameter calibration через scipy.differential_evolution.

    Аргументы:
        predict_fn: принимает tuple параметров, возвращает predicted value
        target: целевое actual value
        bounds: [(min, max), (min, max), ...] для N параметров
        seed: для воспроизводимости
        maxiter: max итераций differential_evolution

    Возвращает (best_params: tuple, mae_pct: float).

    NB: со 2+ параметрами на 1 точку данных получается near-perfect fit,
    что НЕ гарантирует устойчивость. Используется в паре с tune_multi_period
    или когда есть несколько лет actual'ов.
    """
    try:
        from scipy.optimize import differential_evolution
    except ImportError:
        raise ImportError("scipy not installed. Run: pip install scipy")

    def loss(params):
        try:
            pred = predict_fn(tuple(params))
            if pred is None or pred <= 0:
                return 1.0
            return abs(pred - target) / target
        except Exception:
            return 1.0

    result = differential_evolution(
        loss, bounds, seed=seed, tol=1e-6, maxiter=maxiter,
        polish=True, init='sobol',
    )
    return tuple(result.x), result.fun * 100


def save_calibration(module_name: str, calibration: dict):
    """Сохранить калиброванные параметры в JSON."""
    file = CALIBRATION_DIR / f"{module_name}_calibrated.json"
    file.write_text(json.dumps(calibration, indent=2, ensure_ascii=False), encoding="utf-8")


def load_calibration(module_name: str) -> dict:
    """Загрузить калиброванные параметры."""
    file = CALIBRATION_DIR / f"{module_name}_calibrated.json"
    if not file.exists():
        return {}
    return json.loads(file.read_text(encoding="utf-8"))


def apply_calibration(module_name: str, verbose: bool = False) -> int:
    """
    Применить сохранённые калибровки к загруженному модулю.
    Изменяет profile.<param> для каждого эмитента из JSON.

    Использование (бесшовно):
        from osl_calibrator import apply_calibration
        apply_calibration('osl_metallurgy')
        # теперь predict_revenue использует калиброванные параметры

    Возвращает: количество применённых калибровок.
    """
    cal = load_calibration(module_name)
    if not cal:
        return 0

    module = importlib.import_module(module_name)
    profiles = getattr(module, 'PROFILES', {})

    applied = 0
    for company, params in cal.items():
        if company not in profiles:
            continue
        profile = profiles[company]
        for param_name, param_val in params.items():
            if param_name in ('mae_pct', 'actual_rub_bn'):
                continue
            if hasattr(profile, param_name):
                setattr(profile, param_name, param_val)
                applied += 1
                if verbose:
                    print(f"  Applied {module_name}.{company}.{param_name} = {param_val}")
    return applied


def apply_all_calibrations(verbose: bool = False) -> dict:
    """Применить калибровки ко всем 7 OSL-модулям."""
    modules = ['osl_metallurgy', 'osl_oilgas', 'osl_chemistry',
                'osl_pharma', 'osl_retail', 'osl_energy', 'osl_oiv']
    summary = {}
    for m in modules:
        try:
            n = apply_calibration(m, verbose)
            summary[m] = n
        except Exception as e:
            summary[m] = f"error: {e}"
    return summary


def drift_check(module_name: str, threshold_pct: float = 5.0) -> dict:
    """
    Сравнить текущий MAE с baseline (последней калибровки).

    Возвращает:
    {
      'company': {
         'baseline_mae': 5.2,
         'current_mae': 12.5,
         'drift': +7.3,
         'flag': 'NEEDS_RECALIBRATION'
      },
      ...
    }
    """
    baseline = load_calibration(module_name)
    if not baseline:
        return {'error': f'No baseline calibration for {module_name}'}

    module = importlib.import_module(module_name)
    actual_dict = (getattr(module, 'ACTUAL_REVENUE_2025', None) or
                    getattr(module, 'ACTUAL_REVENUE_12M_2025', None) or
                    getattr(module, 'ACTUAL_BUDGET_2025', None))
    if not actual_dict:
        return {'error': f'No actuals in {module_name}'}

    drift = {}
    for company, data in actual_dict.items():
        actual = data.get('rub_bn')
        if not actual:
            continue
        try:
            pred = module.predict_revenue(company).predicted_rub_bn
            current_mae = abs(pred - actual) / actual * 100
            baseline_mae = baseline.get(company, {}).get('mae_pct', current_mae)
            d = current_mae - baseline_mae
            flag = 'OK' if d <= threshold_pct else 'NEEDS_RECALIBRATION'
            drift[company] = {
                'baseline_mae': baseline_mae,
                'current_mae': current_mae,
                'drift_pct': d,
                'flag': flag,
            }
        except Exception as e:
            drift[company] = {'error': str(e)}
    return drift


def calibrate_energy() -> dict:
    """Калибровка энергетики (v0.7): tune profile.tariff_multiplier (диапазон 0.5-2.0).
    Other_revenue_abs (тепло/сбыт/прочее) задано в PROFILES из IR.
    Tariff_multiplier учитывает региональную/сегментную премию или дисконт."""
    import osl_energy as m
    results = {}
    for company in ['Интер РАО', 'РусГидро', 'Юнипро', 'Т Плюс', 'Росатом-Энергоатом']:
        actual = m.ACTUAL_REVENUE_2025.get(company, {}).get('rub_bn')
        if not actual:
            continue
        profile = m.PROFILES[company]

        def predict_with_mult(p):
            profile.tariff_multiplier = max(0.5, min(2.0, p))
            return m.predict_revenue(company).predicted_rub_bn

        best_mult, best_mae = tune_param(predict_with_mult, actual, 0.5, 2.0)
        profile.tariff_multiplier = best_mult
        results[company] = {
            'tariff_multiplier': round(best_mult, 4),
            'mae_pct': round(best_mae, 2),
            'actual_rub_bn': actual,
        }
    save_calibration('osl_energy', results)
    return results


def calibrate_oilgas() -> dict:
    """Калибровка нефтегаза: tune profile.other_share."""
    import osl_oilgas as m
    results = {}
    for company in ['Роснефть', 'ЛУКОЙЛ', 'Газпром', 'Новатэк']:
        actual = m.ACTUAL_REVENUE_12M_2025.get(company, {}).get('rub_bn')
        if not actual:
            continue
        profile = m.PROFILES[company]

        def predict_with_other(p):
            profile.other_share = max(0.0, min(0.50, p))
            return m.predict_revenue(company).predicted_rub_bn

        best_other, best_mae = tune_param(predict_with_other, actual, 0.0, 0.50)
        profile.other_share = best_other
        results[company] = {
            'other_share': round(best_other, 4),
            'mae_pct': round(best_mae, 2),
            'actual_rub_bn': actual,
        }
    save_calibration('osl_oilgas', results)
    return results


def calibrate_chemistry() -> dict:
    """Калибровка химии: tune profile.other_income_pct."""
    import osl_chemistry as m
    results = {}
    for company in ['ФосАгро', 'Акрон', 'СИБУР']:
        actual = m.ACTUAL_REVENUE_2025.get(company, {}).get('rub_bn')
        if not actual:
            continue
        profile = m.PROFILES[company]

        def predict_with(p):
            profile.other_income_pct = max(0.0, min(0.40, p))
            return m.predict_revenue(company).predicted_rub_bn

        best, best_mae = tune_param(predict_with, actual, 0.0, 0.40)
        profile.other_income_pct = best
        results[company] = {'other_income_pct': round(best, 4),
                            'mae_pct': round(best_mae, 2),
                            'actual_rub_bn': actual}
    save_calibration('osl_chemistry', results)
    return results


def calibrate_pharma() -> dict:
    """Калибровка фармы: tune profile.market_share_retail для дистрибуторов."""
    import osl_pharma as m
    results = {}
    for company in ['Пульс', 'Протек', 'Катрен']:
        actual = m.ACTUAL_REVENUE_2025.get(company, {}).get('rub_bn')
        if not actual:
            continue
        profile = m.PROFILES[company]

        def predict_with(p):
            profile.market_share_retail = max(0.05, min(0.30, p))
            return m.predict_revenue(company).predicted_rub_bn

        best, best_mae = tune_param(predict_with, actual, 0.10, 0.30)
        profile.market_share_retail = best
        results[company] = {'market_share_retail': round(best, 4),
                            'mae_pct': round(best_mae, 2),
                            'actual_rub_bn': actual}
    save_calibration('osl_pharma', results)
    return results


def calibrate_retail() -> dict:
    """Калибровка розницы: tune take_rate для маркетплейсов."""
    import osl_retail as m
    results = {}
    for company in ['Wildberries', 'Ozon']:
        actual = m.ACTUAL_REVENUE_2025.get(company, {}).get('rub_bn')
        if not actual:
            continue
        profile = m.PROFILES[company]

        def predict_with(p):
            profile.take_rate = max(0.05, min(0.50, p))
            return m.predict_revenue(company).predicted_rub_bn

        best, best_mae = tune_param(predict_with, actual, 0.05, 0.50)
        profile.take_rate = best
        results[company] = {'take_rate': round(best, 4),
                            'mae_pct': round(best_mae, 2),
                            'actual_rub_bn': actual}
    save_calibration('osl_retail', results)
    return results


def calibrate_oiv() -> dict:
    """Калибровка ОИВ: tune profile.federal_transfer_share + own_revenue_share.
    Главная структурная проблема: predicted считал full budget, actual = СД.
    Решение через калибровочный коэффициент scale_to_actual."""
    import osl_oiv as m
    results = {}
    for region in ['ХМАО-Югра', 'Тюменская обл.', 'ЯНАО', 'Татарстан', 'Сахалинская обл.']:
        actual = m.ACTUAL_BUDGET_2025.get(region, {}).get('rub_bn')
        if not actual:
            continue
        profile = m.PROFILES[region]

        # Двухпараметрическая калибровка через 2 этапа
        # Параметр 1: oil_production_mt (косвенно ~ size of region's nephtyanaya rente)
        # Параметр 2: federal_transfer_share
        orig_oil = profile.oil_production_mt

        def predict_with(p):
            # Скалируем добычу — она пропорциональна доходам региона от налога
            profile.oil_production_mt = max(0, p)
            return m.predict_revenue(region).predicted_rub_bn

        best, best_mae = tune_param(predict_with, actual, 0, orig_oil * 3)
        profile.oil_production_mt = best
        results[region] = {'oil_production_mt': round(best, 1),
                            'mae_pct': round(best_mae, 2),
                            'actual_rub_bn': actual}
    save_calibration('osl_oiv', results)
    return results


def calibrate_metallurgy() -> dict:
    """Калибровка металлургии: tune profile.other_income_pct и domestic_premium_pct."""
    import osl_metallurgy as m

    results = {}
    for company in ['Норникель', 'Северсталь', 'ММК', 'НЛМК', 'Полюс']:
        actual_data = m.ACTUAL_REVENUE_12M_2025.get(company, {})
        actual = actual_data.get('rub_bn')
        if not actual and actual_data.get('usd_bn'):
            actual = actual_data['usd_bn'] * m.FX_12M_2025.avg_usd_rub
        if not actual:
            continue

        profile = m.PROFILES[company]

        # Tune other_income_pct
        def predict_with_other(p):
            profile.other_income_pct = max(0, min(0.5, p))
            return m.predict_revenue(company).predicted_rub_bn

        best_other, best_mae = tune_param(predict_with_other, actual, 0.0, 0.30)
        profile.other_income_pct = best_other

        results[company] = {
            'other_income_pct': round(best_other, 4),
            'mae_pct': round(best_mae, 2),
            'actual_rub_bn': actual,
        }

    save_calibration('osl_metallurgy', results)
    return results


def calibrate_metallurgy_multi_param() -> dict:
    """A3 пилот: multi-parameter tune для гибридных эмитентов металлургии.
    Tune (other_income_pct, domestic_share, domestic_premium_pct) одновременно
    через scipy.differential_evolution. Применяется только для hybrid model."""
    import osl_metallurgy as m
    results = {}
    hybrid_companies = ['Северсталь', 'ММК', 'НЛМК']

    for company in hybrid_companies:
        actual_data = m.ACTUAL_REVENUE_12M_2025.get(company, {})
        actual = actual_data.get('rub_bn')
        if not actual and actual_data.get('usd_bn'):
            actual = actual_data['usd_bn'] * m.FX_12M_2025.avg_usd_rub
        if not actual:
            continue

        profile = m.PROFILES[company]

        def predict_with(params):
            other_inc, dom_share, dom_prem = params
            profile.other_income_pct = max(0, min(0.30, other_inc))
            profile.domestic_share = max(0.0, min(1.0, dom_share))
            profile.domestic_premium_pct = max(0.0, min(0.60, dom_prem))
            return m.predict_revenue(company).predicted_rub_bn

        bounds = [(0.0, 0.30), (0.30, 1.0), (0.0, 0.60)]
        best_params, best_mae = tune_multi_param(predict_with, actual, bounds)

        # Apply best
        profile.other_income_pct = best_params[0]
        profile.domestic_share = best_params[1]
        profile.domestic_premium_pct = best_params[2]

        results[company] = {
            'other_income_pct': round(best_params[0], 4),
            'domestic_share': round(best_params[1], 4),
            'domestic_premium_pct': round(best_params[2], 4),
            'mae_pct': round(best_mae, 2),
            'actual_rub_bn': actual,
        }
    save_calibration('osl_metallurgy_multi_param', results)
    return results


def calibrate_metallurgy_multi_period() -> dict:
    """A2 пилот: калибровка металлургии с одновременной валидацией на 12M + 9M.
    Минимизирует среднюю MAE по обоим периодам. Защита от overfitting."""
    import osl_metallurgy as m
    if not hasattr(m, 'ACTUAL_REVENUE_9M_2025'):
        return {'error': 'No 9M actuals available'}
    results = {}
    for company in ['Норникель', 'Северсталь', 'ММК', 'НЛМК', 'Полюс']:
        a12 = m.ACTUAL_REVENUE_12M_2025.get(company, {})
        a9 = m.ACTUAL_REVENUE_9M_2025.get(company, {})
        if not a12 or not a9:
            continue

        # Выбираем валюту: предпочитаем USD если есть в 12M
        if a12.get('usd_bn'):
            target_12 = a12['usd_bn']
            target_9 = a9.get('usd_bn')
            if not target_9:
                continue
            unit = 'usd_bn'
        else:
            target_12 = a12['rub_bn']
            target_9 = a9.get('rub_bn')
            if not target_9:
                continue
            unit = 'rub_bn'

        targets = {'12M': target_12, '9M': target_9}
        period_shares = {'12M': 1.0, '9M': a9.get('period_share', 0.75)}

        profile = m.PROFILES[company]

        def predict_with(p):
            profile.other_income_pct = max(0, min(0.30, p))
            pred = m.predict_revenue(company)
            return pred.predicted_usd_bn if unit == 'usd_bn' else pred.predicted_rub_bn

        best, metrics = tune_multi_period(predict_with, targets, period_shares, 0.0, 0.30)
        profile.other_income_pct = best
        results[company] = {
            'other_income_pct': round(best, 4),
            'unit': unit,
            'targets': targets,
            **metrics,
        }
    save_calibration('osl_metallurgy_multi_period', results)
    return results


def calibrate_oilgas_multi_period() -> dict:
    """A2 пилот: калибровка нефтегаза на 12M + 9M."""
    import osl_oilgas as m
    if not hasattr(m, 'ACTUAL_REVENUE_9M_2025'):
        return {'error': 'No 9M actuals available'}
    results = {}
    for company in ['Роснефть', 'ЛУКОЙЛ', 'Газпром', 'Новатэк']:
        a12 = m.ACTUAL_REVENUE_12M_2025.get(company, {}).get('rub_bn')
        a9_data = m.ACTUAL_REVENUE_9M_2025.get(company, {})
        a9 = a9_data.get('rub_bn')
        if not a12 or not a9:
            continue

        targets = {'12M': a12, '9M': a9}
        period_shares = {'12M': 1.0, '9M': a9_data.get('period_share', 0.75)}

        profile = m.PROFILES[company]

        def predict_with(p):
            profile.other_share = max(0.0, min(0.50, p))
            return m.predict_revenue(company).predicted_rub_bn

        best, metrics = tune_multi_period(predict_with, targets, period_shares, 0.0, 0.50)
        profile.other_share = best
        results[company] = {
            'other_share': round(best, 4),
            'targets': targets,
            **metrics,
        }
    save_calibration('osl_oilgas_multi_period', results)
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--module', default='all',
                          choices=['metallurgy', 'oilgas', 'chemistry', 'pharma',
                                   'retail', 'energy', 'oiv', 'all', 'drift',
                                   'metallurgy_mp', 'oilgas_mp', 'metallurgy_multiparam'])
    args = parser.parse_args()

    print("=" * 70)
    print("  OSL Calibrator v0.6")
    print("=" * 70)

    calibrators = {
        'metallurgy': ('other_income_pct', calibrate_metallurgy),
        'oilgas': ('other_share', calibrate_oilgas),
        'chemistry': ('other_income_pct', calibrate_chemistry),
        'pharma': ('market_share_retail', calibrate_pharma),
        'retail': ('take_rate', calibrate_retail),
        'energy': ('tariff_multiplier', calibrate_energy),
        'oiv': ('oil_production_mt', calibrate_oiv),
    }

    targets = list(calibrators.keys()) if args.module == 'all' else [args.module] if args.module in calibrators else []

    for ind in targets:
        param_name, fn = calibrators[ind]
        print(f"\nКалибровка {ind} (tune {param_name})...")
        res = fn()
        for c, r in res.items():
            val = r.get(param_name, '?')
            print(f"  {c}: {param_name}={val}, MAE={r['mae_pct']:.1f}%")
        print("  ✅ Saved")

    if args.module == 'metallurgy_multiparam':
        print("\nMulti-parameter калибровка металлургии (scipy.differential_evolution)...")
        res = calibrate_metallurgy_multi_param()
        for c, r in res.items():
            if isinstance(r, dict):
                print(f"  {c}: other_inc={r['other_income_pct']}, "
                      f"dom_share={r['domestic_share']}, "
                      f"dom_prem={r['domestic_premium_pct']}, MAE={r['mae_pct']:.2f}%")
        print("  ✅ Saved → osl_metallurgy_multi_param_calibrated.json")
        return

    if args.module == 'metallurgy_mp':
        print("\nMulti-period калибровка металлургии (12M + 9M)...")
        res = calibrate_metallurgy_multi_period()
        for c, r in res.items():
            if isinstance(r, dict):
                print(f"  {c}: other_income_pct={r.get('other_income_pct', '?')}, "
                      f"avg_MAE={r.get('best_avg_mae_pct', 0):.2f}%, "
                      f"MAE_12M={r.get('mae_12M_pct', 0):.2f}%, "
                      f"MAE_9M={r.get('mae_9M_pct', 0):.2f}%")
        print("  ✅ Saved → osl_metallurgy_multi_period_calibrated.json")
        return

    if args.module == 'oilgas_mp':
        print("\nMulti-period калибровка нефтегаза (12M + 9M)...")
        res = calibrate_oilgas_multi_period()
        for c, r in res.items():
            if isinstance(r, dict):
                print(f"  {c}: other_share={r.get('other_share', '?')}, "
                      f"avg_MAE={r.get('best_avg_mae_pct', 0):.2f}%, "
                      f"MAE_12M={r.get('mae_12M_pct', 0):.2f}%, "
                      f"MAE_9M={r.get('mae_9M_pct', 0):.2f}%")
        print("  ✅ Saved → osl_oilgas_multi_period_calibrated.json")
        return

    if args.module == 'drift':
        print("\nDrift check...")
        for mod in ['osl_metallurgy', 'osl_oilgas', 'osl_chemistry']:
            try:
                d = drift_check(mod)
                if 'error' in d:
                    print(f"  {mod}: {d['error']}")
                    continue
                print(f"\n  {mod}:")
                for company, info in d.items():
                    flag_mark = '✅' if info.get('flag') == 'OK' else '⚠️'
                    print(f"    {company}: baseline={info.get('baseline_mae', 0):.1f}%, "
                          f"current={info.get('current_mae', 0):.1f}%, "
                          f"drift={info.get('drift_pct', 0):+.1f}% {flag_mark}")
            except Exception as e:
                print(f"  {mod}: error {e}")


if __name__ == '__main__':
    main()
