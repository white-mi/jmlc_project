"""
Split-conformal интервалы на остатках калибровочного фолда → ЧЕСТНОЕ out-of-sample
покрытие (Stage E). В отличие от perturbation-интервалов в conformal_prediction.py
(они IN-SAMPLE — острота, а не обобщение), здесь остатки берутся из периода,
НЕ пересекающегося ни с train, ни с test, а test строго в будущем.

Метод (inductive/split conformal, Vovk):
  1. proper-train: обучаем модель (osl_models) на ранних годах;
  2. calibration: |y−ŷ|/y по отдельному году → ОТНОСИТЕЛЬНЫЕ остатки
     (относительные, т.к. таргеты смешанной валюты USD/RUB — относит. ошибка
     обменивается между эмитентами; иначе квантиль захватил бы только RUB-масштаб);
  3. q = конформный квантиль уровня ceil((n+1)(1−α))/n (finite-sample корректный);
  4. интервал test: [ŷ·(1−q), ŷ·(1+q)] (мультипликативный — под относит. остатки);
  5. покрытие = доля test-actual внутри интервала. При обмениваемости остатков
     маржинальное покрытие ≥ 1−α (на малом calib — шумно, но метод корректен).

Возвращает PredictionInterval из conformal_prediction.py (единая форма для pipeline/тестов).

CLI:
  python conformal_split.py --industry metallurgy
"""

import argparse
import sys
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).parent
sys.path.insert(0, str(TOOLS))
from conformal_prediction import PredictionInterval  # reuse
import osl_panel  # noqa: E402
import osl_models as Mo  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def relative_residuals(model, calib_rows) -> np.ndarray:
    """|y − ŷ| / y по калибровочным строкам (только где прогноз конечен).

    Деление на y (факт), не на ŷ — интервал [ŷ(1−q), ŷ(1+q)] инвертирует |y−ŷ|/ŷ≤q,
    что эквивалентно с точностью до 1-го порядка при таких остатках (стандартно).
    Допущение: остатки ОБМЕНИВАЕМЫ между эмитентами (пулим 5 эмитентов в один квантиль).
    Это приближение — у gas-vink Норникеля и золотого Полюса разная природа ошибки;
    относительная нормировка делает их сопоставимыми, но строгой обмениваемости нет."""
    pred = model.predict(calib_rows)
    out = []
    for r, p in zip(calib_rows, pred):
        if np.isfinite(p) and r.target_bn:
            out.append(abs(p - r.target_bn) / r.target_bn)
    return np.array(out, dtype=float)


def conformal_quantile(residuals: np.ndarray, alpha: float = 0.10):
    """Finite-sample конформный квантиль уровня ceil((n+1)(1−α))/n.
    None, если калибровочных остатков нет."""
    n = len(residuals)
    if n == 0:
        return None
    level = np.ceil((n + 1) * (1 - alpha)) / n
    if level >= 1.0:
        return float(np.max(residuals))  # уровень >1 на малом n → самый широкий остаток
    return float(np.quantile(residuals, level, method="higher"))


def split_conformal(model_ctor, proper_train, calib, test, alpha: float = 0.10):
    """3-факторный temporal split. Возвращает dict с q, intervals (PredictionInterval),
    coverage_rate, n_calib, n_test."""
    model = model_ctor().fit(proper_train)
    resid = relative_residuals(model, calib)
    q = conformal_quantile(resid, alpha)
    preds = model.predict(test)
    intervals, inside, total = [], 0, 0
    for r, p in zip(test, preds):
        if not np.isfinite(p) or q is None:
            intervals.append(None)
            continue
        low, high = p * (1 - q), p * (1 + q)
        cov = None
        if r.target_bn is not None:
            cov = bool(low <= r.target_bn <= high)
            total += 1
            inside += int(cov)
        metric = None
        if cov is not None:
            metric = "INSIDE" if cov else ("BELOW" if r.target_bn < low else "ABOVE")
        intervals.append(
            PredictionInterval(
                f"{r.issuer} {r.period}",
                float(p),
                float(low),
                float(high),
                q * 200.0,
                r.target_bn,
                cov,
                metric,
            )
        )
    return {
        "q": q,
        "intervals": intervals,
        "coverage_rate": (inside / total if total else None),
        "n_calib": int(len(resid)),
        "n_test": int(total),
        "inside": int(inside),
    }


def temporal_holdout(rows, model_ctor, alpha=0.10, train_max=2022, calib_year=2023):
    """Удобная обёртка: proper-train ≤ train_max, calib = calib_year, test = годы > calib_year."""
    rows = [r for r in rows if r.has_target and r.period_end]
    proper = [r for r in rows if r.period_end.year <= train_max]
    calib = [r for r in rows if r.period_end.year == calib_year]
    test = [r for r in rows if r.period_end.year > calib_year]
    return split_conformal(model_ctor, proper, calib, test, alpha)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--industry", default="metallurgy")
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--model", default="structural_osl", choices=list(Mo.MODELS))
    args = ap.parse_args()
    rows = osl_panel.load_panel(industry=args.industry)
    res = temporal_holdout(rows, Mo.MODELS[args.model], alpha=args.alpha)
    print("=" * 64)
    print(f"  SPLIT-CONFORMAL (out-of-sample) — {args.industry} / {args.model}")
    print("=" * 64)
    print(f"  α={args.alpha} → цель покрытия {100*(1-args.alpha):.0f}%")
    print(f'  proper-train ≤2022 | calib 2023 (n={res["n_calib"]}) | test >2023')
    q_str = f'{res["q"]:.3f}' if res["q"] is not None else "—"
    print(f"  q (относит. полуширина) = {q_str}")
    if res["coverage_rate"] is not None:
        print(f'  Покрытие OOS: {res["inside"]}/{res["n_test"]} = {100*res["coverage_rate"]:.0f}%')
    else:
        print("  нет test-строк")
    print()
    for iv in res["intervals"]:
        if iv is None:
            continue
        mark = {"INSIDE": "✓", "BELOW": "↓", "ABOVE": "↑"}.get(iv.coverage_metric, "?")
        print(
            f"  {mark} {iv.company:18s} ŷ={iv.predicted_base:8.1f} "
            f"[{iv.predicted_low:7.1f}; {iv.predicted_high:7.1f}] факт={iv.actual}"
        )


if __name__ == "__main__":
    main()
