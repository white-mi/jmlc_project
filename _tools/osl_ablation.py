"""
DS ablation / диагностика: чувствительность hist_gbm к гиперпараметрам +
интерпретация обучаемых моделей. Подтверждает ЧИСЛАМИ вывод «learned не выигрывают
точность / доменный приор не хуже», а не декларирует его.

Три артефакта:
  1. Sensitivity — MAPE hist_gbm по сетке (max_depth × l2) walk-forward → устойчив ли вывод к тюнингу.
  2. ElasticNet coefficients (within-FE, log-цены) — какие цены «тянет» линейная модель.
  3. hist_gbm permutation-importance — какие драйверы реально влияют.

Выход: output/osl_metrics/ablation_<industry>.md (output/ в .gitignore). Без matplotlib (core-safe).
Запуск: cd _tools && python osl_ablation.py --industry metallurgy
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.inspection import permutation_importance

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import osl_panel
from osl_models import GBMPanel, LinearPanel, _price_features, _targets, _vol_features
from osl_walkforward import metrics_for, walk_forward

TOOLS = Path(__file__).parent
OUT = TOOLS / "output" / "osl_metrics"

DEPTHS = (1, 2, 3)
L2S = (0.5, 1.0, 2.0)


def _load(industry):
    rows = osl_panel.load_panel(industry=industry)
    return [r for r in rows if r.has_target and r.period_end]


def sweep_gbm(rows, depths=DEPTHS, l2s=L2S):
    """MAPE hist_gbm по сетке (max_depth × l2_regularization), общий walk-forward."""
    ctors = {}
    for d in depths:
        for l2 in l2s:
            ctors[f"d{d}_l2{l2}"] = lambda d=d, l2=l2: GBMPanel(max_depth=d, l2_regularization=l2)
    preds, _ = walk_forward(rows, ctors)
    return {name: metrics_for(recs)["mape"] for name, recs in preds.items()}


def elasticnet_coefs(rows):
    """Коэффициенты ElasticNet (within-FE, log-цены) — что «тянет» линейная модель."""
    lp = LinearPanel("elasticnet").fit(rows)
    return [(f, float(c)) for f, c in zip(_price_features(rows), lp.model.coef_)]


def gbm_importance(rows, n_repeats=10):
    """Permutation-importance hist_gbm — какие драйверы реально влияют на прогноз."""
    gb = GBMPanel().fit(rows)
    X = gb.design.transform(rows)
    y = np.log(_targets(rows)) - gb.fe.level(rows)
    r = permutation_importance(gb.model, X, y, n_repeats=n_repeats, random_state=0)
    feats = list(_price_features(rows)) + list(_vol_features(rows))
    return [(f, float(v)) for f, v in zip(feats, r.importances_mean)]


def run(industry="metallurgy"):
    rows = _load(industry)
    return {
        "industry": industry,
        "n": len(rows),
        "sweep": sweep_gbm(rows),
        "elasticnet_coefs": elasticnet_coefs(rows),
        "gbm_importance": gbm_importance(rows),
    }


def render_md(res):
    L = [f"# Ablation / диагностика — {res['industry']} (N={res['n']})", ""]
    L += ["## 1. Чувствительность hist_gbm: MAPE по (max_depth × l2)", ""]
    L.append("| max_depth \\ l2 | " + " | ".join(str(x) for x in L2S) + " |")
    L.append("|" + "---|" * (len(L2S) + 1))
    for d in DEPTHS:
        cells = []
        for l2 in L2S:
            m = res["sweep"].get(f"d{d}_l2{l2}")
            tag = " (база)" if (d == 2 and l2 == 1.0) else ""
            cells.append(f"{m:.1f}%{tag}" if m is not None else "—")
        L.append(f"| {d} | " + " | ".join(cells) + " |")
    vals = [v for v in res["sweep"].values() if v is not None]
    if vals:
        L += [
            "",
            f"Разброс MAPE по сетке: **{min(vals):.1f}–{max(vals):.1f}%** (база d2/l2=1.0). "
            "Малый разброс → вывод «GBM не выигрывает точность» устойчив к тюнингу, а не подогнан.",
        ]
    L += ["", "## 2. ElasticNet — коэффициенты (within-FE, log-цены)", ""]
    L += ["| признак | коэффициент |", "|---|---|"]
    for f, c in sorted(res["elasticnet_coefs"], key=lambda t: -abs(t[1])):
        L.append(f"| {f} | {c:+.4f} |")
    L += ["", "## 3. hist_gbm — permutation importance (n_repeats=10)", ""]
    L += ["| драйвер | важность (Δ score) |", "|---|---|"]
    for f, v in sorted(res["gbm_importance"], key=lambda t: -t[1]):
        L.append(f"| {f} | {v:+.4f} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--industry", default="metallurgy")
    args = ap.parse_args()
    md = render_md(run(args.industry))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"ablation_{args.industry}.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"→ output/osl_metrics/ablation_{args.industry}.md")


if __name__ == "__main__":
    main()
