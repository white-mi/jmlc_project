"""Smoke-тесты DS-ablation (osl_ablation): сетка hist_gbm + coef/importance диагностика."""

import osl_ablation as A


def test_sweep_returns_full_grid():
    rows = A._load("metallurgy")
    sweep = A.sweep_gbm(rows)
    assert len(sweep) == len(A.DEPTHS) * len(A.L2S) == 9
    assert all(v is None or v > 0 for v in sweep.values())
    # база (d2/l2=1.0) должна совпасть с прод-hist_gbm (GBMPanel() байт-идентичен)
    assert sweep.get("d2_l21.0") is not None


def test_elasticnet_coefs_nonempty():
    rows = A._load("metallurgy")
    coefs = A.elasticnet_coefs(rows)
    assert len(coefs) >= 1
    assert all(isinstance(name, str) and isinstance(c, float) for name, c in coefs)


def test_gbm_importance_nonempty():
    rows = A._load("metallurgy")
    imp = A.gbm_importance(rows, n_repeats=3)
    # важность по всем признакам (цены + объёмы)
    assert len(imp) >= 1
    assert all(isinstance(name, str) and isinstance(v, float) for name, v in imp)


def test_run_and_render_markdown():
    res = A.run("metallurgy")
    assert res["sweep"] and res["elasticnet_coefs"] and res["gbm_importance"]
    md = A.render_md(res)
    assert "Ablation" in md
    assert "permutation importance" in md
    assert "max_depth" in md
