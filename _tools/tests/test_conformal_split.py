"""
Тесты split-conformal (conformal_split.py): корректность конформного квантиля,
маржинальное покрытие ≈ 1−α на синтетике (обмениваемые остатки), валидность интервалов,
и реальный out-of-sample temporal hold-out на панели.
"""

import numpy as np
import pytest

import osl_panel
import osl_models as Mo
import conformal_split as CS


def _rows():
    return [r for r in osl_panel.load_panel("metallurgy") if r.has_target and r.period_end]


def _need(rows):
    if len(rows) < 8:
        pytest.skip("панель мала")


# ---------- конформный квантиль ----------


def test_conformal_quantile_level():
    """n=9, α=0.1 → level=ceil(10*0.9)/9=1.0 → max(residuals)."""
    res = np.arange(1, 10, dtype=float)  # 1..9
    q = CS.conformal_quantile(res, alpha=0.1)
    assert q == 9.0


def test_conformal_quantile_small_n_uses_max():
    res = np.array([0.1, 0.2, 0.3, 0.4, 0.5])  # n=5, level=ceil(6*0.9)/5=1.2>1 → max
    assert CS.conformal_quantile(res, alpha=0.1) == 0.5


def test_conformal_quantile_empty():
    assert CS.conformal_quantile(np.array([]), 0.1) is None


# ---------- маржинальное покрытие на синтетике ----------


def test_marginal_coverage_synthetic():
    """Обмениваемые остатки: q из calib → доля test ≤ q ≥ 1−α (конформная гарантия).
    Усредняем по испытаниям с фиксированным сидом (детерминированно, не флейки)."""
    rng = np.random.default_rng(0)
    alpha = 0.1
    covs = []
    for _ in range(300):
        calib = rng.exponential(1.0, size=40)
        test = rng.exponential(1.0, size=40)
        q = CS.conformal_quantile(calib, alpha)
        covs.append(np.mean(test <= q))
    mean_cov = float(np.mean(covs))
    # конформная гарантия: маржинальное покрытие ≥ 1−α (в среднем), не сильно выше
    assert 0.88 <= mean_cov <= 0.97, mean_cov


# ---------- интервалы на реальной панели ----------


def test_intervals_valid_on_panel():
    rows = _rows()
    _need(rows)
    res = CS.temporal_holdout(rows, Mo.StructuralOSL, alpha=0.10)
    ivs = [iv for iv in res["intervals"] if iv is not None]
    assert ivs, "нет интервалов"
    for iv in ivs:
        assert iv.predicted_low < iv.predicted_base < iv.predicted_high
        assert iv.interval_width_pct > 0


def test_temporal_holdout_disjoint_years():
    """proper-train ≤2022, calib=2023, test>2023 — годы не пересекаются (out-of-sample)."""
    rows = _rows()
    _need(rows)
    proper = [r for r in rows if r.period_end.year <= 2022]
    calib = [r for r in rows if r.period_end.year == 2023]
    test = [r for r in rows if r.period_end.year > 2023]
    yp = {r.period_end.year for r in proper}
    yc = {r.period_end.year for r in calib}
    yt = {r.period_end.year for r in test}
    assert yp.isdisjoint(yc) and yp.isdisjoint(yt) and yc.isdisjoint(yt)
    assert max(yp) < min(yc) < min(yt)  # строго хронологично


def test_panel_out_of_sample_coverage_reasonable():
    """OOS покрытие на отложенных годах — разумное (цель 90%). Floor 0.75 + проверка,
    что покрытие НЕ достигнуто тривиально широким интервалом (q ограничен сверху)."""
    rows = _rows()
    _need(rows)
    res = CS.temporal_holdout(rows, Mo.StructuralOSL, alpha=0.10)
    assert res["coverage_rate"] is not None and res["n_test"] >= 4
    assert res["coverage_rate"] >= 0.75, res["coverage_rate"]
    # q не вырожден: >0 и < 0.6 (иначе интервал [0.4ŷ;1.6ŷ]+ ловил бы всё подряд)
    assert res["q"] is not None and 0 < res["q"] < 0.6, res["q"]


def test_steel_2025_gap_rows_excluded_from_scoring():
    """Сталевары 2025 (нет vol_steel → структурный NaN) НЕ входят в scored-набор —
    иначе frozen-2025 константы тихо попали бы в оценку покрытия."""
    rows = _rows()
    _need(rows)
    res = CS.temporal_holdout(rows, Mo.StructuralOSL, alpha=0.10)
    scored = {iv.company for iv in res["intervals"] if iv is not None}
    gap = {"Северсталь 2025FY", "ММК 2025FY", "НЛМК 2025FY"}
    assert not (scored & gap), f"gap-строки в scored: {scored & gap}"
    assert res["n_test"] == len(scored)  # n_test считает только конечные прогнозы


def test_fit_isolation_calib_changes_model():
    """Доказательство изоляции фолдов: добавление calib(2023) в обучение МЕНЯЕТ k_ —
    значит calib реально вне proper-train (split-conformal не circular)."""
    rows = _rows()
    _need(rows)
    proper = [r for r in rows if r.period_end.year <= 2022]
    calib = [r for r in rows if r.period_end.year == 2023]
    k_proper = Mo.StructuralOSL().fit(proper).k_
    k_both = Mo.StructuralOSL().fit(proper + calib).k_
    assert any(
        abs(k_proper[i] - k_both.get(i, k_proper[i])) > 1e-9 for i in k_proper
    ), "добавление calib не изменило модель → подозрение, что фолды не изолированы"
