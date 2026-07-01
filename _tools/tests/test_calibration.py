"""Тесты OSL Calibrator: tune_param, tune_multi_period, tune_multi_param,
apply_all_calibrations работают и не падают."""

import json
from pathlib import Path

import pytest

CAL_DIR = Path(__file__).resolve().parent.parent / "calibration"


def test_apply_all_calibrations_returns_summary():
    from osl_calibrator import apply_all_calibrations

    summary = apply_all_calibrations()
    assert isinstance(summary, dict)
    assert len(summary) == 7
    expected = {
        "osl_metallurgy",
        "osl_oilgas",
        "osl_chemistry",
        "osl_pharma",
        "osl_retail",
        "osl_energy",
        "osl_oiv",
    }
    assert set(summary.keys()) == expected


@pytest.mark.parametrize(
    "module",
    [
        "osl_metallurgy",
        "osl_oilgas",
        "osl_chemistry",
        "osl_pharma",
        "osl_retail",
        "osl_energy",
        "osl_oiv",
    ],
)
def test_calibration_json_present(module):
    path = CAL_DIR / f"{module}_calibrated.json"
    assert path.exists(), f"Missing calibration JSON: {path}"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    assert len(data) > 0, f"{module}: empty calibration JSON"


def test_tune_param_finds_min():
    from osl_calibrator import tune_param

    # Калибровка y = (x - 5)^2 → minimum at x=5, target=0
    def predict(p):
        return (p - 5) ** 2 + 0.001  # +epsilon чтобы не было /0

    best, mae = tune_param(predict, target=0.001, param_min=0, param_max=10)
    assert abs(best - 5) < 0.5, f"Expected best≈5, got {best}"


def test_tune_multi_period():
    from osl_calibrator import tune_multi_period

    # predict(p) возвращает 12M — multi-period должен учитывать оба
    def predict(p):
        return p * 100  # параметр × 100

    targets = {"12M": 1000, "9M": 750}
    period_shares = {"12M": 1.0, "9M": 0.75}
    best, metrics = tune_multi_period(predict, targets, period_shares, param_min=0, param_max=20)
    # При p=10 → predict=1000, 9M=750 → perfect fit
    assert abs(best - 10.0) < 0.5
    assert metrics["mae_12M_pct"] < 5
    assert metrics["mae_9M_pct"] < 5


def test_tune_multi_param_via_scipy():
    from osl_calibrator import tune_multi_param

    # f(x, y) = (x-3)^2 + (y-7)^2 → min at (3, 7), value 0
    def predict(params):
        x, y = params
        return (x - 3) ** 2 + (y - 7) ** 2 + 100  # +100 чтобы было > 0 (target=100)

    bounds = [(0, 10), (0, 10)]
    best, mae = tune_multi_param(predict, target=100, bounds=bounds, maxiter=50)
    assert abs(best[0] - 3) < 0.5
    assert abs(best[1] - 7) < 0.5
    assert mae < 5
