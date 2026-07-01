"""
Smoke-тесты EDA (eda_osl.py), industry-параметрические. Пропускаются, если нет extra [eda]
(pandas/matplotlib), чтобы core-CI без тяжёлого стека оставался зелёным.
"""

import pytest

pytest.importorskip("pandas")
pytest.importorskip("matplotlib")

import numpy as np  # noqa: E402
import eda_osl  # noqa: E402
import osl_panel  # noqa: E402


def _has_panel(ind):
    return bool(osl_panel.load_panel(ind))


@pytest.mark.parametrize(
    "industry,price_cols",
    [
        ("metallurgy", {"p_gold", "p_copper", "p_usd_rub"}),
        ("energy", {"p_electricity_rsv", "p_capacity_kom", "p_usd_rub"}),
    ],
)
def test_build_df_shape(industry, price_cols):
    if not _has_panel(industry):
        pytest.skip(f"панель {industry} пуста")
    df = eda_osl.build_df(industry)
    assert len(df) >= 18
    assert {"issuer", "year", "target", "currency"}.issubset(df.columns)
    assert price_cols.issubset(df.columns)  # цены отрасли приджойнены


@pytest.mark.parametrize("industry", ["metallurgy", "energy", "chemistry"])
def test_run_generates_all_figures(industry, tmp_path):
    if not _has_panel(industry):
        pytest.skip(f"панель {industry} пуста")
    notes = eda_osl.run(industry, out_dir=tmp_path)
    assert len(notes) == len(eda_osl.FIGURES) == 8
    assert len(list(tmp_path.glob("*.png"))) == 8
    assert (tmp_path / "implications.md").exists()
    # ни одна фигура не должна упасть (run() ловит исключения в строку 'ОШИБКА')
    errs = [n for n in notes if "ОШИБКА" in n]
    assert not errs, errs


def test_metallurgy_anchor_fx_present(tmp_path):
    """Металлургия — регрессионный якорь: FX-фигура (USD-корзина) ДОЛЖНА строиться, не пропускаться."""
    if not _has_panel("metallurgy"):
        pytest.skip("панель пуста")
    notes = eda_osl.run("metallurgy", out_dir=tmp_path)
    fx = [n for n in notes if n.startswith("03")]
    assert fx and "ПРОПУЩЕНО" not in fx[0], "у металлургии FX-фигура должна строиться (USD-корзина)"


def test_fx_skipped_for_rub_industry(tmp_path):
    """RUB-отрасль (энергетика): FX-фигура честно ПРОПУЩЕНА, но PNG-заглушка всё равно создаётся."""
    if not _has_panel("energy"):
        pytest.skip("панель пуста")
    notes = eda_osl.run("energy", out_dir=tmp_path)
    fx = [n for n in notes if n.startswith("03")]
    assert fx and "ПРОПУЩЕНО" in fx[0]
    assert (tmp_path / "03_fx_passthrough.png").exists()


def test_vif_detects_collinearity():
    """VIF идентичных колонок → очень высокий (коллинеарность поймана)."""
    X = np.array(
        [[1.0, 2.0, 2.0], [2.0, 4.0, 4.0], [3.0, 1.0, 1.0], [4.0, 3.0, 3.0], [5.0, 5.0, 5.0]]
    )  # col1 ≡ col2
    vif = eda_osl._vif(X)
    assert vif[1] > 50 and vif[2] > 50, vif


def test_loglog_slope_known():
    """y = x^2 → log-log наклон = 2."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    slope, n = eda_osl._loglog_slope(x, x**2)
    assert n == 5
    assert abs(slope - 2.0) < 1e-6
