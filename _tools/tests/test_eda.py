"""
Smoke-тесты EDA (eda_osl.py). Пропускаются, если нет extra [eda] (pandas/matplotlib),
чтобы core-CI без тяжёлого стека оставался зелёным.
"""

import pytest

pytest.importorskip('pandas')
pytest.importorskip('matplotlib')

import numpy as np  # noqa: E402
import eda_osl       # noqa: E402
import osl_panel     # noqa: E402


def _has_panel():
    return bool(osl_panel.load_panel('metallurgy'))


def test_build_df_shape():
    if not _has_panel():
        pytest.skip('панель пуста')
    df = eda_osl.build_df()
    assert len(df) >= 20
    assert {'issuer', 'year', 'target', 'currency'}.issubset(df.columns)
    # цены приджойнены
    assert {'p_gold', 'p_copper', 'p_usd_rub'}.issubset(df.columns)


def test_run_generates_all_figures(tmp_path, monkeypatch):
    if not _has_panel():
        pytest.skip('панель пуста')
    monkeypatch.setattr(eda_osl, 'OUT', tmp_path)
    notes = eda_osl.run()
    assert len(notes) == len(eda_osl.FIGURES) == 8
    assert len(list(tmp_path.glob('*.png'))) == 8
    assert (tmp_path / 'implications.md').exists()
    # ни одна фигура не должна упасть (run() ловит исключения в строку 'ОШИБКА')
    errs = [n for n in notes if 'ОШИБКА' in n]
    assert not errs, errs


def test_vif_detects_collinearity():
    """VIF идентичных колонок → очень высокий (коллинеарность поймана)."""
    X = np.array([[1., 2., 2.], [2., 4., 4.], [3., 1., 1.],
                  [4., 3., 3.], [5., 5., 5.]])  # col1 ≡ col2
    vif = eda_osl._vif(X)
    assert vif[1] > 50 and vif[2] > 50, vif


def test_loglog_slope_known():
    """y = x^2 → log-log наклон = 2."""
    x = np.array([1., 2., 3., 4., 5.])
    slope, n = eda_osl._loglog_slope(x, x ** 2)
    assert n == 5
    assert abs(slope - 2.0) < 1e-6
