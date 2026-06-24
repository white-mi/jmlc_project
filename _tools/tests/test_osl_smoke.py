"""Smoke-тесты для всех 7 OSL-модулей: каждый возвращает выручку
в ожидаемом диапазоне после применения калибровок."""

import pytest


@pytest.fixture(scope='module', autouse=True)
def apply_calibrations():
    from osl_calibrator import apply_all_calibrations
    apply_all_calibrations()


def _check_predict(module_name, company, expected_rub_bn, tolerance_pct=15):
    import importlib
    m = importlib.import_module(module_name)
    pred = m.predict_revenue(company).predicted_rub_bn
    assert pred > 0, f'{company}: predicted_rub_bn must be > 0'
    rel_err = abs(pred - expected_rub_bn) / expected_rub_bn
    assert rel_err < tolerance_pct / 100, (
        f'{company}: predicted={pred:.0f}, expected≈{expected_rub_bn}, '
        f'err={rel_err*100:.1f}% > tol={tolerance_pct}%'
    )


@pytest.mark.parametrize('company,expected', [
    ('Норникель', 1225),  # USD bn → RUB через FX
    ('Северсталь', 713),
    ('ММК', 610),
    ('НЛМК', 831),
    ('Полюс', 776),
])
def test_metallurgy(company, expected):
    _check_predict('osl_metallurgy', company, expected, tolerance_pct=15)


@pytest.mark.parametrize('company,expected', [
    ('Роснефть', 8236),
    ('ЛУКОЙЛ', 3768),
    ('Газпром', 7000),
    ('Новатэк', 1446),
])
def test_oilgas(company, expected):
    _check_predict('osl_oilgas', company, expected, tolerance_pct=10)


@pytest.mark.parametrize('company,expected', [
    ('ФосАгро', 590),
    ('Акрон', 238),
    ('СИБУР', 1200),
])
def test_chemistry(company, expected):
    _check_predict('osl_chemistry', company, expected, tolerance_pct=15)


@pytest.mark.parametrize('company,expected', [
    ('Интер РАО', 1540),
    ('РусГидро', 580),
    ('Юнипро', 130),
    ('Т Плюс', 470),
    ('Росатом-Энергоатом', 480),
])
def test_energy(company, expected):
    _check_predict('osl_energy', company, expected, tolerance_pct=10)


@pytest.mark.parametrize('company,expected', [
    ('Wildberries', 945),
    ('Ozon', 998),
    ('М.Видео', 451),
])
def test_retail(company, expected):
    _check_predict('osl_retail', company, expected, tolerance_pct=15)


@pytest.mark.parametrize('company,expected', [
    ('Пульс', 386),
    ('Протек', 374),
    ('Катрен', 325),
])
def test_pharma(company, expected):
    _check_predict('osl_pharma', company, expected, tolerance_pct=10)


@pytest.mark.parametrize('region,expected,tolerance', [
    ('ХМАО-Югра', 360, 15),
    ('Тюменская обл.', 240, 15),
    ('Татарстан', 480, 15),
    ('Сахалинская обл.', 172, 15),
])
def test_oiv(region, expected, tolerance):
    _check_predict('osl_oiv', region, expected, tolerance_pct=tolerance)
