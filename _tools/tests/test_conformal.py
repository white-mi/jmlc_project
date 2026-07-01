"""Тесты Conformal Prediction: 90% interval содержит actual для большинства
эмитентов; width > 0 (perturbation работает)."""

import pytest


@pytest.fixture(scope="module")
def conformal():
    from conformal_prediction import (
        make_interval_from_metallurgy,
        make_interval_generic,
        make_interval_retail,
        make_interval_energy,
        make_interval_oiv,
        make_interval_pharma,
    )

    return {
        "metallurgy": make_interval_from_metallurgy,
        "generic": make_interval_generic,
        "retail": make_interval_retail,
        "energy": make_interval_energy,
        "oiv": make_interval_oiv,
        "pharma": make_interval_pharma,
    }


@pytest.mark.parametrize("company", ["Норникель", "Северсталь", "ММК", "НЛМК", "Полюс"])
def test_metallurgy_inside(company, conformal):
    r = conformal["metallurgy"](company)
    assert r.interval_width_pct > 1.0, f"{company}: width too low ({r.interval_width_pct:.2f}%)"
    assert r.actual_in_interval is True, (
        f"{company}: actual={r.actual} not in [{r.predicted_low:.0f}; " f"{r.predicted_high:.0f}]"
    )


@pytest.mark.parametrize(
    "company",
    [
        "Интер РАО",
        "РусГидро",
        "Юнипро",
        "Т Плюс",
        "Росатом-Энергоатом",
    ],
)
def test_energy_inside(company, conformal):
    r = conformal["energy"](company)
    assert r.interval_width_pct > 1.0
    assert r.actual_in_interval is True


@pytest.mark.parametrize("company", ["Wildberries", "Ozon", "М.Видео"])
def test_retail_inside(company, conformal):
    r = conformal["retail"](company)
    assert r.interval_width_pct > 1.0
    assert r.actual_in_interval is True


@pytest.mark.parametrize("company", ["Пульс", "Протек", "Катрен"])
def test_pharma_inside(company, conformal):
    r = conformal["pharma"](company)
    assert r.interval_width_pct > 1.0
    assert r.actual_in_interval is True


def test_in_sample_coverage(conformal):
    """S2.3: IN-SAMPLE покрытие ≥85%.

    ВАЖНО: это НЕ out-of-sample валидация. Интервал калибруется и проверяется
    на ОДНИХ И ТЕХ ЖЕ захардкоженных ACTUAL_REVENUE_*_2025, поэтому метрика
    измеряет внутреннюю согласованность/остроту интервалов, а не обобщающую
    способность модели. Настоящий temporal hold-out (калибровка на 9M →
    проверка на 12M) требует НЕЗАВИСИМЫХ 9M-actuals из IR — см. skip-тест ниже
    и пункт S4.2 плана. Раньше тест назывался test_overall_coverage_above_85_pct
    и выдавал «96% покрытие» за out-of-sample (находка F5 аудита)."""
    import osl_metallurgy as m_met
    import osl_oilgas as m_og
    import osl_chemistry as m_chem
    import osl_pharma as m_ph
    import osl_retail as m_ret
    import osl_oiv as m_oiv

    inside, total = 0, 0

    # Metallurgy
    for c in m_met.PROFILES.keys():
        r = conformal["metallurgy"](c)
        if r.actual_in_interval is not None:
            total += 1
            if r.actual_in_interval:
                inside += 1

    # Oilgas (через generic)
    for c, data in m_og.ACTUAL_REVENUE_12M_2025.items():
        actual = data.get("rub_bn")
        if actual:
            r = conformal["generic"](c, "osl_oilgas")
            if r.actual_in_interval is not None:
                total += 1
                if r.actual_in_interval:
                    inside += 1

    # Chemistry (через generic)
    for c, data in m_chem.ACTUAL_REVENUE_2025.items():
        actual = data.get("rub_bn")
        if actual:
            r = conformal["generic"](c, "osl_chemistry")
            if r.actual_in_interval is not None:
                total += 1
                if r.actual_in_interval:
                    inside += 1

    # Pharma
    for c in m_ph.PROFILES.keys():
        r = conformal["pharma"](c)
        if r.actual_in_interval is not None:
            total += 1
            if r.actual_in_interval:
                inside += 1

    # Retail
    for c in m_ret.PROFILES.keys():
        r = conformal["retail"](c)
        if r.actual_in_interval is not None:
            total += 1
            if r.actual_in_interval:
                inside += 1

    # Energy + OIV (специальные)
    import osl_energy as m_en

    for c in m_en.PROFILES.keys():
        r = conformal["energy"](c)
        if r.actual_in_interval is not None:
            total += 1
            if r.actual_in_interval:
                inside += 1
    for region in m_oiv.PROFILES.keys():
        r = conformal["oiv"](region)
        if r.actual_in_interval is not None:
            total += 1
            if r.actual_in_interval:
                inside += 1

    pct = 100 * inside / max(1, total)
    assert pct >= 85, f"In-sample coverage {inside}/{total} = {pct:.0f}% < 85%"


def test_holdout_coverage_metallurgy():
    """S2.3: НАСТОЯЩИЙ temporal hold-out — РАЗБЛОКИРОВАН (DS-доработка).

    Раньше блокировался, т.к. 9M-actuals = period_share × 12M (не независимы).
    Теперь есть независимая панель FY2021-2025 (data/panel/) → split-conformal:
      proper-train ≤2022 → calib 2023 (относит. остатки) → ОТЛОЖЕННЫЙ test 2024-2025.
    Годы не пересекаются, test строго в будущем → покрытие ЧЕСТНО out-of-sample
    (а не in-sample, как perturbation-интервалы в этом же модуле).
    """
    import osl_panel
    import conformal_split as CS
    import osl_models as Mo

    rows = [r for r in osl_panel.load_panel("metallurgy") if r.has_target and r.period_end]
    if len(rows) < 8:
        pytest.skip("панель не заполнена")

    res = CS.temporal_holdout(rows, Mo.StructuralOSL, alpha=0.10)
    # out-of-sample: калибровка и тест на разных годах
    assert res["n_calib"] >= 3 and res["n_test"] >= 4
    # q не вырожден: интервал содержательный, а не тривиально широкий
    assert res["q"] is not None and 0 < res["q"] < 0.6, res["q"]
    # цель покрытия 90%; floor 0.75 (метод консервативен на n_calib≈5, но покрытие
    # не должно достигаться абсурдно широким интервалом — это ловит q<0.6 выше)
    assert (
        res["coverage_rate"] >= 0.75
    ), f"OOS coverage {res['coverage_rate']:.0%} (inside {res['inside']}/{res['n_test']})"
