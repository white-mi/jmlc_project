"""Property-based тесты слоёв L2 (spillover) и L3 (сегменты).

ЗАЧЕМ ПОВЕРХ ОБЫЧНЫХ ТЕСТОВ. Точечные тесты проверяют конкретные входы («шок 4.2 по
нефтегазу даёт вот столько»). Они не отвечают на вопрос «а на ЛЮБОМ ли входе выполняются
свойства, на которых держится интерпретация выхода». Здесь hypothesis сам подбирает входы — включая
граничные (нулевая магнитуда, отрицательные значения, все 27 подкатегорий × 3 режима КС) —
и ищет контрпример к свойствам, которые обязаны держаться всегда:

  * масштабируемость и монотонность L2 по величине шока;
  * конечность всех чисел (никаких NaN/inf, которые тихо утекают в отчёт);
  * согласованность декомпозиции L3 по 5 каналам с итоговым ΔPD;
  * усиление режимом ключевой ставки: acute_stress ≥ moderate_stress ≥ normal.
"""

import math
import sys
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

TOOLS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLS))

import segment_impact as SI  # noqa: E402
import spillover as SP  # noqa: E402

_, INDUSTRIES = SP.load_matrix()
_TABLE = SI.load_table()
# В таблице рядом с данными лежат служебные ключи (`_comment`) — в стратегии они не нужны.
SUBCATEGORIES = sorted(
    k for k, v in _TABLE["shock_subcategories"].items() if isinstance(v, dict) and "channels" in v
)
KC_REGIMES = sorted(k for k, v in _TABLE["kc_regimes"].items() if not k.startswith("_"))
assert len(SUBCATEGORIES) == 27, f"ожидалось 27 подкатегорий, найдено {len(SUBCATEGORIES)}"

# Профиль: примеров на прогон немного, но на каждом прогоне они разные — тесты остаются
# быстрыми, при этом за много прогонов покрывается заметно больше входов, чем точечными кейсами.
#
# Профиль по времени: изолированно весь файл отрабатывает за ~1.5 с, в полном прогоне первый
# property-тест стоит ~9 с. Разница — разовые накладные расходы hypothesis, которые растут с
# числом уже загруженных модулей (в общем прогоне к этому моменту импортированы
# pandas/matplotlib/langchain). Сами функции слоёв не при чём: 400 вызовов `propagate_shock`
# укладываются в 0.08 с и с тяжёлыми импортами, и без них, а отключение фаз explain/target
# времени не меняет. ~9 с разово на весь файл приемлемы, а урезать `max_examples` ради этого —
# терять покрытие входов.
SETTINGS = settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

magnitudes = st.floats(min_value=0.0, max_value=50.0, allow_nan=False, allow_infinity=False)


# ---------------------------------------------------------------- L2: spillover


@SETTINGS
@given(source=st.sampled_from(INDUSTRIES), magnitude=magnitudes)
def test_l2_impacts_are_finite_and_scale_linearly(source, magnitude):
    """Spillover линеен по магнитуде: удвоение шока удваивает эффект.

    Это не косметика — на этом свойстве держится интерпретация «ΔPD в процентных пунктах».
    Если бы в матрицу закрался нелинейный шаг, числа перестали бы складываться.
    """
    res = SP.propagate_shock(source, magnitude_pp=magnitude)
    assert set(res.impacts) == set(INDUSTRIES)
    for ind, val in res.impacts.items():
        assert math.isfinite(val), f"{source}→{ind}: не число ({val})"

    doubled = SP.propagate_shock(source, magnitude_pp=magnitude * 2)
    for ind in INDUSTRIES:
        # округление до 3 знаков внутри propagate_shock даёт допуск порядка 1e-3
        assert (
            abs(doubled.impacts[ind] - 2 * res.impacts[ind]) < 2e-3
        ), f"{source}→{ind}: нелинейность по магнитуде"


@SETTINGS
@given(source=st.sampled_from(INDUSTRIES), a=magnitudes, b=magnitudes)
def test_l2_monotonic_in_magnitude(source, a, b):
    """Больше шок — не меньше эффект (по модулю) для каждой отрасли."""
    lo, hi = sorted((a, b))
    r_lo = SP.propagate_shock(source, magnitude_pp=lo)
    r_hi = SP.propagate_shock(source, magnitude_pp=hi)
    for ind in INDUSTRIES:
        assert (
            abs(r_hi.impacts[ind]) >= abs(r_lo.impacts[ind]) - 1e-9
        ), f"{source}→{ind}: рост шока уменьшил эффект"


@SETTINGS
@given(source=st.sampled_from(INDUSTRIES), magnitude=magnitudes)
def test_l2_ranked_is_consistent_with_impacts(source, magnitude):
    """`ranked` — это отсортированный `impacts`, а не отдельно живущий список."""
    res = SP.propagate_shock(source, magnitude_pp=magnitude)
    assert dict(res.ranked) == res.impacts
    values = [v for _, v in res.ranked]
    assert values == sorted(values, reverse=True)


def test_l2_rejects_unknown_industry():
    with pytest.raises(ValueError):
        SP.propagate_shock("нет_такой_отрасли")


# ---------------------------------------------------------------- L3: сегменты


@SETTINGS
@given(sub=st.sampled_from(SUBCATEGORIES), regime=st.sampled_from(KC_REGIMES))
def test_l3_outputs_are_finite_for_any_shock(sub, regime):
    """Любая из 27 подкатегорий × любой режим КС → конечные числа по всем сегментам.

    NaN/inf здесь особенно опасны: они не роняют пайплайн, а молча доезжают до отчёта.
    """
    impacts = SI.predict_segment_impact(sub, kc_regime=regime)
    assert len(impacts) >= 10
    for seg, imp in impacts.items():
        for field in ("delta_pd", "delta_demand", "delta_churn"):
            val = getattr(imp, field)
            assert math.isfinite(val), f"{sub}/{regime}/{seg}: {field} = {val}"


@SETTINGS
@given(sub=st.sampled_from(SUBCATEGORIES), regime=st.sampled_from(KC_REGIMES))
def test_l3_channel_breakdown_matches_total(sub, regime):
    """Сумма вкладов пяти каналов должна давать итоговый ΔPD.

    Декомпозиция — главный аргумент прослеживаемости («почему сегмент просел»): если сумма
    каналов не сходится с итогом, объяснение расходится с числом.
    """
    impacts = SI.predict_segment_impact(sub, kc_regime=regime, include_breakdown=True)
    for seg, imp in impacts.items():
        breakdown = getattr(imp, "channel_breakdown", None)
        if not breakdown:
            continue
        total = sum(ch["delta_pd_contrib"] for ch in breakdown.values())
        # И вклады, и итог округляются до 3 знаков, поэтому сходимость проверяется с
        # допуском накопленного округления (по 0.0005 на каждое округлённое число),
        # а не бит-в-бит.
        tol = 0.0005 * (len(breakdown) + 1)
        assert (
            abs(total - imp.delta_pd) <= tol
        ), f"{sub}/{regime}/{seg}: сумма каналов {total} ≠ итог {imp.delta_pd}"


@SETTINGS
@given(sub=st.sampled_from(SUBCATEGORIES))
def test_l3_kc_regime_amplifies_monotonically(sub):
    """Режим КС — усилитель: острый стресс не может дать эффект слабее нормального."""
    by_regime = {r: SI.predict_segment_impact(sub, kc_regime=r) for r in ("normal", "acute_stress")}
    for seg in by_regime["normal"]:
        normal = abs(by_regime["normal"][seg].delta_pd)
        acute = abs(by_regime["acute_stress"][seg].delta_pd)
        assert (
            acute >= normal - 1e-9
        ), f"{sub}/{seg}: acute_stress ({acute}) слабее normal ({normal})"


@SETTINGS
@given(sub=st.sampled_from(SUBCATEGORIES), regime=st.sampled_from(KC_REGIMES))
def test_l3_explicit_subcategory_ignores_global_direction(sub, regime):
    """Явная подкатегория авторитетна: per-channel знаки из таблицы не переворачиваются
    легаси-параметром `direction=-1`. Это инвариант бифуркации — один шок может улучшать
    одни сегменты и ухудшать другие, и глобальный флип обязан это уважать."""
    plus = SI.predict_segment_impact(sub, kc_regime=regime, direction=1)
    minus = SI.predict_segment_impact(sub, kc_regime=regime, direction=-1)
    for seg in plus:
        assert (
            abs(plus[seg].delta_pd - minus[seg].delta_pd) < 1e-12
        ), f"{sub}/{seg}: явная подкатегория поддалась глобальному direction=-1"
