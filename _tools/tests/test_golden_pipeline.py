"""Golden-снапшот числового пайплайна L1 → L1.5 → L2 → L3.

ЗАЧЕМ. Остальные тесты слоёв проверяют инварианты и диапазоны («выручка в пределах ±15 %»,
«ΔPD одного сегмента больше другого»). Это ловит грубые поломки, но НЕ ловит тихий численный
дрейф: при сдвинутом коэффициенте в матрице spillover все инварианты продолжают выполняться,
а числа на выходе уже другие, и это проходит незамеченным. Здесь выход фиксируется целиком
и сравнивается поэлементно.

Снапшот — это не «правильный ответ», а зафиксированный эталон выдачи. Если изменение чисел
ожидаемое, снапшот пересоздаётся осознанно:

    cd _tools && RADAR_UPDATE_GOLDEN=1 python -m pytest tests/test_golden_pipeline.py -q

и diff файла `tests/golden/pipeline_smoke_4.2_oilgas.json` идёт в ревью как часть правки.
"""

import json
import os
from pathlib import Path

import pytest

GOLDEN = Path(__file__).parent / "golden" / "pipeline_smoke_4.2_oilgas.json"

# timestamp — время запуска; в снапшот не входит. Дата новости фиксируется параметром,
# чтобы прогон не зависел от календаря.
VOLATILE_KEYS = {"timestamp"}
FIXED_DATE = "2026-01-01"
TOL = 1e-9


def _run_pipeline() -> dict:
    import run_pipeline

    state = run_pipeline.run_full_pipeline(
        smoke_shock="4.2",
        smoke_industry="oilgas",
        date=FIXED_DATE,
    )
    state = {k: v for k, v in state.items() if k not in VOLATILE_KEYS}
    # Прогон через JSON — тот же путь, что у `run_pipeline --json`: кортежи становятся
    # списками, и сравнение идёт в терминах фактического формата выдачи, а не внутренних
    # типов Python.
    return json.loads(json.dumps(state, ensure_ascii=False, default=str))


def _compare(actual, expected, path=""):
    """Рекурсивное сравнение с допуском по числам. Возвращает список расхождений."""
    diffs = []
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return [f"{path}: ожидался объект, получен {type(actual).__name__}"]
        for key in expected.keys() | actual.keys():
            if key not in actual:
                diffs.append(f"{path}/{key}: поле исчезло из выхода пайплайна")
            elif key not in expected:
                diffs.append(f"{path}/{key}: новое поле, которого нет в снапшоте")
            else:
                diffs += _compare(actual[key], expected[key], f"{path}/{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list):
            return [f"{path}: ожидался список, получен {type(actual).__name__}"]
        if len(actual) != len(expected):
            return [f"{path}: длина списка {len(actual)} вместо {len(expected)}"]
        for i, (a, e) in enumerate(zip(actual, expected)):
            diffs += _compare(a, e, f"{path}[{i}]")
    elif isinstance(expected, bool) or isinstance(actual, bool):
        if actual != expected:
            diffs.append(f"{path}: {actual} вместо {expected}")
    elif isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if abs(actual - expected) > TOL:
            diffs.append(f"{path}: {actual} вместо {expected} (Δ={actual - expected:+.6g})")
    elif actual != expected:
        diffs.append(f"{path}: {actual!r} вместо {expected!r}")
    return diffs


def test_pipeline_matches_golden_snapshot():
    actual = _run_pipeline()

    if os.environ.get("RADAR_UPDATE_GOLDEN") == "1":
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(json.dumps(actual, ensure_ascii=False, indent=2), encoding="utf-8")
        pytest.skip(f"снапшот перезаписан: {GOLDEN.name} (RADAR_UPDATE_GOLDEN=1)")

    assert (
        GOLDEN.exists()
    ), f"нет эталона {GOLDEN}; создать: RADAR_UPDATE_GOLDEN=1 pytest {Path(__file__).name}"
    expected = json.loads(GOLDEN.read_text(encoding="utf-8"))
    diffs = _compare(actual, expected)
    assert not diffs, "числовой выход пайплайна разошёлся с эталоном:\n  " + "\n  ".join(diffs[:25])


def test_golden_covers_all_layers():
    """Эталон должен покрывать все слои — иначе снапшот «зелёный», а половина пайплайна
    вне наблюдения."""
    expected = json.loads(GOLDEN.read_text(encoding="utf-8"))
    for layer in ("L1_macro", "L1_5_osl", "L2_spillover", "L3_segments"):
        assert layer in expected and expected[layer], f"снапшот не покрывает слой {layer}"
    assert len(expected["L3_segments"]) >= 10, "в снапшоте подозрительно мало сегментов L3"


def test_compare_detects_drift():
    """Сам компаратор обязан ловить дрейф — иначе golden-тест зелёный по недосмотру."""
    base = {"a": {"b": 1.0}, "c": [1, 2]}
    assert _compare(base, base) == []
    assert _compare({"a": {"b": 1.0 + 1e-6}, "c": [1, 2]}, base)
    assert _compare({"a": {"b": 1.0}, "c": [1, 3]}, base)
    assert _compare({"a": {}, "c": [1, 2]}, base)
