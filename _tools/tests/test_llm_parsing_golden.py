"""Golden-фикстуры сырых ответов LLM → контракт парсера `orchestrator.extract_json`.

ЗАЧЕМ. Это единственная точка, где недетерминированный текст модели превращается в
структуру, на которой дальше стоит весь численный пайплайн. Ошибка здесь не «падает
красиво»: агент возвращает мусор, а L1.5→L3 честно считают числа по мусору.

Прогнать реальную модель в CI нельзя (платно, недетерминированно), поэтому зафиксированы
СЫРЫЕ ФОРМЫ ответов, которые модели реально выдают: чистый JSON, markdown-фенс с языком и
без, болтовня вокруг JSON, незакрытый фенс, обрезанный по лимиту токенов JSON, вложенные
объекты с юникодом и экранированными кавычками, префикс рассуждения у thinking-моделей.
Фикстуры лежат в `tests/fixtures/llm_raw/` — добавить новый случай = положить файл.

Тесты API не требуют.
"""

import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLS / "agents"))

import orchestrator  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures" / "llm_raw"

# Что обязано быть извлечено из каждой «хорошей» формы ответа.
EXPECTED = {
    "01_plain_json.txt": ("4", "4.2"),
    "02_fenced_json.txt": ("1", "1.4"),
    "03_fenced_no_lang.txt": ("3", "3.1"),
    "04_prose_around_json.txt": ("3", "3.5"),
    "05_unclosed_fence.txt": ("2", "2.3"),
    "06_truncated_json.txt": ("2", "2.1"),
    "07_nested_unicode.txt": ("5", "5.2"),
    "08_thinking_prefix.txt": ("5", "5.2"),
}
UNPARSEABLE = ["90_no_json_at_all.txt", "91_empty.txt"]


def _read(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def test_fixtures_present():
    """Набор фикстур не должен молча усохнуть до одного тривиального случая."""
    on_disk = {p.name for p in FIXTURES.glob("*.txt")}
    assert on_disk == set(EXPECTED) | set(
        UNPARSEABLE
    ), f"состав фикстур разошёлся с тестом: {on_disk ^ (set(EXPECTED) | set(UNPARSEABLE))}"


@pytest.mark.parametrize("name,expected", sorted(EXPECTED.items()))
def test_extract_json_handles_real_response_shapes(name, expected):
    obj = orchestrator.extract_json(_read(name))
    assert isinstance(obj, dict), f"{name}: получен не объект"
    assert (
        obj.get("main_category"),
        obj.get("subcategory"),
    ) == expected, f"{name}: классификация извлечена неверно"
    assert obj.get("WHAT"), f"{name}: потеряно поле WHAT"


def test_truncated_json_is_recovered_not_faked():
    """Обрезанный ответ достраивается скобками, но данные не выдумываются:
    поля, до которых модель не дошла, остаются отсутствующими."""
    obj = orchestrator.extract_json(_read("06_truncated_json.txt"))
    assert obj["subcategory"] == "2.1"
    assert obj["severity_score"] == 91
    # хвост оборвался внутри вложенного объекта — он либо восстановлен частично, либо пуст,
    # но выдуманных ключей быть не должно
    assert set(obj) <= {
        "main_category",
        "subcategory",
        "severity_score",
        "severity_level",
        "WHAT",
        "detail",
    }


def test_nested_structures_survive():
    obj = orchestrator.extract_json(_read("07_nested_unicode.txt"))
    assert obj["notes"]["tags"] == ["шум", "ожидаемо"]
    assert '"нейтрально"' in obj["notes"]["quote"], "потеряны экранированные кавычки"


@pytest.mark.parametrize("name", UNPARSEABLE)
def test_unparseable_response_raises_instead_of_returning_garbage(name):
    """Если JSON'а нет вовсе — парсер обязан упасть, а не вернуть пустой dict:
    молчаливый `{}` уехал бы дальше по пайплайну как «шок без параметров»."""
    with pytest.raises(Exception):
        orchestrator.extract_json(_read(name))
