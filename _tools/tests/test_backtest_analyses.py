"""
Тесты backtest_analyses.py — воспроизводимая сводка корпуса `_Анализы/` (proxy-feedback).

Аддитивный модуль продуктового слоя: читает только `_Анализы/`, не пишет туда, stdlib-only.
Тесты мягко SKIP-аются, если корпус не найден (например, у контрибьютора без полного vault).
"""

import os

import pytest

import backtest_analyses as ba

_HAS_CORPUS = os.path.isdir(ba.ANALYSES_DIR)
pytestmark = pytest.mark.skipif(
    not _HAS_CORPUS, reason="корпус _Анализы/ недоступен в этой среде"
)


def test_corpus_loads():
    items = ba.load_corpus()
    assert len(items) >= 10, "ожидаем ≥10 разборов в корпусе"
    # каждый элемент имеет ожидаемые поля
    for it in items:
        assert {"file", "date", "tags", "is_osl_backtest", "size"} <= set(it)


def test_summary_keys_and_dates():
    s = ba.summarize_corpus()
    assert s["n_total"] >= 10
    assert s["n_dated"] >= 1
    # даты в формате YYYY-MM-DD, max не раньше min
    assert s["date_min"] and s["date_min"].count("-") == 2
    assert s["date_max"] >= s["date_min"]


def test_osl_backtest_present():
    # в корпусе есть хотя бы один OSL/conformal/бэктест-разбор
    assert ba.summarize_corpus()["n_osl"] >= 1


def test_frontmatter_parser_minimal():
    fm = ba._parse_frontmatter('---\nдата_анализа: "2026-04-25"\nтег: x\n---\nтело')
    assert fm.get("дата_анализа") == "2026-04-25"
    # без фронтматтера — пустой dict
    assert ba._parse_frontmatter("обычный текст") == {}


def test_render_nonempty():
    s = ba.summarize_corpus()
    text = ba.render_summary(s)
    assert "Сводка корпуса" in text
    assert "Всего разборов" in text
    # таблица содержит хотя бы один файл
    assert text.count("|") >= 6


def test_emit_writes_file(tmp_path, monkeypatch):
    # перенаправляем _HERE так, чтобы output/ писался во временную папку
    monkeypatch.setattr(ba, "_HERE", str(tmp_path))
    ba.main(["--emit"])
    out = tmp_path / "output" / "backtest" / "feedback.md"
    assert out.exists()
    assert "Сводка корпуса" in out.read_text(encoding="utf-8")
