"""
Тесты backtest_analyses.py — воспроизводимая сводка корпуса разборов (proxy-feedback).

Детерминизм: тесты гоняются на **bundled-фикстуре** `tests/fixtures/analyses/` (синтетические
разборы), а не на реальном корпусе `_Анализы/` — тот внутренний и в публичный репозиторий не входит.
Поэтому тесты всегда выполняются (без skip) на чистом клоне.
"""

import os

import pytest

import backtest_analyses as ba

FIXTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "analyses")


@pytest.fixture(autouse=True)
def _use_fixture_corpus(monkeypatch):
    # Подменяем каталог корпуса на bundled-фикстуру (load/summarize резолвят ANALYSES_DIR при вызове).
    monkeypatch.setattr(ba, "ANALYSES_DIR", FIXTURE_DIR)


def test_corpus_loads():
    items = ba.load_corpus()
    assert len(items) >= 3, "ожидаем ≥3 разбора в фикстуре"
    for it in items:
        assert {"file", "date", "tags", "is_osl_backtest", "size"} <= set(it)


def test_summary_keys_and_dates():
    s = ba.summarize_corpus()
    assert s["n_total"] >= 3
    assert s["n_dated"] >= 1
    # даты в формате YYYY-MM-DD, max не раньше min
    assert s["date_min"] and s["date_min"].count("-") == 2
    assert s["date_max"] >= s["date_min"]


def test_osl_backtest_present():
    # в фикстуре есть OSL- и conformal-разборы
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
