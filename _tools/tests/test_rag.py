"""Тест RAG-индексации (S1.2): index_single делает UPSERT и НЕ стирает корпус."""

import sqlite3
import sys
from pathlib import Path


TOOLS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLS / 'agents' / 'rag'))


def _make_md(dir_path: Path, name: str, title: str) -> Path:
    p = dir_path / name
    p.write_text(
        '---\n'
        'дата_новости: "2026-06-01"\n'
        '---\n'
        f'# {title}\n\n'
        '## L0 — Классификация\n'
        'Что-то про ключевую ставку ЦБ, рубль, санкции и кризис.\n',
        encoding='utf-8',
    )
    return p


def _count(db_path: Path) -> int:
    conn = sqlite3.connect(str(db_path))
    try:
        return conn.execute('SELECT COUNT(*) FROM news_analyses').fetchone()[0]
    finally:
        conn.close()


def test_index_single_upserts_without_wiping(tmp_path):
    import init_db
    import index_news as ix

    db = tmp_path / 'rag_test.db'
    analyses = tmp_path / '_Анализы'
    analyses.mkdir()

    # Схема + 2 исходных файла → полный индекс
    init_db.init_db(db_path=db)
    _make_md(analyses, '2026-06-01 — A.md', 'Анализ A про ставку')
    _make_md(analyses, '2026-06-02 — B.md', 'Анализ B про санкции')
    n0 = ix.index_all(db_path=db, analyses_dir=analyses)
    assert n0 == 2
    assert _count(db) == 2

    # index_single нового файла → 3 (корпус НЕ стёрт)
    c = _make_md(analyses, '2026-06-03 — C.md', 'Анализ C про кризис')
    assert ix.index_single(c, db_path=db) is True
    assert _count(db) == 3, 'index_single стёр корпус вместо UPSERT'

    # Повторный index_single того же файла → остаётся 3 (UPSERT, не дубль)
    assert ix.index_single(c, db_path=db) is True
    assert _count(db) == 3, 'повторный index_single создал дубликат'
