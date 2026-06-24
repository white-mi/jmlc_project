"""
RAG v1.1 — индексация анализов из _Анализы/ в БД.

Парсит markdown-файлы с frontmatter, генерирует embeddings, сохраняет в БД.
"""

import sqlite3
import sys
import re
import argparse
import yaml
import numpy as np
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ANALYSES_DIR = Path(__file__).parent.parent.parent.parent / "_Анализы"
DB_PATH = Path(__file__).parent / "radar_rag.db"


def parse_markdown_analysis(file_path: Path) -> dict:
    """
    Распарсить файл анализа: frontmatter + ключевые секции.
    """
    text = file_path.read_text(encoding="utf-8")

    # Frontmatter
    fm = {}
    if text.startswith("---"):
        fm_end = text.find("\n---\n", 3)
        if fm_end > 0:
            try:
                fm = yaml.safe_load(text[3:fm_end]) or {}
            except yaml.YAMLError:
                fm = {}
            text = text[fm_end + 5:]

    # Заголовок (первый # )
    title_match = re.search(r"^# (.+?)$", text, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else file_path.stem

    # WHAT — обычно в L0 секции или в Источнике
    what_match = re.search(r"## L0[^\n]*\n(.+?)(?=##|\Z)", text, re.DOTALL)
    what = what_match.group(1).strip()[:500] if what_match else ""

    # Полный текст для full embedding
    full_text = text[:3000]  # ограничение для эмбеддинга

    # Извлекаем регион из заголовка / тегов
    macro_region = ""
    micro_region = ""
    for r in ["MSK_SPB", "DONOR", "INDUSTRIAL", "SOUTH_CAUCASUS", "FAR_EAST", "SIBERIA", "CENTRAL", "RURAL"]:
        if r in text:
            macro_region = r
            break
    for r in ["TOURIST_RESORT", "URBAN_INDUSTRIAL", "MONOTOWN", "AGRICULTURAL_RURAL", "CAPITAL_DIVERSIFIED"]:
        if r in text:
            micro_region = r
            break

    # Извлекаем категорию шока
    shock_cat = fm.get("шок_категория", "")
    main_cat = shock_cat.split()[0] if shock_cat else ""

    severity_str = fm.get("сила_шока", "")
    severity_score = None
    severity_level = None
    sev_match = re.search(r"(\d+)/100", severity_str)
    if sev_match:
        severity_score = int(sev_match.group(1))
    lvl_match = re.search(r"\b([HML]+)\b", severity_str)
    if lvl_match:
        severity_level = lvl_match.group(1)

    return {
        "file_path": str(file_path),
        "date": str(fm.get("дата_новости", file_path.stem.split(" — ")[0] if " — " in file_path.stem else "")),
        "title": title,
        "main_category": main_cat,
        "subcategory": shock_cat,
        "severity_score": severity_score,
        "severity_level": severity_level,
        "impact_horizon": fm.get("impact_horizon", ""),
        "macro_region": macro_region,
        "micro_region": micro_region,
        "industries": "",  # пока не извлекаем
        "shock_summary": what[:300],
        "actual_outcome_summary": "",
        "full_text": full_text,
        "what_text": what or title,
    }


def _connect(db_path: Path):
    """Открывает БД и пытается загрузить sqlite_vec. Возвращает (conn, vec_loaded)."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    try:
        import sqlite_vec
        sqlite_vec.load(conn)
        vec_loaded = True
    except Exception:
        vec_loaded = False
    return conn, vec_loaded


def _write_row(conn, embedder, data: dict, vec_loaded: bool) -> None:
    """UPSERT одной записи анализа + её эмбеддингов по file_path.

    Удаляет ТОЛЬКО совпадающую по file_path запись (если была), не трогая
    остальной корпус — в отличие от полного DELETE в index_all.
    """
    old = conn.execute(
        "SELECT id FROM news_analyses WHERE file_path = ?", (data["file_path"],)
    ).fetchone()
    if old is not None:
        if vec_loaded:
            conn.execute("DELETE FROM news_embeddings WHERE news_id = ?", (old["id"],))
        conn.execute("DELETE FROM news_analyses WHERE id = ?", (old["id"],))

    cursor = conn.execute("""
        INSERT INTO news_analyses
        (file_path, date, title, main_category, subcategory, severity_score,
         severity_level, impact_horizon, macro_region, micro_region,
         industries, shock_summary, actual_outcome_summary)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data["file_path"], data["date"], data["title"],
        data["main_category"], data["subcategory"], data["severity_score"],
        data["severity_level"], data["impact_horizon"],
        data["macro_region"], data["micro_region"],
        data["industries"], data["shock_summary"], data["actual_outcome_summary"]
    ))
    news_id = cursor.lastrowid

    if vec_loaded:
        title_emb = embedder.encode(data["title"])
        what_emb = embedder.encode(data["what_text"])
        conn.execute("""
            INSERT INTO news_embeddings (news_id, title_embedding, what_embedding)
            VALUES (?, ?, ?)
        """, (news_id, title_emb.tobytes(), what_emb.tobytes()))


def _load_corpus_texts(conn) -> list[str]:
    """Тексты существующего корпуса (title + shock_summary) для fit embedder."""
    texts = []
    for row in conn.execute("SELECT title, shock_summary FROM news_analyses"):
        if row["title"]:
            texts.append(row["title"])
        if row["shock_summary"]:
            texts.append(row["shock_summary"])
    return texts


def index_single(file_path, db_path: Path = DB_PATH,
                 use_st: bool | None = None) -> bool:
    """Инкрементально индексирует ОДИН файл анализа (S1.2): UPSERT без стирания БД.

    Embedder обучается на полном корпусе (существующий + новый док), затем
    считается эмбеддинг нового дока. Эмбеддинги старых записей не пересчитываются —
    приемлемый компромисс для TF-IDF (find_analogs всё равно re-fit при запросе);
    после перехода на нейроэмбеддинги (S3.3) вопрос снимается.
    """
    from embeddings import get_embedder

    file_path = Path(file_path)
    if not file_path.exists():
        print(f"  ❌ Файл не найден: {file_path}")
        return False

    from embeddings import RAG_USE_ST
    if use_st is None:
        use_st = RAG_USE_ST

    data = parse_markdown_analysis(file_path)
    conn, vec_loaded = _connect(db_path)
    try:
        embedder = get_embedder(prefer_st=use_st)
        corpus = _load_corpus_texts(conn)
        corpus += [data["title"], data["what_text"]]
        embedder.fit(corpus)
        _write_row(conn, embedder, data, vec_loaded)
        conn.commit()
    finally:
        conn.close()
    print(f"  ✅ Проиндексирован 1 файл (incremental): {file_path.name}")
    return True


def index_all(db_path: Path = DB_PATH, analyses_dir: Path = ANALYSES_DIR,
                use_st: bool | None = None) -> int:
    """Полный реиндекс всех .md в _Анализы/ (с очисткой БД). Возвращает количество."""
    from embeddings import get_embedder, RAG_USE_ST
    if use_st is None:
        use_st = RAG_USE_ST

    if not analyses_dir.exists():
        print(f"  ❌ Папка анализов не найдена: {analyses_dir}")
        return 0

    md_files = sorted(analyses_dir.glob("*.md"))
    print(f"  Найдено {len(md_files)} файлов анализов")

    parsed = []
    for f in md_files:
        try:
            parsed.append(parse_markdown_analysis(f))
        except Exception as e:
            print(f"  ⚠️ Ошибка парсинга {f.name}: {e}")

    if not parsed:
        return 0

    print(f"\n  Инициализация embedder...")
    embedder = get_embedder(prefer_st=use_st)
    corpus_texts = []
    for p in parsed:
        corpus_texts.append(p["title"])
        corpus_texts.append(p["what_text"])
    embedder.fit(corpus_texts)

    conn, vec_loaded = _connect(db_path)
    indexed = 0
    try:
        # Полная очистка — только для full reindex (index_single этого не делает).
        conn.execute("DELETE FROM news_analyses")
        if vec_loaded:
            conn.execute("DELETE FROM news_embeddings")
        for data in parsed:
            _write_row(conn, embedder, data, vec_loaded)
            indexed += 1
            print(f"  [{indexed}/{len(parsed)}] {data['date']} | {data['title'][:60]}...")
        conn.commit()
    finally:
        conn.close()
    print(f"\n  ✅ Индексировано {indexed} анализов")
    return indexed


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="RAG — индексация анализов")
    parser.add_argument("--file", help="Инкрементально индексировать один файл (UPSERT, без стирания БД)")
    parser.add_argument("--use-st", action="store_const", const=True, default=None,
                        help="Использовать sentence-transformers вместо TF-IDF "
                             "(по умолчанию — из env RADAR_RAG_USE_ST)")
    args = parser.parse_args()

    print("=" * 70)
    if args.file:
        print(f"  RAG — incremental index: {args.file}")
        print("=" * 70)
        ok = index_single(args.file, use_st=args.use_st)
        sys.exit(0 if ok else 1)
    else:
        print(f"  RAG — full reindex from _Анализы/")
        print("=" * 70)
        n = index_all(use_st=args.use_st)
        print(f"\n  Total: {n} files indexed")
