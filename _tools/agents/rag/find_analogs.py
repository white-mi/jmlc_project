"""
RAG — поиск исторических аналогов через cosine similarity.

Используется Agent 3 для подмешивания контекста при анализе новой новости.
"""

import sqlite3
import sys
import argparse
import numpy as np
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DB_PATH = Path(__file__).parent / "radar_rag.db"

# Типы документов, которые вообще могут быть «историческим аналогом шока».
# `devlog` (журналы разработки радара) исключён: это записи о ходе разработки,
# а не разборы событий — в выдаче они дают чистый шум.
RETRIEVABLE_DOC_TYPES = ("news", "retro", "digest")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _score_row(row, query_emb: np.ndarray, threshold: float) -> dict | None:
    """Косинус запроса к title/what-эмбеддингам строки → dict аналога или None (< threshold).

    Единый код для vec0 и BLOB-fallback: обе таблицы хранят float32-эмбеддинги.
    Если у документа не нашлось отдельного WHAT (`has_what=0`), what-вектор равен title-вектору
    и вторым независимым сигналом не является — тогда учитываем только title (без двойного счёта).
    """
    keys = row.keys()
    has_what = bool(row["has_what"]) if "has_what" in keys else True
    title_emb = np.frombuffer(row["title_embedding"], dtype=np.float32)
    sim_title = cosine_similarity(query_emb, title_emb)
    if has_what:
        what_emb = np.frombuffer(row["what_embedding"], dtype=np.float32)
        sim_what = cosine_similarity(query_emb, what_emb)
    else:
        sim_what = sim_title
    sim = max(sim_what, sim_title)
    if sim < threshold:
        return None
    return {
        "file_path": row["file_path"],
        "date": row["date"],
        "title": row["title"],
        "subcategory": row["subcategory"],
        "severity_score": row["severity_score"],
        "severity_level": row["severity_level"],
        "macro_region": row["macro_region"],
        "micro_region": row["micro_region"],
        "shock_summary": row["shock_summary"],
        "doc_type": row["doc_type"] if "doc_type" in keys else "news",
        "has_what": has_what,
        "similarity": sim,
        "similarity_what": sim_what,
        "similarity_title": sim_title,
    }


def find_analogs(
    query_text: str,
    subcategory: str | None = None,
    macro_region: str | None = None,
    severity_min: int | None = None,
    severity_max: int | None = None,
    top_k: int = 5,
    threshold: float | None = None,
    db_path: Path = DB_PATH,
    use_st: bool | None = None,
    doc_types: tuple[str, ...] | None = RETRIEVABLE_DOC_TYPES,
) -> list[dict]:
    """
    Найти top_k исторических аналогов через cosine similarity.

    Args:
      query_text: текст текущей новости
      subcategory: фильтр по подкатегории шока (например "1.3")
      macro_region: фильтр по макро-региону (например "SOUTH_CAUCASUS")
      severity_min/max: диапазон силы шока
      top_k: количество результатов
      threshold: минимальный косинус (0.0-1.0). None → берётся из пространства эмбеддингов
        (TF-IDF 0.15, e5 0.80): у e5 косинусы сжаты в узкий диапазон, и порог TF-IDF там
        пропустил бы вообще всё
      doc_types: какие типы документов допускать в выдачу (None — вообще без фильтра)

    Returns:
      Список dicts с file_path, title, date, similarity, ...
    """
    from embeddings import (
        RAG_USE_ST,
        default_threshold,
        embedder_name,
        get_embedder,
        load_embedder,
    )
    from index_news import _embedder_path

    # единый выбор эмбеддера (env RADAR_RAG_USE_ST), согласован с индексацией
    use = RAG_USE_ST if use_st is None else use_st

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Единый базис: грузим персистнутый при индексации эмбеддер (index_all/index_single),
    # чтобы запрос кодировался в ТОМ ЖЕ SVD-пространстве, что и БД (косинусы сопоставимы
    # только в одном базисе).
    embedder = load_embedder(_embedder_path(db_path))
    if embedder is None or not getattr(embedder, "fitted", True):
        # Легаси-БД без персиста (или ST-режим) → откат на re-fit по корпусу БД.
        embedder = get_embedder(prefer_st=use)
        cur = conn.execute("SELECT title, shock_summary FROM news_analyses")
        corpus = []
        for row in cur:
            corpus.append(row["title"])
            if row["shock_summary"]:
                corpus.append(row["shock_summary"])
        if corpus:
            embedder.fit(corpus)

    # Гард пространств: БД, проиндексированная TF-IDF, не ищется e5-запросом и наоборот —
    # косинусы несопоставимы, а выдача выглядела бы «правдоподобно», просто будучи мусором.
    name = embedder_name(embedder)
    try:
        stored = conn.execute("SELECT value FROM db_meta WHERE key = 'embedder'").fetchone()
    except sqlite3.Error:
        stored = None
    if stored is not None and stored[0] != name:
        conn.close()
        raise RuntimeError(
            f"RAG: БД проиндексирована эмбеддером '{stored[0]}', а запрос кодируется '{name}'. "
            f"Косинусы из разных пространств несопоставимы — нужен полный реиндекс: "
            f"RADAR_RAG_USE_ST={'1' if name != 'tfidf' else '0'} python index_news.py"
        )

    if threshold is None:
        threshold = default_threshold(name)

    query_emb = embedder.encode(query_text, is_query=True)

    # Загрузить эмбеддинги + метаданные
    conn.enable_load_extension(True)
    try:
        import sqlite_vec

        sqlite_vec.load(conn)
        vec_loaded = True
    except Exception:
        vec_loaded = False

    # Фильтры
    where_clauses = []
    params = []
    if subcategory:
        where_clauses.append("n.subcategory LIKE ?")
        params.append(f"%{subcategory}%")
    if macro_region:
        where_clauses.append("n.macro_region = ?")
        params.append(macro_region)
    if severity_min is not None:
        where_clauses.append("n.severity_score >= ?")
        params.append(severity_min)
    if severity_max is not None:
        where_clauses.append("n.severity_score <= ?")
        params.append(severity_max)
    if doc_types:
        # COALESCE — для строк из БД, созданных до появления колонки doc_type.
        where_clauses.append(f"COALESCE(n.doc_type, 'news') IN ({','.join('?' * len(doc_types))})")
        params.extend(doc_types)

    where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

    # Единая обработка vec0 и BLOB-fallback: обе таблицы хранят float32-эмбеддинги, cosine
    # считаем в Python. Ранжирование по запросу и threshold применяются в обеих ветках
    # одинаково: fallback-ветка (без sqlite-vec — дефолтная установка!) иначе отдавала бы
    # строки с фейковым similarity=0.0, без сортировки и без порога.
    emb_table = "news_embeddings" if vec_loaded else "news_embeddings_fallback"
    has_emb = conn.execute("SELECT 1 FROM sqlite_master WHERE name = ?", (emb_table,)).fetchone()

    results = []
    if has_emb:
        cursor = conn.execute(
            f"""
            SELECT n.id, n.file_path, n.date, n.title, n.subcategory,
                   n.severity_score, n.severity_level, n.macro_region, n.micro_region,
                   n.shock_summary, COALESCE(n.doc_type, 'news') AS doc_type,
                   COALESCE(n.has_what, 1) AS has_what,
                   e.what_embedding, e.title_embedding
            FROM news_analyses n
            JOIN {emb_table} e ON n.id = e.news_id
            {where_sql}
        """,
            params,
        )
        for row in cursor:
            packed = _score_row(row, query_emb, threshold)
            if packed is not None:
                results.append(packed)

    conn.close()

    # Sort by similarity
    results.sort(key=lambda r: r["similarity"], reverse=True)
    return results[:top_k]


def format_analogs(analogs: list[dict]) -> str:
    """Форматировать список аналогов в человекочитаемый вид."""
    if not analogs:
        return "  Аналогов не найдено."
    lines = []
    for i, a in enumerate(analogs, 1):
        lines.append(f"  [{i}] {a['date']} | sim={a['similarity']:.3f}")
        lines.append(f"      {a['title'][:90]}")
        lines.append(
            f"      Категория: {a.get('subcategory') or 'н/д'} | "
            f"Сила: {a.get('severity_score') or '?'}/{a.get('severity_level') or '?'}"
        )
        if a.get("macro_region"):
            lines.append(f"      Регион: {a['macro_region']}/{a.get('micro_region') or '—'}")
        if a.get("shock_summary"):
            lines.append(f"      WHAT: {a['shock_summary'][:120]}...")
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="RAG — Find Analogs")
    parser.add_argument("query", help="Текст новой новости для поиска аналогов")
    parser.add_argument("--subcategory", help="Фильтр по подкатегории шока (например 1.3)")
    parser.add_argument("--region", help="Фильтр по макро-региону")
    parser.add_argument("--severity-min", type=int, help="Мин. severity_score")
    parser.add_argument("--severity-max", type=int, help="Макс. severity_score")
    parser.add_argument("--top-k", type=int, default=5, help="Количество результатов")
    parser.add_argument("--threshold", type=float, default=0.20, help="Мин. cosine similarity")
    args = parser.parse_args()

    print("=" * 70)
    print("  RAG — Search for Analogs")
    print("=" * 70)
    print(f"  Query: {args.query[:80]}")
    print(
        f"  Filters: subcat={args.subcategory}, region={args.region}, sev=[{args.severity_min},{args.severity_max}]"
    )
    print(f"  Top-K={args.top_k}, threshold={args.threshold}")
    print()

    results = find_analogs(
        args.query,
        subcategory=args.subcategory,
        macro_region=args.region,
        severity_min=args.severity_min,
        severity_max=args.severity_max,
        top_k=args.top_k,
        threshold=args.threshold,
    )

    print(format_analogs(results))


if __name__ == "__main__":
    main()
