"""
RAG v1.1 — поиск исторических аналогов через cosine similarity.

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


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def find_analogs(
    query_text: str,
    subcategory: str | None = None,
    macro_region: str | None = None,
    severity_min: int | None = None,
    severity_max: int | None = None,
    top_k: int = 5,
    threshold: float = 0.30,
    db_path: Path = DB_PATH,
    use_st: bool | None = None,
) -> list[dict]:
    """
    Найти top_k исторических аналогов через cosine similarity.

    Args:
      query_text: текст текущей новости
      subcategory: фильтр по подкатегории шока (например "1.3")
      macro_region: фильтр по макро-региону (например "SOUTH_CAUCASUS")
      severity_min/max: диапазон силы шока
      top_k: количество результатов
      threshold: минимальный косинус (0.0-1.0); для TF-IDF реальные значения 0.2-0.5

    Returns:
      Список dicts с file_path, title, date, similarity, ...
    """
    from embeddings import get_embedder, RAG_USE_ST

    # S3.3: единый выбор эмбеддера (env RADAR_RAG_USE_ST), согласован с индексацией
    use = RAG_USE_ST if use_st is None else use_st
    embedder = get_embedder(prefer_st=use)

    # Re-fit на корпусе (TF-IDF требует весь корпус)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Загрузить корпус
    cursor = conn.execute("SELECT title, shock_summary FROM news_analyses")
    corpus = []
    for row in cursor:
        corpus.append(row["title"])
        if row["shock_summary"]:
            corpus.append(row["shock_summary"])

    if len(corpus) > 0:
        embedder.fit(corpus)

    query_emb = embedder.encode(query_text)

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

    where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

    if vec_loaded:
        # Не используем vec_distance в WHERE из-за фильтров — делаем просто SELECT всех + cosine в Python
        cursor = conn.execute(
            f"""
            SELECT n.id, n.file_path, n.date, n.title, n.subcategory,
                   n.severity_score, n.severity_level, n.macro_region, n.micro_region,
                   n.shock_summary,
                   e.what_embedding, e.title_embedding
            FROM news_analyses n
            JOIN news_embeddings e ON n.id = e.news_id
            {where_sql}
        """,
            params,
        )

        results = []
        for row in cursor:
            what_emb = np.frombuffer(row["what_embedding"], dtype=np.float32)
            title_emb = np.frombuffer(row["title_embedding"], dtype=np.float32)
            sim_what = cosine_similarity(query_emb, what_emb)
            sim_title = cosine_similarity(query_emb, title_emb)
            sim = max(sim_what, sim_title)
            if sim >= threshold:
                results.append(
                    {
                        "file_path": row["file_path"],
                        "date": row["date"],
                        "title": row["title"],
                        "subcategory": row["subcategory"],
                        "severity_score": row["severity_score"],
                        "severity_level": row["severity_level"],
                        "macro_region": row["macro_region"],
                        "micro_region": row["micro_region"],
                        "shock_summary": row["shock_summary"],
                        "similarity": sim,
                        "similarity_what": sim_what,
                        "similarity_title": sim_title,
                    }
                )
    else:
        cursor = conn.execute(
            f"""
            SELECT id, file_path, date, title, subcategory, severity_score,
                   severity_level, macro_region, micro_region, shock_summary
            FROM news_analyses {where_sql}
        """,
            params,
        )
        results = []
        for row in cursor:
            results.append({**dict(row), "similarity": 0.0})

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
    parser = argparse.ArgumentParser(description="RAG v1.1 — Find Analogs")
    parser.add_argument("query", help="Текст новой новости для поиска аналогов")
    parser.add_argument("--subcategory", help="Фильтр по подкатегории шока (например 1.3)")
    parser.add_argument("--region", help="Фильтр по макро-региону")
    parser.add_argument("--severity-min", type=int, help="Мин. severity_score")
    parser.add_argument("--severity-max", type=int, help="Макс. severity_score")
    parser.add_argument("--top-k", type=int, default=5, help="Количество результатов")
    parser.add_argument("--threshold", type=float, default=0.20, help="Мин. cosine similarity")
    args = parser.parse_args()

    print("=" * 70)
    print("  RAG v1.1 — Search for Analogs")
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
