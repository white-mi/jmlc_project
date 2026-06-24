"""
RAG v1.1 — инициализация БД для хранения news embeddings и метаданных.

Создаёт SQLite-БД с расширением sqlite-vec для cosine similarity поиска.
Если sqlite-vec не подгружается — fallback на чистый SQLite + Python cosine.
"""

import sqlite3
import sys
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

DB_PATH = Path(__file__).parent / "radar_rag.db"


def init_db(db_path: Path = DB_PATH, embedding_dim: int = 384) -> sqlite3.Connection:
    """
    Инициализировать БД.
    embedding_dim:
      - 384 для multilingual-e5-small / paraphrase-multilingual-MiniLM-L12-v2
      - 1024 для multilingual-e5-large
      - 768 для FinBERT
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Попытка загрузить sqlite-vec
    vec_loaded = False
    try:
        conn.enable_load_extension(True)
        import sqlite_vec
        sqlite_vec.load(conn)
        vec_loaded = True
        print("  ✅ sqlite-vec loaded successfully")
    except Exception as e:
        print(f"  ⚠️ sqlite-vec не загрузился: {e}")
        print("  → fallback на чистый SQLite + numpy для cosine")

    # Основная таблица анализов
    conn.execute("""
        CREATE TABLE IF NOT EXISTS news_analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT UNIQUE NOT NULL,
            date TEXT NOT NULL,
            title TEXT NOT NULL,
            main_category TEXT,
            subcategory TEXT,
            severity_score INTEGER,
            severity_level TEXT,
            impact_horizon TEXT,
            macro_region TEXT,
            micro_region TEXT,
            industries TEXT,
            shock_summary TEXT,
            actual_outcome_summary TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Векторная таблица (если sqlite-vec доступен)
    if vec_loaded:
        try:
            conn.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS news_embeddings USING vec0(
                    news_id INTEGER PRIMARY KEY,
                    title_embedding FLOAT[{embedding_dim}],
                    what_embedding FLOAT[{embedding_dim}]
                )
            """)
            print(f"  ✅ Vector table created (dim={embedding_dim})")
        except Exception as e:
            print(f"  ⚠️ vec0 table failed: {e}; используем blob fallback")
            vec_loaded = False

    if not vec_loaded:
        # Fallback: храним эмбеддинги как BLOB
        conn.execute("""
            CREATE TABLE IF NOT EXISTS news_embeddings_fallback (
                news_id INTEGER PRIMARY KEY,
                title_embedding BLOB,
                what_embedding BLOB,
                FOREIGN KEY (news_id) REFERENCES news_analyses(id)
            )
        """)

    # Индексы
    conn.execute("CREATE INDEX IF NOT EXISTS idx_subcategory ON news_analyses(subcategory)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_macro_region ON news_analyses(macro_region)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_date ON news_analyses(date)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_severity ON news_analyses(severity_score)")

    # Сохраняем флаг наличия sqlite-vec
    conn.execute("CREATE TABLE IF NOT EXISTS db_meta (key TEXT PRIMARY KEY, value TEXT)")
    conn.execute("INSERT OR REPLACE INTO db_meta VALUES ('vec_loaded', ?)", (str(vec_loaded),))
    conn.execute("INSERT OR REPLACE INTO db_meta VALUES ('embedding_dim', ?)", (str(embedding_dim),))

    conn.commit()
    print(f"  ✅ Database initialized at: {db_path}")
    return conn


if __name__ == '__main__':
    print("=" * 60)
    print("  RAG v1.1 — DB initialization")
    print("=" * 60)
    conn = init_db()

    # Показать что есть
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor]
    print(f"\n  Tables created: {tables}")

    cursor = conn.execute("SELECT key, value FROM db_meta")
    print("\n  Meta:")
    for row in cursor:
        print(f"    {row[0]}: {row[1]}")
    conn.close()
