---
type: rag_architecture
version: 1.0
date: 2026-04-25
---

# RAG Template — News Embeddings + Historical Lookup

> [[README|← README]] · [[orchestrator|← Orchestrator]]

Архитектура RAG-системы для **Agent 3 (Backtest-Analog)** в multi-agent pipeline. Цель — найти **исторические аналоги** текущей новости через cosine similarity на news embeddings.

**Статус (v0.9, июнь 2026):** ✅ реализовано в `agents/rag/*.py` (`init_db`, `embeddings`, `find_analogs`, `index_news`). Этот документ — спецификация; ниже отмечены отличия фактической реализации от первоначального плана (эмбеддер по умолчанию TF-IDF, dim 384).

---

## Зачем нужен RAG

Текущий Agent 3 (v1.0) ищет аналоги через семантический поиск Claude по файлам в `_Анализы/`. Это работает для 5-10 анализов; для **сотен анализов** нужно **векторное хранилище**.

При накоплении 100+ анализов (через 6-12 месяцев) — Claude не сможет за разумное время прочитать все. RAG решает: эмбеддинги хранятся в БД, поиск top-K за миллисекунды.

---

## Архитектура (3 уровня)

```
┌──────────────────────────────────────────────────────┐
│ Level 1 — Embedding Generation                       │
│                                                       │
│ Каждый сохранённый анализ из _Анализы/ получает      │
│ embedding-вектор для:                                 │
│   - Заголовок новости                                 │
│   - L0.WHAT (краткое описание)                       │
│   - L0.subcategory (классификатор)                   │
│   - Полный текст анализа (для контекста)             │
└────────────────────┬─────────────────────────────────┘
                     │ vectors (768d или 1024d)
                     ▼
┌──────────────────────────────────────────────────────┐
│ Level 2 — Vector Storage                              │
│                                                       │
│ SQLite + sqlite-vec / pgvector / FAISS / ChromaDB    │
│   - news_id (PK)                                      │
│   - date                                              │
│   - main_category, subcategory                       │
│   - severity_score                                    │
│   - region, industries                                │
│   - title_embedding [768d]                           │
│   - what_embedding [768d]                            │
│   - full_embedding [768d]                            │
│   - actual_outcome_json (фактический исход)          │
└────────────────────┬─────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────┐
│ Level 3 — Similarity Search                           │
│                                                       │
│ При новой новости:                                    │
│   1. Генерим embedding                                │
│   2. Cosine similarity по what_embedding              │
│   3. Filters: subcategory, region, severity range    │
│   4. Top-K (обычно K=3-5) аналогов                   │
│   5. Threshold ≥0.75 для «хорошего аналога»          │
└──────────────────────────────────────────────────────┘
```

---

## Выбор embedding-модели

| Модель | Параметры | Размер | Преимущества | Недостатки |
|---|---|---|---|---|
| **FinBERT** (от ProsusAI) | 110M | 768d | Open-source, специализирован на финтекстах, **+15.6% над BERT** на FinMTEB | Английский preferred; для русских текстов нужен RU-FinBERT или multilingual |
| **Fin-E5** (от FinanceFactor) | 7B | 4096d | Best-in-class на финансовых текстах (+4.5% над e5-mistral); finetuned на FinMTEB | 7B параметров — heavier inference |
| **Anthropic Claude API embeddings** | proprietary | varies | Качественно для общих текстов; native интеграция с Claude | Не специализирован под финтексты; платный API |
| **OpenAI text-embedding-3-large** | proprietary | 3072d | Очень высокое качество; multilingual | Платный API; vendor lock-in |
| **multilingual-e5-large** | 560M | 1024d | Open-source, **поддерживает русский**, мощный | Не специализирован под финансы |

### Реализованный выбор (v0.9, июнь 2026)

**По умолчанию:** TF-IDF-эмбеддер (sklearn `TfidfVectorizer` + усечение/паддинг до 384d) — всегда доступен, без тяжёлых моделей и без сети.

**Опционально:** нейросетевой `intfloat/multilingual-e5-small` (open-source, RU-поддержка, **384d** — совпадает с размерностью БД) — включается через env `RADAR_RAG_USE_ST=1` + reindex, с graceful fallback на TF-IDF при отсутствии `sentence-transformers`.

**Опциональные кандидаты (не дефолт):** FinBERT, Fin-E5, `multilingual-e5-large` (1024d, требует другой dim для vec0). Изначально планировался `multilingual-e5-large` + FinBERT — заменено на e5-small/TF-IDF ради лёгкости инференса и совпадения размерности.

---

## Схема БД (SQLite + sqlite-vec)

```sql
CREATE TABLE news_analyses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT UNIQUE NOT NULL,         -- путь к md-файлу анализа
    date TEXT NOT NULL,                      -- YYYY-MM-DD
    title TEXT NOT NULL,
    main_category TEXT NOT NULL,             -- "1.1", "2.5", и т.п.
    subcategory TEXT NOT NULL,
    severity_score INTEGER,                  -- 0-100
    severity_level TEXT,                     -- "L"|"M"|"H"
    impact_horizon TEXT,
    region TEXT,
    industries_json TEXT,                    -- ["Нефтегаз", "Металлургия"]
    actual_outcome_json TEXT,                -- если уже есть факт-проверка
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- sqlite-vec extension для embedding
CREATE VIRTUAL TABLE news_embeddings USING vec0(
    news_id INTEGER PRIMARY KEY,
    title_embedding FLOAT[768],
    what_embedding FLOAT[768],
    full_embedding FLOAT[768]
);

-- Индексы для фильтрации
CREATE INDEX idx_subcategory ON news_analyses(subcategory);
CREATE INDEX idx_region ON news_analyses(region);
CREATE INDEX idx_date ON news_analyses(date);
CREATE INDEX idx_severity ON news_analyses(severity_score);
```

---

## Pipeline индексирования (план v1.1)

```python
# index_news.py — индексация всех анализов из _Анализы/

import sqlite3
from pathlib import Path
from sentence_transformers import SentenceTransformer
import yaml
import json

# Загрузить модель эмбеддингов
model = SentenceTransformer("intfloat/multilingual-e5-large")

# Подключиться к БД
conn = sqlite3.connect("radar_rag.db")
conn.enable_load_extension(True)
conn.load_extension("vec0")

# Пройти по всем анализам
analyses_dir = Path("Макро-радар/_Анализы")
for md_file in analyses_dir.glob("*.md"):
    # Парсить frontmatter
    text = md_file.read_text(encoding="utf-8")
    fm_end = text.find("---", 3)
    frontmatter = yaml.safe_load(text[3:fm_end])

    # Извлечь поля
    main_cat = frontmatter.get("шок_категория", "")
    severity = frontmatter.get("сила_шока", "")
    date = frontmatter.get("дата_новости", "")

    # Извлечь сегменты текста для эмбеддинга
    title = md_file.stem
    what_section = extract_section(text, "## L0 — Классификация")
    full_text = text

    # Сгенерировать эмбеддинги
    title_emb = model.encode(title)
    what_emb = model.encode(what_section)
    full_emb = model.encode(full_text)

    # Сохранить в БД
    conn.execute("""INSERT INTO news_analyses
        (file_path, date, title, main_category, ...) VALUES (?, ?, ?, ?, ...)""",
        (str(md_file), date, title, main_cat, ...))
    news_id = conn.lastrowid

    conn.execute("""INSERT INTO news_embeddings
        (news_id, title_embedding, what_embedding, full_embedding)
        VALUES (?, ?, ?, ?)""",
        (news_id, title_emb.tolist(), what_emb.tolist(), full_emb.tolist()))

conn.commit()
```

---

## Pipeline поиска аналогов (план v1.1)

```python
# find_analogs.py — для Agent 3

def find_analogs(news_what: str, subcategory: str, region: str = None,
                 severity_score: int = None, top_k: int = 5,
                 threshold: float = 0.75) -> list:
    """
    Найти top_k исторических аналогов через cosine similarity.
    Фильтр по subcategory + region + severity range.
    """
    query_emb = model.encode(news_what)

    # Cosine similarity через sqlite-vec
    cursor = conn.execute("""
        SELECT n.id, n.title, n.date, n.main_category, n.actual_outcome_json,
               vec_distance_cosine(e.what_embedding, ?) as distance
        FROM news_embeddings e
        JOIN news_analyses n ON e.news_id = n.id
        WHERE n.subcategory = ?
          AND (n.region = ? OR ? IS NULL)
        ORDER BY distance ASC
        LIMIT ?
    """, (query_emb.tolist(), subcategory, region, region, top_k))

    results = []
    for row in cursor:
        similarity = 1 - row['distance']  # cosine distance → similarity
        if similarity >= threshold:
            results.append({
                "title": row['title'],
                "date": row['date'],
                "similarity": similarity,
                "main_category": row['main_category'],
                "actual_outcome": json.loads(row['actual_outcome_json'])
            })

    return results
```

---

## Установка зависимостей (план v1.1)

```bash
pip install sentence-transformers
pip install sqlite-vec
pip install pyyaml
pip install pydantic
```

Модели (~3-5 GB на диск):
- `intfloat/multilingual-e5-large` — 2.2 GB
- (опционально) FinBERT — 440 MB

---

## Метрики качества RAG (для калибровки)

При запуске v1.1 — собрать через:

| Метрика | Значение цели |
|---|---|
| **Precision@5** | ≥80% (5 из 5 top результатов должны быть релевантными) |
| **Recall@10** | ≥90% (10 из 10 ground-truth аналогов в top-10) |
| **MRR** (Mean Reciprocal Rank) | ≥0.6 |
| **Время поиска** | ≤200 мс на запрос |

**Ground truth:** ручная разметка аналогов на 20 тестовых кейсах из истории Радара.

---

## Связь с D.1 — Multi-Agent Pipeline

В **v1.0** Agent 3 ищет аналоги через семантический поиск Claude по файлам — медленно, но работает на малых объёмах.

В **v1.1** Agent 3 заменит свою логику на:
1. Вызов `find_analogs()` через RAG-БД
2. Получение top-K результатов с фактическими outcome'ами
3. Передача Agent 4 в виде структурированного `historical_analogs` JSON

Это сократит время Agent 3 с **30-60 сек** до **<1 сек** при ≥100 анализов в БД.

---

## Roadmap

| Этап | Срок | Что |
|---|---|---|
| v1.0 | 2026-04-25 | Архитектура зафиксирована, ручной поиск Claude |
| v1.1 | ✅ реализовано в v0.9 (июнь 2026) | SQLite+sqlite-vec, эмбеддинги (TF-IDF по умолчанию, опц. e5-small 384d), индексация исторических анализов (`index_single` UPSERT + `index_all`) |
| v1.2 (опция) | впереди | FinBERT / Fin-E5 для финансовых текстов; гибридный rerank |
| v2.0 (план) | впереди | Расширение на новости из открытых источников (Reuters/ТАСС) — не только наши анализы |
| v2.1 (план) | впереди | Cross-encoder rerank top-50 → top-5 для качества |

---

*RAG Template v1.0 · 2026-04-25 · Базируется на FinMTEB (arXiv:2502.10990) и News Sentiment Embeddings (arXiv:2507.01970)*
