"""
RAG — индексация анализов из _Анализы/ в БД.

Парсит markdown-файлы с frontmatter, генерирует embeddings, сохраняет в БД.
"""

import sqlite3
import sys
import re
import argparse
import yaml
import numpy as np
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ANALYSES_DIR = Path(__file__).parent.parent.parent.parent / "_Анализы"
DB_PATH = Path(__file__).parent / "radar_rag.db"

# Подпапки корпуса: индексируем разборы событий (корень + _история), но НЕ журналы
# разработки и не батч-выгрузки — они не являются «историческими аналогами шока».
INDEXED_SUBDIRS = ("_история",)
EXCLUDED_SUBDIRS = ("_журнал", "_batch")

# Размеры текстовых фрагментов (см. docs/EVAL.md — «стратегия чанкинга»). Ретрив-единица
# здесь — весь разбор события, поэтому это не sliding-window чанкинг, а извлечение полей;
# обрезка делается по границе предложения, а не по символу.
WHAT_MAX_CHARS = 700
SUMMARY_MAX_CHARS = 300

# BLOB-таблица эмбеддингов для режима без sqlite-vec (numpy-cosine fallback).
FALLBACK_EMB_DDL = """
    CREATE TABLE IF NOT EXISTS news_embeddings_fallback (
        news_id INTEGER PRIMARY KEY,
        title_embedding BLOB,
        what_embedding BLOB,
        FOREIGN KEY (news_id) REFERENCES news_analyses(id)
    )
"""


def _embedder_path(db_path):
    """Путь к персистнутому TF-IDF-эмбеддеру рядом с БД (общий базис index↔query)."""
    return Path(str(db_path) + ".embedder.joblib")


def _clean_markdown(s: str) -> str:
    """Снять разметку, которая мешает эмбеддингу: пайпы таблиц, разделители, жирный,
    вики-ссылки, сноски. Boilerplate шапок таблиц («Параметр | Значение») одинаков во всех
    разборах и искусственно сближает их векторы — поэтому чистим до кодирования."""
    s = re.sub(r"^\s*\|?\s*-{2,}[\s|:-]*\|?\s*$", " ", s, flags=re.MULTILINE)  # |---|---|
    s = re.sub(r"\[\[([^\]|]+)\|?[^\]]*\]\]", r"\1", s)  # [[ссылка|текст]] → ссылка
    s = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", s)  # [текст](url) → текст
    s = re.sub(r"\[\^[^\]]+\]", " ", s)  # сноски
    s = s.replace("|", " ").replace("*", "").replace("`", "").replace(">", " ")
    return re.sub(r"\s+", " ", s).strip()


def _truncate_sentence(s: str, limit: int) -> str:
    """Обрезка по границе предложения, а не по символу (иначе режем середину таблицы/слова)."""
    if len(s) <= limit:
        return s
    head = s[:limit]
    cut = max(head.rfind(". "), head.rfind("! "), head.rfind("? "), head.rfind("; "))
    # откатываемся к границе, только если она не отрезает больше 40% фрагмента
    return head[: cut + 1].strip() if cut >= int(limit * 0.6) else head.rstrip()


def _section(text: str, header_re: str) -> str:
    """Тело секции по регэкспу её заголовка (до следующего ## или конца файла)."""
    m = re.search(rf"^##\s*{header_re}[^\n]*\n(.+?)(?=^##\s|\Z)", text, re.DOTALL | re.MULTILINE)
    return m.group(1).strip() if m else ""


def extract_what(text: str) -> str:
    """Каскад извлечения WHAT («что произошло») из разбора.

    Источники пробуются от самого точного к самому общему. Опора на единственный источник
    (секцию `## L0`) означала бы, что у документа без неё `what_text` равен заголовку,
    и оба «независимых» вектора документа совпадают буквально.
    """
    l0 = _section(text, r"L0")
    if l0:
        # 1) строка таблицы «Что произошло / Что случилось / Что» — самый точный источник
        row = re.search(
            r"^\s*\|\s*\**\s*(?:Что произошло|Что случилось|Что)\s*\**\s*\|(.+?)\|?\s*$",
            l0,
            re.MULTILINE | re.IGNORECASE,
        )
        if row:
            cleaned = _clean_markdown(row.group(1))
            if cleaned:
                return cleaned
    # 2) буллет «**WHAT:** …» / «**Что:** …»
    bullet = re.search(r"^[-*]?\s*\**\s*(?:WHAT|Что)\s*:\**\s*(.+)$", text, re.MULTILINE)
    if bullet:
        cleaned = _clean_markdown(bullet.group(1))
        if cleaned:
            return cleaned
    # 3) цитата новости в «## Источник» — это сам текст события, лучший материал для ретрива
    src = _section(text, r"Источник")
    if src:
        quote = "\n".join(ln for ln in src.splitlines() if ln.lstrip().startswith(">"))
        cleaned = _clean_markdown(quote)
        if len(cleaned) > 40:
            return cleaned
    # 4) TL;DR / Итог
    for header in (r"TL;?DR", r"Итог", r"Что было дальше"):
        sec = _section(text, header)
        if sec:
            cleaned = _clean_markdown(sec)
            if cleaned:
                return cleaned
    # 5) вся секция L0 без разметки (у части разборов это таблица классификации — слабый,
    #    но всё же непустой сигнал, лучше чем дубль заголовка)
    if l0:
        cleaned = _clean_markdown(l0)
        if cleaned:
            return cleaned
    # 6) первый содержательный абзац после H1
    body = re.sub(r"^#\s+[^\n]*\n", "", text, count=1, flags=re.MULTILINE)
    for para in re.split(r"\n\s*\n", body):
        p = para.strip()
        if not p or p.startswith(("#", ">", "|", "---", "![")):
            continue
        cleaned = _clean_markdown(p)
        if len(cleaned) > 40:
            return cleaned
    return ""


def detect_doc_type(fm: dict, title: str, file_path: Path, text: str = "") -> str:
    """news | retro | digest | devlog. Явный `doc_type` во frontmatter имеет приоритет.

    Журналы разработки («OSL v0.3 final», «Conformal v0.5») — не разборы событий и не должны
    возвращаться как «исторические аналоги шока»; ретрив по умолчанию их отсекает.
    """
    explicit = str(fm.get("doc_type") or "").strip().lower()
    if explicit in {"news", "retro", "digest", "devlog"}:
        return explicit
    if "_история" in file_path.parts:
        return "retro"
    if "дайджест" in title.lower() or "дайджест" in file_path.stem.lower():
        return "digest"
    if str(fm.get("шок_категория") or "").strip():
        return "news"
    # Секция «## L0 — Классификация» = документ прошёл классификацию шока → это разбор события,
    # даже если frontmatter неполный (часть ранних разборов писалась без `шок_категория`).
    if re.search(r"^##\s*L0\b", text, re.MULTILINE):
        return "news"
    return "devlog"


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
            text = text[fm_end + 5 :]

    # Заголовок (первый # ); служебный префикс «Анализ новости — » не несёт сигнала
    title_match = re.search(r"^# (.+?)$", text, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else file_path.stem
    title = re.sub(r"^Анализ новости\s*[—–-]\s*", "", title).strip()

    # WHAT — каскад источников (см. extract_what)
    what = _truncate_sentence(extract_what(text), WHAT_MAX_CHARS)

    # Извлекаем регион из заголовка / тегов
    macro_region = ""
    micro_region = ""
    for r in [
        "MSK_SPB",
        "DONOR",
        "INDUSTRIAL",
        "SOUTH_CAUCASUS",
        "FAR_EAST",
        "SIBERIA",
        "CENTRAL",
        "RURAL",
    ]:
        if r in text:
            macro_region = r
            break
    for r in [
        "TOURIST_RESORT",
        "URBAN_INDUSTRIAL",
        "MONOTOWN",
        "AGRICULTURAL_RURAL",
        "CAPITAL_DIVERSIFIED",
    ]:
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
        "date": str(
            fm.get(
                "дата_новости", file_path.stem.split(" — ")[0] if " — " in file_path.stem else ""
            )
        ),
        "title": title,
        "main_category": main_cat,
        "subcategory": shock_cat,
        "severity_score": severity_score,
        "severity_level": severity_level,
        "impact_horizon": fm.get("impact_horizon", ""),
        "macro_region": macro_region,
        "micro_region": micro_region,
        "industries": "",  # пока не извлекаем
        "shock_summary": _truncate_sentence(what, SUMMARY_MAX_CHARS),
        "actual_outcome_summary": "",
        "doc_type": detect_doc_type(fm, title, file_path, text),
        # has_what=0 → второго независимого вектора у документа нет: what_text падает
        # обратно на title, и векторы совпали бы буквально. Флаг фиксирует это явно,
        # и find_analogs не учитывает what-вектор для таких документов.
        "has_what": 1 if what and what.strip() != title.strip() else 0,
        "what_text": what or title,
    }


def iter_corpus_files(analyses_dir: Path) -> list[Path]:
    """Файлы корпуса для индексации: корень `_Анализы/` + `_история/` (ретро-разборы).

    `_журнал/` (журналы разработки) и `_batch/` (пакетные выгрузки) исключены: они не
    являются разборами событий, а как «исторические аналоги» только шумят.
    """
    files = [p for p in analyses_dir.glob("*.md")]
    for sub in INDEXED_SUBDIRS:
        files.extend((analyses_dir / sub).glob("*.md"))
    return sorted(f for f in files if not set(f.parts) & set(EXCLUDED_SUBDIRS))


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
    if not vec_loaded:
        conn.execute(FALLBACK_EMB_DDL)
    old = conn.execute(
        "SELECT id FROM news_analyses WHERE file_path = ?", (data["file_path"],)
    ).fetchone()
    if old is not None:
        if vec_loaded:
            conn.execute("DELETE FROM news_embeddings WHERE news_id = ?", (old["id"],))
        else:
            conn.execute("DELETE FROM news_embeddings_fallback WHERE news_id = ?", (old["id"],))
        conn.execute("DELETE FROM news_analyses WHERE id = ?", (old["id"],))

    cursor = conn.execute(
        """
        INSERT INTO news_analyses
        (file_path, date, title, main_category, subcategory, severity_score,
         severity_level, impact_horizon, macro_region, micro_region,
         industries, shock_summary, actual_outcome_summary, doc_type, has_what)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        (
            data["file_path"],
            data["date"],
            data["title"],
            data["main_category"],
            data["subcategory"],
            data["severity_score"],
            data["severity_level"],
            data["impact_horizon"],
            data["macro_region"],
            data["micro_region"],
            data["industries"],
            data["shock_summary"],
            data["actual_outcome_summary"],
            data.get("doc_type", "news"),
            data.get("has_what", 0),
        ),
    )
    news_id = cursor.lastrowid

    title_emb = np.asarray(embedder.encode(data["title"]), dtype=np.float32)
    what_emb = np.asarray(embedder.encode(data["what_text"]), dtype=np.float32)
    if vec_loaded:
        conn.execute(
            "INSERT INTO news_embeddings (news_id, title_embedding, what_embedding) "
            "VALUES (?, ?, ?)",
            (news_id, title_emb.tobytes(), what_emb.tobytes()),
        )
    else:
        conn.execute(
            "INSERT INTO news_embeddings_fallback (news_id, title_embedding, what_embedding) "
            "VALUES (?, ?, ?)",
            (news_id, title_emb.tobytes(), what_emb.tobytes()),
        )


def _record_embedder(conn, embedder) -> None:
    """Записать пространство эмбеддингов в `db_meta`.

    Без этой отметки БД, проиндексированная TF-IDF, молча искалась бы e5-запросом (и наоборот):
    косинусы из разных пространств несопоставимы, выдача превращалась бы в шум без единой ошибки.
    """
    from embeddings import embedder_name

    conn.execute("CREATE TABLE IF NOT EXISTS db_meta (key TEXT PRIMARY KEY, value TEXT)")
    conn.execute(
        "INSERT OR REPLACE INTO db_meta VALUES ('embedder', ?)", (embedder_name(embedder),)
    )


def _load_corpus_texts(conn) -> list[str]:
    """Тексты существующего корпуса (title + shock_summary) для fit embedder."""
    texts = []
    for row in conn.execute("SELECT title, shock_summary FROM news_analyses"):
        if row["title"]:
            texts.append(row["title"])
        if row["shock_summary"]:
            texts.append(row["shock_summary"])
    return texts


def index_single(file_path, db_path: Path = DB_PATH, use_st: bool | None = None) -> bool:
    """Инкрементально индексирует ОДИН файл анализа: UPSERT без стирания БД.

    Использует ПЕРСИСТНУТЫЙ эмбеддер (из index_all): новый док лишь трансформируется в
    существующем базисе → его вектор консистентен и с БД, и с запросом (find_analogs грузит
    тот же артефакт). Если персиста нет (легаси/ST/первый вызов) — фит на полном корпусе +
    персист. Новую лексику полностью подхватит следующий полный index_all.
    """
    from embeddings import get_embedder, load_embedder, save_embedder

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
        embedder = load_embedder(_embedder_path(db_path))
        if embedder is None or not getattr(embedder, "fitted", True):
            # Нет персистнутого базиса → фит на полном корпусе (существующий + новый) + персист.
            embedder = get_embedder(prefer_st=use_st)
            corpus = _load_corpus_texts(conn) + [data["title"], data["what_text"]]
            embedder.fit(corpus)
            save_embedder(embedder, _embedder_path(db_path))
        _write_row(conn, embedder, data, vec_loaded)
        _record_embedder(conn, embedder)
        conn.commit()
    finally:
        conn.close()
    print(f"  ✅ Проиндексирован 1 файл (incremental): {file_path.name}")
    return True


def index_all(
    db_path: Path = DB_PATH, analyses_dir: Path = ANALYSES_DIR, use_st: bool | None = None
) -> int:
    """Полный реиндекс всех .md в _Анализы/ (с очисткой БД). Возвращает количество."""
    from embeddings import get_embedder, RAG_USE_ST

    if use_st is None:
        use_st = RAG_USE_ST

    if not analyses_dir.exists():
        print(f"  ❌ Папка анализов не найдена: {analyses_dir}")
        return 0

    md_files = iter_corpus_files(analyses_dir)
    print(f"  Найдено {len(md_files)} файлов анализов")

    parsed = []
    for f in md_files:
        try:
            parsed.append(parse_markdown_analysis(f))
        except Exception as e:
            print(f"  ⚠️ Ошибка парсинга {f.name}: {e}")

    if not parsed:
        return 0

    print("\n  Инициализация embedder...")
    embedder = get_embedder(prefer_st=use_st)
    corpus_texts = []
    for p in parsed:
        corpus_texts.append(p["title"])
        corpus_texts.append(p["what_text"])
    embedder.fit(corpus_texts)
    # Персист фитнутого базиса → find_analogs/index_single используют ТОТ ЖЕ эмбеддер.
    from embeddings import save_embedder

    save_embedder(embedder, _embedder_path(db_path))

    conn, vec_loaded = _connect(db_path)
    indexed = 0
    try:
        # Полная очистка — только для full reindex (index_single этого не делает).
        conn.execute("DELETE FROM news_analyses")
        if vec_loaded:
            conn.execute("DELETE FROM news_embeddings")
        else:
            conn.execute(FALLBACK_EMB_DDL)
            conn.execute("DELETE FROM news_embeddings_fallback")
        for data in parsed:
            _write_row(conn, embedder, data, vec_loaded)
            indexed += 1
            print(f"  [{indexed}/{len(parsed)}] {data['date']} | {data['title'][:60]}...")
        _record_embedder(conn, embedder)
        conn.commit()
    finally:
        conn.close()
    print(f"\n  ✅ Индексировано {indexed} анализов")
    return indexed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG — индексация анализов")
    parser.add_argument(
        "--file", help="Инкрементально индексировать один файл (UPSERT, без стирания БД)"
    )
    parser.add_argument(
        "--use-st",
        action="store_const",
        const=True,
        default=None,
        help="Использовать sentence-transformers вместо TF-IDF "
        "(по умолчанию — из env RADAR_RAG_USE_ST)",
    )
    args = parser.parse_args()

    print("=" * 70)
    if args.file:
        print(f"  RAG — incremental index: {args.file}")
        print("=" * 70)
        ok = index_single(args.file, use_st=args.use_st)
        sys.exit(0 if ok else 1)
    else:
        print("  RAG — full reindex from _Анализы/")
        print("=" * 70)
        n = index_all(use_st=args.use_st)
        print(f"\n  Total: {n} files indexed")
