"""A/B гранулярности чанкинга для L0-RAG на реальном gold-set.

Проверяет утверждение из docs/EVAL.md «на нашем корпусе документ-уровень не хуже
секционного» замером, а не рассуждением. Один и тот же эмбеддер (e5-small) и один и
тот же текст документа сравниваются в двух режимах гранулярности — изолируется ровно
переменная чанкинга:

  * doc  — один вектор на весь разбор: score = cos(query, embed(title + тело)).
  * sect — тело режется по заголовкам H2 (## …), каждая секция кодируется отдельно;
           score = max по {title, секции} от cos(query, вектор). Это max-pooling:
           документ релевантен, если релевантна хотя бы одна его секция.

Титул включён в оба режима, чтобы ни один не был искусственно ущемлён.

Ретрив-единица в обоих случаях — документ (событие); меняется только то, на скольких
векторах он представлен. Офлайн, детерминированно (e5 из локального кэша HF); прод-БД
не трогается.

Запуск:  cd _tools && RADAR_RAG_USE_ST=1 python experiments/chunking_ab.py
"""

import json
import sys
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(TOOLS / "agents" / "rag"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import index_news as ix  # noqa: E402  (переиспользуем чистку markdown и извлечение полей)
from embeddings import embedder_name, get_embedder  # noqa: E402
from eval_rag import metrics_for_query  # noqa: E402

GOLD_REAL = TOOLS / "data" / "rag_gold_set_real.json"
H2 = "\n## "


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _sections(body: str) -> list[str]:
    """Тело разбора → список секций по H2. Мелкие склеиваются с соседями, чтобы не
    плодить обрывки в один-два токена."""
    raw = ("\n" + body).split(H2)
    parts = []
    for i, chunk in enumerate(raw):
        text = ix._clean_markdown(chunk if i == 0 else "## " + chunk).strip()
        if len(text) >= 40:  # секции короче — не самостоятельный смысловой блок
            parts.append(text)
    return parts or [ix._clean_markdown(body).strip()]


def _load_docs(corpus_dir: Path) -> dict:
    """{doc_id: (title, [тексты для кодирования: весь_док, секции...])}."""
    docs = {}
    for path in sorted(corpus_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        parsed = ix.parse_markdown_analysis(path)
        title = parsed["title"]
        # тело без frontmatter
        body = text.split("---", 2)[-1] if text.startswith("---") else text
        whole = ix._clean_markdown(f"{title}\n{body}").strip()
        docs[path.stem] = {
            "title": title,
            "doc_text": whole,
            "sections": [title] + _sections(body),
        }
    return docs


def _embed_all(docs: dict, emb) -> dict:
    """Предкодировать оба представления каждого документа."""
    out = {}
    for did, d in docs.items():
        doc_vec = emb.encode(d["doc_text"], is_query=False)
        sec_vecs = [emb.encode(s, is_query=False) for s in d["sections"]]
        out[did] = {"doc": doc_vec, "sect": sec_vecs}
    return out


def _rank(query_vec, embedded: dict, mode: str) -> list[str]:
    scored = []
    for did, rep in embedded.items():
        if mode == "doc":
            score = _cos(query_vec, rep["doc"])
        else:
            score = max(_cos(query_vec, v) for v in rep["sect"])
        scored.append((score, did))
    scored.sort(reverse=True)
    return [did for _, did in scored]


def run(k: int = 5) -> dict:
    gold = json.loads(GOLD_REAL.read_text(encoding="utf-8"))
    corpus_dir = (TOOLS / gold["corpus_dir"]).resolve()
    queries = gold["queries"]

    emb = get_embedder(prefer_st=True)
    name = embedder_name(emb)
    if name == "tfidf":
        # TF-IDF нужно фитить на корпусе; A/B имеет смысл на нейроэмбеддере с фиксированным
        # пространством. Просим явно.
        print(
            "e5 недоступен (fallback на TF-IDF) — A/B чанкинга требует нейроэмбеддера. "
            "Проверьте RADAR_RAG_USE_ST=1 и локальный кэш intfloat/multilingual-e5-small.",
            file=sys.stderr,
        )

    docs = _load_docs(corpus_dir)
    embedded = _embed_all(docs, emb)

    results = {}
    for mode in ("doc", "sect"):
        per_query = []
        for q in queries:
            qv = emb.encode(q["query"], is_query=True)
            ranked = _rank(qv, embedded, mode)
            per_query.append((ranked, q["gold"]))
        h1 = hk = mrr = 0.0
        for ranked, g in per_query:
            a, b, c = metrics_for_query(ranked, g, k)
            h1 += a
            hk += b
            mrr += c
        n = len(per_query)
        results[mode] = {
            "precision_at_1": round(h1 / n, 4),
            "recall_at_k": round(hk / n, 4),
            "mrr": round(mrr / n, 4),
        }

    avg_sections = round(sum(len(d["sections"]) for d in docs.values()) / len(docs), 1)
    return {
        "embedder": name,
        "n_queries": len(queries),
        "n_docs": len(docs),
        "avg_vectors_per_doc_sect": avg_sections,
        "k": k,
        "doc_level": results["doc"],
        "section_level": results["sect"],
    }


def main():
    res = run()
    print("=" * 60, file=sys.stderr)
    print(
        f"Чанкинг A/B [{res['embedder']}] N={res['n_queries']} запросов, "
        f"{res['n_docs']} доков, k={res['k']}",
        file=sys.stderr,
    )
    print(f"  секций/док в среднем: {res['avg_vectors_per_doc_sect']}", file=sys.stderr)
    for mode, label in (("doc_level", "документ-уровень"), ("section_level", "секции (max-pool)")):
        m = res[mode]
        print(
            f"  {label:22s}: p@1 {m['precision_at_1']:.3f}  "
            f"recall@{res['k']} {m['recall_at_k']:.3f}  MRR {m['mrr']:.3f}",
            file=sys.stderr,
        )
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
