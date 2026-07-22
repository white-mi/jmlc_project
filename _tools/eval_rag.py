"""
L0 RAG retrieval eval — precision@1 / recall@k / MRR / margin на размеченных gold-set'ах.

ДВА НАБОРА:
  * `--gold synthetic` — 10 синтетических «исторических разборов» + 12 запросов
    (`data/rag_gold_set.json`, тексты доков лежат прямо в JSON). Быстрый, полностью
    самодостаточный: используется как регрессионный гейт в CI.
  * `--gold real` — реальные ретро-разборы исторических шоков РФ из `_Анализы/_история/`
    и запросы-парафразы к ним (`data/rag_gold_set_real.json`). Меряет то, что реально
    поедет в прод: настоящие документы, настоящая лексика.

ЧЕСТНЫЕ ОГОВОРКИ: N мал (десятки запросов) — это sanity-метрика, НЕ бенчмарк. Числа честны
только относительно этих наборов. `margin` (разрыв top1↔top2) считается потому, что у e5
абсолютные косинусы сжаты в узкий диапазон и сами по себе неинформативны.

Ранжирование работает и БЕЗ `sqlite-vec` (BLOB-fallback считает косинус в numpy и даёт
те же числа) — поэтому eval запускается в CI без тяжёлых зависимостей.

ЗАПУСК:
  cd _tools
  python eval_rag.py                                   # synthetic, TF-IDF
  python eval_rag.py --gold real                       # реальные разборы
  RADAR_RAG_USE_ST=1 python eval_rag.py --gold real --embedder e5
  python eval_rag.py --gold both --emit-showcase       # обновить data/rag_eval_results.json
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

TOOLS = Path(__file__).parent
sys.path.insert(0, str(TOOLS / "agents" / "rag"))
GOLD = TOOLS / "data" / "rag_gold_set.json"
GOLD_REAL = TOOLS / "data" / "rag_gold_set_real.json"
SHOWCASE = TOOLS / "data" / "rag_eval_results.json"
OUT = TOOLS / "output" / "rag_eval"


def load_gold(path: Path = GOLD):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def metrics_for_query(ranked_ids, gold_ids, k=5):
    """Чистая функция: (ранжированные doc-id, gold-id) → (hit@1, hit@k, reciprocal-rank)."""
    gold = set(gold_ids)
    hit1 = 1.0 if ranked_ids and ranked_ids[0] in gold else 0.0
    hitk = 1.0 if any(x in gold for x in ranked_ids[:k]) else 0.0
    rr = 0.0
    for i, x in enumerate(ranked_ids, 1):
        if x in gold:
            rr = 1.0 / i
            break
    return hit1, hitk, rr


def aggregate(per_query, k=5):
    """per_query: list of (ranked_ids, gold_ids) → сводка precision@1 / recall@k / MRR."""
    if not per_query:
        return {"n": 0, "k": k, "precision_at_1": 0.0, "recall_at_k": 0.0, "mrr": 0.0}
    h1 = hk = mrr = 0.0
    for ranked, gold in per_query:
        a, b, c = metrics_for_query(ranked, gold, k)
        h1 += a
        hk += b
        mrr += c
    n = len(per_query)
    return {
        "n": n,
        "k": k,
        "precision_at_1": round(h1 / n, 4),
        "recall_at_k": round(hk / n, 4),
        "mrr": round(mrr / n, 4),
    }


def _write_doc(dir_path, doc):
    (dir_path / (doc["id"] + ".md")).write_text(
        '---\ndата_новости: "2025-01-01"\nшок_категория: "0.0 (синтетика)"\n---\n'
        f'# {doc["title"]}\n\n## L0 — Классификация\n\n'
        f'| Параметр | Значение |\n|---|---|\n| **Что произошло** | {doc["what"]} |\n',
        encoding="utf-8",
    )


def _margin(hits) -> float | None:
    """Разрыв между top1 и top2. У e5 абсолютный косинус почти не различает документы,
    поэтому уверенность выдачи видна именно в разрыве, а не в самом значении."""
    if len(hits) < 2:
        return None
    return round(hits[0]["similarity"] - hits[1]["similarity"], 4)


def _eval_corpus(corpus_dir: Path, queries, id_of, k, use_st, db_path):
    """Проиндексировать корпус и прогнать запросы. id_of: hit-dict → doc-id (или None)."""
    import find_analogs as fa
    import index_news as ix
    import init_db

    # init_db возвращает открытое соединение: без close() файл БД остаётся залоченным и
    # TemporaryDirectory падает на Windows с PermissionError при уборке.
    init_db.init_db(db_path=db_path).close()
    n_idx = ix.index_all(db_path=db_path, analyses_dir=corpus_dir, use_st=use_st)

    per_query, rows, margins = [], [], []
    for q in queries:
        hits = fa.find_analogs(q["query"], db_path=db_path, threshold=0.0, top_k=k, use_st=use_st)
        ranked = [x for x in (id_of(h) for h in hits) if x]
        per_query.append((ranked, q["gold"]))
        m = _margin(hits)
        if m is not None:
            margins.append(m)
        rows.append(
            {
                "query": q["query"],
                "gold": q["gold"],
                "top": ranked[:k],
                "top1_sim": round(hits[0]["similarity"], 3) if hits else None,
                "margin_top1_top2": m,
            }
        )
    summary = aggregate(per_query, k)
    summary["indexed"] = n_idx
    summary["mean_margin"] = round(sum(margins) / len(margins), 4) if margins else None
    summary["rows"] = rows
    return summary


def run_synthetic(k=5, use_st=None):
    """Синтетический набор: доки лежат в JSON, корпус разворачивается во временную папку."""
    gold = load_gold(GOLD)
    docs = gold["docs"]
    title_to_id = {d["title"]: d["id"] for d in docs}
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        an = d / "_Анализы"
        an.mkdir()
        for doc in docs:
            _write_doc(an, doc)
        res = _eval_corpus(
            an,
            gold["queries"],
            lambda h: title_to_id.get(h["title"]),
            k,
            use_st,
            d / "rag_eval.db",
        )
    res["gold_set"] = "synthetic"
    return res


def run_real(k=5, use_st=None):
    """Реальный набор: индексируется настоящий корпус ретро-разборов `_Анализы/_история/`."""
    if not GOLD_REAL.exists():
        print(f"gold-set не найден: {GOLD_REAL}", file=sys.stderr)
        return None
    gold = load_gold(GOLD_REAL)
    corpus_dir = (TOOLS / gold["corpus_dir"]).resolve()
    if not corpus_dir.exists():
        print(f"корпус не найден: {corpus_dir}", file=sys.stderr)
        return None
    with tempfile.TemporaryDirectory() as d:
        res = _eval_corpus(
            corpus_dir,
            gold["queries"],
            lambda h: Path(h["file_path"]).stem,
            k,
            use_st,
            Path(d) / "rag_eval_real.db",
        )
    res["gold_set"] = "real"
    return res


def run(k=5, gold="synthetic", use_st=None):
    return run_real(k, use_st) if gold == "real" else run_synthetic(k, use_st)


def _report(res):
    print("=" * 66, file=sys.stderr)
    print(
        f"RAG retrieval eval [{res['gold_set']}] N={res['n']} k={res['k']} "
        f"docs={res['indexed']}",
        file=sys.stderr,
    )
    print(f"  precision@1 : {res['precision_at_1']:.0%}", file=sys.stderr)
    print(f"  recall@{res['k']:<7}: {res['recall_at_k']:.0%}", file=sys.stderr)
    print(f"  MRR         : {res['mrr']:.3f}", file=sys.stderr)
    if res.get("mean_margin") is not None:
        print(f"  margin 1↔2  : {res['mean_margin']:.3f}", file=sys.stderr)
    for r in res["rows"]:
        ok = (
            "OK  "
            if r["top"] and r["top"][0] in r["gold"]
            else ("~   " if any(g in r["top"] for g in r["gold"]) else "MISS")
        )
        print(
            f"  {ok} '{r['query'][:44]}' -> {r['top'][:2]} (sim {r['top1_sim']})", file=sys.stderr
        )


def main():
    ap = argparse.ArgumentParser(description="RAG retrieval eval")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--gold", choices=["synthetic", "real", "both"], default="synthetic")
    ap.add_argument(
        "--embedder",
        choices=["tfidf", "e5"],
        default=None,
        help="по умолчанию — из env RADAR_RAG_USE_ST",
    )
    ap.add_argument("--emit", action="store_true", help="output/rag_eval/results.json")
    ap.add_argument(
        "--emit-showcase", action="store_true", help="обновить data/rag_eval_results.json"
    )
    args = ap.parse_args()

    use_st = None if args.embedder is None else (args.embedder == "e5")
    names = ["synthetic", "real"] if args.gold == "both" else [args.gold]

    results = []
    for name in names:
        res = run(args.k, name, use_st)
        if res is None:
            continue
        _report(res)
        results.append(res)
    if not results:
        return

    if args.emit:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "results.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("→ output/rag_eval/results.json", file=sys.stderr)
    if args.emit_showcase:
        _emit_showcase(results, use_st)


def _emit_showcase(results, use_st):
    """Обновить закоммиченную витрину, сохранив уже записанные строки других эмбеддеров."""
    from embeddings import RAG_USE_ST, EMBEDDER_E5, EMBEDDER_TFIDF

    emb = EMBEDDER_E5 if (RAG_USE_ST if use_st is None else use_st) else EMBEDDER_TFIDF
    old = json.loads(SHOWCASE.read_text(encoding="utf-8")) if SHOWCASE.exists() else {}
    runs = {
        (r.get("embedder"), r.get("gold_set")): r
        for r in old.get("runs", [])
        if isinstance(r, dict)
    }
    for res in results:
        runs[(emb, res["gold_set"])] = {
            "embedder": emb,
            "gold_set": res["gold_set"],
            "n_docs": res["indexed"],
            "n_queries": res["n"],
            "k": res["k"],
            "precision_at_1": res["precision_at_1"],
            "recall_at_k": res["recall_at_k"],
            "mrr": res["mrr"],
            "mean_margin": res.get("mean_margin"),
            "misses": [
                r["query"] for r in res["rows"] if not (r["top"][:1] and r["top"][0] in r["gold"])
            ],
        }
    payload = {
        "_comment": old.get(
            "_comment",
            "Закоммиченная витрина retrieval-eval L0-RAG (find_analogs). Малый-N SANITY-метрика, "
            "НЕ бенчмарк. Воспроизведение: cd _tools && python eval_rag.py --gold both "
            "--emit-showcase (sqlite-vec не требуется). Пороги регрессии — tests/test_rag_eval.py.",
        ),
        "eval": "L0 RAG — retrieval-качество find_analogs",
        "gold_sets": {"synthetic": str(GOLD.name), "real": str(GOLD_REAL.name)},
        "runs": sorted(runs.values(), key=lambda r: (r["gold_set"], r["embedder"])),
    }
    SHOWCASE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"→ {SHOWCASE.relative_to(TOOLS)}", file=sys.stderr)


if __name__ == "__main__":
    main()
