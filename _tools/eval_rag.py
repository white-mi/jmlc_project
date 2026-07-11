"""
L0 RAG retrieval eval — precision@1 / recall@k / MRR на синтетическом публичном gold-set.

ЧТО ДЕЛАЕТ: индексирует 10 синтетических «исторических разборов» разных типов шоков
(`data/rag_gold_set.json`), прогоняет 12 размеченных запросов через боевой `find_analogs`
и меряет, попадает ли правильный аналог в top-1 / top-k.

ЧЕСТНЫЕ ОГОВОРКИ: это малый-N sanity-метрика (N=12), НЕ бенчмарк. Данные синтетические и
публичные; TF-IDF-эмбеддер (детерминированный). Метрики честны только относительно этого
маленького набора и разных-по-теме доков. Ранжирование требует `sqlite-vec`; без него —
graceful-exit (CI-safe, как у `eval_l0_classifier.py`).

ЗАПУСК: cd _tools && python eval_rag.py [--emit]   (--emit → output/rag_eval/results.json)
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
OUT = TOOLS / "output" / "rag_eval"


def load_gold():
    return json.loads(GOLD.read_text(encoding="utf-8"))


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
        '---\nдата_новости: "2025-01-01"\n---\n'
        f'# {doc["title"]}\n\n## L0 — Классификация\n{doc["what"]}\n',
        encoding="utf-8",
    )


def run(k=5):
    import init_db
    import index_news as ix
    import find_analogs as fa

    try:
        import sqlite_vec  # noqa: F401
    except Exception:
        print("sqlite-vec не установлен — ранжирование недоступно (CI-safe exit).", file=sys.stderr)
        return None

    gold = load_gold()
    docs = gold["docs"]
    title_to_id = {d["title"]: d["id"] for d in docs}

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        db = d / "rag_eval.db"
        an = d / "_Анализы"
        an.mkdir()
        for doc in docs:
            _write_doc(an, doc)
        init_db.init_db(db_path=db)
        n_idx = ix.index_all(db_path=db, analyses_dir=an)

        per_query, rows = [], []
        for q in gold["queries"]:
            hits = fa.find_analogs(q["query"], db_path=db, threshold=0.0, top_k=k)
            ranked = [title_to_id.get(h["title"]) for h in hits]
            ranked = [x for x in ranked if x]
            per_query.append((ranked, q["gold"]))
            rows.append(
                {
                    "query": q["query"],
                    "gold": q["gold"],
                    "top": ranked[:k],
                    "top1_sim": round(hits[0]["similarity"], 3) if hits else None,
                }
            )
        summary = aggregate(per_query, k)
        summary["indexed"] = n_idx
        summary["rows"] = rows
        return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--emit", action="store_true")
    args = ap.parse_args()
    res = run(args.k)
    if res is None:
        return
    print("=" * 60, file=sys.stderr)
    print(f"RAG retrieval eval (synthetic, N={res['n']}, k={res['k']})", file=sys.stderr)
    print(f"  precision@1 : {res['precision_at_1']:.0%}", file=sys.stderr)
    print(f"  recall@{res['k']:<7}: {res['recall_at_k']:.0%}", file=sys.stderr)
    print(f"  MRR         : {res['mrr']:.3f}", file=sys.stderr)
    for r in res["rows"]:
        ok = (
            "OK "
            if r["top"] and r["top"][0] in r["gold"]
            else ("~  " if any(g in r["top"] for g in r["gold"]) else "MISS")
        )
        print(
            f"  {ok} '{r['query'][:38]}' -> {r['top'][:3]} (sim {r['top1_sim']})", file=sys.stderr
        )
    if args.emit:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "results.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("→ output/rag_eval/results.json", file=sys.stderr)


if __name__ == "__main__":
    main()
