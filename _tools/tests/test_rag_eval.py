"""CI-safe тесты RAG-eval: чистые метрик-хелперы + валидность gold-set (без sqlite-vec/ретрива)."""

import eval_rag as E


def test_metrics_for_query_arithmetic():
    assert E.metrics_for_query(["a", "b", "c"], ["a"], k=5) == (1.0, 1.0, 1.0)  # top-1
    h1, hk, rr = E.metrics_for_query(["x", "y", "a"], ["a"], k=5)  # gold на ранге 3
    assert h1 == 0.0 and hk == 1.0 and abs(rr - 1.0 / 3) < 1e-9
    assert E.metrics_for_query(["x", "y"], ["a"], k=5) == (0.0, 0.0, 0.0)  # промах
    # gold за пределами k
    assert E.metrics_for_query(["x", "y", "z", "w", "v", "a"], ["a"], k=5)[1] == 0.0


def test_aggregate():
    pq = [(["a"], ["a"]), (["x", "a"], ["a"])]
    agg = E.aggregate(pq, k=5)
    assert agg["n"] == 2
    assert agg["precision_at_1"] == 0.5
    assert agg["recall_at_k"] == 1.0
    assert abs(agg["mrr"] - (1.0 + 0.5) / 2) < 1e-9
    assert E.aggregate([], k=5)["n"] == 0


def test_gold_set_schema():
    gold = E.load_gold()
    ids = {d["id"] for d in gold["docs"]}
    assert len(gold["docs"]) >= 8 and len(gold["queries"]) >= 10
    assert len(ids) == len(gold["docs"]), "duplicate doc id"
    for q in gold["queries"]:
        assert q["gold"], "query without gold"
        assert all(g in ids for g in q["gold"]), "gold id not among docs"
    for d in gold["docs"]:
        assert d["title"] and d["what"], "doc missing title/what"
