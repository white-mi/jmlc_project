"""Тесты RAG-eval: чистые метрик-хелперы, валидность обоих gold-set'ов И регрессионные
гейты качества ретрива.

Про гейты. Eval реально ПРОГОНЯЕТСЯ (BLOB-fallback считает косинусы в numpy,
`sqlite-vec` не нужен) и сравнивается с зафиксированным baseline: проверок схемы и
арифметики недостаточно — при них качество поиска может упасть до нуля, а CI останется
зелёным. Порог задан с запасом «минус один промах», чтобы тест ловил деградацию, а не
дрожание на границе.

Гоняем в TF-IDF-пространстве: детерминированно, без сети и torch. Числа e5 меряются
отдельно (`RADAR_RAG_USE_ST=1 python eval_rag.py --gold real --embedder e5`) и лежат в
витрине `data/rag_eval_results.json`.
"""

import json
from pathlib import Path

import eval_rag as E

TOOLS = Path(E.__file__).resolve().parent

# Зафиксированные baseline (TF-IDF). Источник — витрина data/rag_eval_results.json.
SYNTHETIC_P1_BASELINE = 0.9167  # 11/12
REAL_P1_BASELINE = 0.5789  # 22/38
REAL_R5_BASELINE = 0.7632  # 29/38


def _floor(baseline: float, n: int, slack: int = 1) -> float:
    """Порог = baseline минус `slack` промахов: тест ловит деградацию, а не шум."""
    return baseline - slack / n - 1e-9


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


def test_real_gold_set_points_at_existing_docs():
    """Каждый gold-id реального набора — существующий файл корпуса `_Анализы/_история/`."""
    gold = E.load_gold(E.GOLD_REAL)
    corpus = (TOOLS / gold["corpus_dir"]).resolve()
    stems = {p.stem for p in corpus.glob("*.md")}
    assert len(stems) >= 30, f"корпус ретро-разборов подозрительно мал: {len(stems)}"
    assert len(gold["queries"]) >= 25
    for q in gold["queries"]:
        assert q["gold"], "query without gold"
        for g in q["gold"]:
            assert g in stems, f"gold-id без файла в корпусе: {g}"


def test_real_gold_queries_are_paraphrases():
    """Запрос не должен дословно повторять заголовок документа — иначе eval меряет
    совпадение строк, а не способность найти смысловой аналог."""
    gold = E.load_gold(E.GOLD_REAL)
    for q in gold["queries"]:
        qwords = {w.lower().strip(",.:;»«") for w in q["query"].split() if len(w) > 4}
        for g in q["gold"]:
            title = g.split("—", 1)[-1]
            twords = {w.lower().strip(",.:;»«") for w in title.split() if len(w) > 4}
            overlap = len(qwords & twords) / max(1, len(twords))
            assert overlap < 0.5, f"запрос слишком близок к заголовку ({overlap:.0%}): {q['query']}"


def test_retrieval_synthetic_meets_baseline():
    res = E.run_synthetic(k=5, use_st=False)
    assert res is not None
    floor = _floor(SYNTHETIC_P1_BASELINE, res["n"])
    assert (
        res["precision_at_1"] >= floor
    ), f"precision@1 упал: {res['precision_at_1']} < {floor:.4f} (baseline {SYNTHETIC_P1_BASELINE})"


def test_retrieval_real_meets_baseline():
    """Главный гейт: реальные документы + запросы-парафразы. Числа скромнее синтетики —
    это и есть честная оценка ретрива, а не витринная."""
    res = E.run_real(k=5, use_st=False)
    assert res is not None, "реальный gold-set/корпус недоступен"
    assert res["indexed"] >= 30
    p1_floor = _floor(REAL_P1_BASELINE, res["n"])
    r5_floor = _floor(REAL_R5_BASELINE, res["n"])
    assert res["precision_at_1"] >= p1_floor, f"precision@1 деградировал: {res['precision_at_1']}"
    assert res["recall_at_k"] >= r5_floor, f"recall@5 деградировал: {res['recall_at_k']}"


def test_showcase_matches_measured_tfidf():
    """Закоммиченная витрина не должна расходиться с тем, что реально считается сейчас."""
    showcase = json.loads((TOOLS / "data" / "rag_eval_results.json").read_text(encoding="utf-8"))
    rows = {(r["gold_set"], r["embedder"]): r for r in showcase["runs"]}
    assert ("real", "e5-small") in rows, "в витрине нет строки e5 — A/B неполон"
    for gold_set, runner in (("synthetic", E.run_synthetic), ("real", E.run_real)):
        row = rows[(gold_set, "tfidf")]
        res = runner(k=5, use_st=False)
        assert abs(row["precision_at_1"] - res["precision_at_1"]) < 1e-6, (
            f"витрина [{gold_set}/tfidf] разошлась с замером: "
            f"{row['precision_at_1']} vs {res['precision_at_1']}"
        )
