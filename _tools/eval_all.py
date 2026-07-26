"""
Сводка качества моделей одной командой: что меряем, на чём, какое число получили
и какой порог зафиксирован в тестах.

Смысл файла — чтобы «покажите ваши метрики» решалось одним запуском, а не поиском чисел
по докам. Всё, что можно посчитать без API и без сети, СЧИТАЕТСЯ ЗДЕСЬ И СЕЙЧАС
(retrieval-eval в TF-IDF-пространстве). Прогон L0-классификатора требует ANTHROPIC_API_KEY,
поэтому его числа берутся из закоммиченной витрины `data/l0_eval_results.json` — и это
явно помечено в колонке «источник».

ЗАПУСК:
  cd _tools && python eval_all.py
  cd _tools && python eval_all.py --json     # машиночитаемо
"""

import argparse
import contextlib
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

TOOLS = Path(__file__).parent
sys.path.insert(0, str(TOOLS))

L0_SHOWCASE = TOOLS / "data" / "l0_eval_results.json"
L0_REAL_SHOWCASE = TOOLS / "data" / "l0_eval_real_results.json"
RAG_SHOWCASE = TOOLS / "data" / "rag_eval_results.json"


def _rag_rows():
    """Считаем retrieval-метрики прямо сейчас (TF-IDF: детерминированно, без сети)."""
    import eval_rag

    from tests.test_rag_eval import (  # пороги живут в тестах — единый источник истины
        REAL_P1_BASELINE,
        REAL_R5_BASELINE,
        SYNTHETIC_P1_BASELINE,
        _floor,
    )

    rows = []
    for gold_set, runner, p1_base in (
        ("synthetic", eval_rag.run_synthetic, SYNTHETIC_P1_BASELINE),
        ("real", eval_rag.run_real, REAL_P1_BASELINE),
    ):
        res = runner(k=5, use_st=False)
        if res is None:
            continue
        rows.append(
            {
                "eval": f"RAG retrieval [{gold_set}]",
                "space": "tfidf",
                "n": res["n"],
                "metric": "precision@1",
                "value": res["precision_at_1"],
                "gate": round(_floor(p1_base, res["n"]), 4),
                "source": "измерено сейчас",
                "repro": f"python eval_rag.py --gold {gold_set} --embedder tfidf",
            }
        )
        if gold_set == "real":
            rows.append(
                {
                    "eval": f"RAG retrieval [{gold_set}]",
                    "space": "tfidf",
                    "n": res["n"],
                    "metric": "recall@5",
                    "value": res["recall_at_k"],
                    "gate": round(_floor(REAL_R5_BASELINE, res["n"]), 4),
                    "source": "измерено сейчас",
                    "repro": f"python eval_rag.py --gold {gold_set} --embedder tfidf",
                }
            )
    return rows


def _rag_e5_rows():
    """Числа e5 берём из витрины: прогон требует torch и локальной модели."""
    if not RAG_SHOWCASE.exists():
        return []
    runs = json.loads(RAG_SHOWCASE.read_text(encoding="utf-8")).get("runs", [])
    rows = []
    for r in runs:
        if r.get("embedder") != "e5-small":
            continue
        rows.append(
            {
                "eval": f"RAG retrieval [{r['gold_set']}]",
                "space": "e5-small",
                "n": r["n_queries"],
                "metric": "precision@1",
                "value": r["precision_at_1"],
                "gate": None,
                "source": "витрина (нужен torch)",
                "repro": f"RADAR_RAG_USE_ST=1 python eval_rag.py --gold {r['gold_set']} --embedder e5",
            }
        )
    return rows


def _l0_rows():
    if not L0_SHOWCASE.exists():
        return []
    from tests.test_l0_gold_set import MIN_SUBCATEGORY_ACCURACY

    data = json.loads(L0_SHOWCASE.read_text(encoding="utf-8"))
    rows = []
    for m in data["models"]:
        if "subcategory_accuracy" not in m:
            continue
        rows.append(
            {
                "eval": "L0 классификация шоков",
                "space": m["alias"],
                "n": m["n"],
                "metric": "subcategory accuracy",
                "value": m["subcategory_accuracy"],
                "gate": MIN_SUBCATEGORY_ACCURACY,
                "source": f"витрина от {data['run_date']} (нужен API-ключ)",
                "repro": f"ANTHROPIC_API_KEY=… python eval_l0_classifier.py --model {m['alias']}",
            }
        )
    return rows


def _l0_real_rows():
    if not L0_REAL_SHOWCASE.exists():
        return []
    from tests.test_l0_gold_set_real import MIN_SUB_REAL

    data = json.loads(L0_REAL_SHOWCASE.read_text(encoding="utf-8"))
    rows = []
    for m in data["models"]:
        if "subcategory_accuracy" not in m:
            continue
        rows.append(
            {
                "eval": "L0 классификация (реальные новости)",
                "space": m["alias"],
                "n": m["n"],
                "metric": "subcategory accuracy",
                "value": m["subcategory_accuracy"],
                "gate": MIN_SUB_REAL,
                "source": f"витрина от {data['run_date']} (реальный silver-набор, нужен API-ключ)",
                "repro": (
                    f"ANTHROPIC_API_KEY=… python eval_l0_classifier.py --model {m['alias']} "
                    f"--gold data/l0_gold_set_real.json"
                ),
            }
        )
    return rows


def collect():
    # Индексация болтлива (init_db/index_all печатают прогресс в stdout). Уводим её в stderr,
    # чтобы stdout оставался чистой таблицей и корректным JSON под `--json`.
    with contextlib.redirect_stdout(sys.stderr):
        rag = _rag_rows()
    return rag + _rag_e5_rows() + _l0_real_rows() + _l0_rows()


def main():
    ap = argparse.ArgumentParser(description="Сводная витрина метрик качества")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    rows = collect()
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return

    print()
    print(
        f"{'ЧТО МЕРЯЕМ':26} {'ПРОСТРАНСТВО':13} {'N':>4} {'МЕТРИКА':22} {'ЗНАЧ':>6} "
        f"{'ПОРОГ':>6}  ИСТОЧНИК"
    )
    print("-" * 108)
    for r in rows:
        gate = f"{r['gate']:.2f}" if r["gate"] is not None else "  —"
        mark = "" if r["gate"] is None or r["value"] >= r["gate"] else "  ← НИЖЕ ПОРОГА"
        print(
            f"{r['eval']:26} {r['space']:13} {r['n']:4} {r['metric']:22} "
            f"{r['value']:6.2f} {gate:>6}  {r['source']}{mark}"
        )
    print()
    print("Как воспроизвести:")
    for r in rows:
        print(f"  {r['eval']:26} [{r['space']:9}] → {r['repro']}")
    print()
    print("Оговорки и разбор промахов — docs/EVAL.md. Пороги гейтятся тестами:")
    print("  tests/test_rag_eval.py, tests/test_l0_gold_set.py")


if __name__ == "__main__":
    main()
