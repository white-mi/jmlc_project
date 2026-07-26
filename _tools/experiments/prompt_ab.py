"""A/B эксперимент над промптом L0-классификатора с КОНТРОЛЕМ ОВЕРФИТА.

Сравнивает варианты промпта на ДВУХ наборах:
  • dev      = data/l0_gold_set_real.json      (93; на нём диагностированы confusion-стыки)
  • held-out = data/l0_gold_set_heldout.json   (свежие новости, НЕ виденные при написании правил)

Варианты промпта (правила ОБЩИЕ, из таксономии, без item-specific паттернов):
  • baseline — прод-промпт как есть;
  • all      — прод + ВСЕ 4 правила (A holds, B strikes, C systemic, D threat) — априорная гипотеза;
  • ac       — прод + только A+C (dev-диагноз: B backfires, D бесполезно) — уточнённая гипотеза.

Контроль оверфита: правила зафиксированы ДО прогона held-out; каждый вариант мерится на held-out
РОВНО ОДИН раз (dev — для генерации/уточнения гипотез, held-out — для валидации). Честный сигнал —
дельта на HELD-OUT. Прод-промпт (agents/agent_1_classifier.md) НЕ меняется.

Запуск (нужен ANTHROPIC_API_KEY):
  cd _tools && python experiments/prompt_ab.py --sets dev heldout --variants baseline all ac
Результаты → output/prompt_ab/<set>_<variant>.json (в .gitignore).
"""

import argparse
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

TOOLS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(TOOLS / "agents"))

import eval_l0_classifier as E  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

OUT = TOOLS / "output" / "prompt_ab"
SETS = {
    "dev": TOOLS / "data" / "l0_gold_set_real.json",
    "heldout": TOOLS / "data" / "l0_gold_set_heldout.json",
}

# Правила разграничения границ таксономии (обоснованы _Справочники/Таксономия шоков.md).
# НЕ содержат имён/дат/событий из gold-set — только смысловые определения границ.
RULE_A = """\
A. Решение ЦБ по ставке (4.x) vs «ожидаемое/инфошум» (5.2):
   - Ставка ИЗМЕНЕНА (повышена/снижена) -> 4.1/4.2 по знаку и размеру, даже если решение ожидалось.
   - Ставка СОХРАНЕНА без изменения -> реши по forward-сигналу в тексте:
       * намёк на будущее ПОВЫШЕНИЕ или «поддержание жёсткости ДКУ» -> 4.3;
       * намёк на будущее СНИЖЕНИЕ или готовность смягчать -> 4.4;
       * НЕТ направленного сигнала и решение полностью «в ценах»/ожидаемо -> 5.2.
   5.2 — про ОТСУТСТВИЕ нового сигнала, а не про сам факт сохранения ставки."""

RULE_B = """\
B. Полномасштабный конфликт (1.1) vs точечный удар/диверсия (1.3):
   - 1.1 — начало/эскалация ПОЛНОМАСШТАБНЫХ боевых действий между государствами (вторжение,
     мобилизация, удары одного государства по другому как акт войны).
   - 1.3 — ЛОКАЛЬНАЯ диверсия/точечный удар по ОТДЕЛЬНОМУ объекту (НПЗ, мост, газопровод,
     подстанция) без перехода к полномасштабной войне. Диверсия на инфраструктуре — это 1.3,
     а НЕ 2.4 (2.4 — невоенный техсбой: кибератака, отказ оборудования)."""

RULE_C = """\
C. Системный финкризис (2.1) vs техсбой (2.4) vs дефолт эмитента (2.2):
   - 2.1 — системный шок финсистемы: обвал рынка, набег на банк, заморозка ликвидности,
     санация системно значимого банка регулятором.
   - 2.4 — технический/инфраструктурный сбой, не финансовый кризис.
   - 2.2 — дефолт/реструктуризация ОДНОГО эмитента, не система."""

RULE_D = """\
D. Введённая мера (1.4) vs угроза (5.3):
   - 1.4 — санкции/меры УЖЕ приняты (конкретный пакет/список/решение).
   - 5.3 — только угроза/ультиматум ввести меры в будущем, без конкретного введённого действия."""

HEADER = (
    "ПРАВИЛА РАЗГРАНИЧЕНИЯ СПОРНЫХ ГРАНИЦ (применяй по СМЫСЛУ текста, НЕ по конкретным названиям):"
)
VARIANTS = {
    "baseline": [],
    "all": [RULE_A, RULE_B, RULE_C, RULE_D],
    "ac": [RULE_A, RULE_C],
}


def variant_suffix(variant):
    blocks = VARIANTS[variant]
    return "" if not blocks else HEADER + "\n\n" + "\n\n".join(blocks)


def build_prompt(item, variant):
    base = E.build_prompt(item)  # прод-промпт + инлайн-таксономия
    suffix = variant_suffix(variant)
    return base + "\n\n" + suffix if suffix else base


def run_variant(items, variant, model="claude-haiku-4-5-20251001"):
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    rows, tin, tout = [], 0, 0
    for it in items:
        prompt = build_prompt(it, variant)
        resp = None
        for attempt in range(4):
            try:
                resp = client.messages.create(
                    model=model,
                    max_tokens=1200,
                    messages=[{"role": "user", "content": prompt}],
                )
                break
            except Exception as exc:
                if attempt == 3:
                    raise
                print(f"  [retry {it['id']} {attempt + 1}] {exc}", file=sys.stderr)
                time.sleep(2 * (attempt + 1))
        raw = E.response_text(resp)
        tin += resp.usage.input_tokens
        tout += resp.usage.output_tokens
        m, s, _ = E.parse_pred(raw)
        rows.append(
            {
                "id": it["id"],
                "boundary": bool(it["boundary"]),
                "gold_main": it["gold_main"],
                "gold_sub": it["gold_sub"],
                "pred_main": m,
                "pred_sub": s,
                "main_ok": m == it["gold_main"],
                "sub_ok": s == it["gold_sub"],
            }
        )
    return summarize(rows, tin, tout, model)


def summarize(rows, tin, tout, model):
    n = len(rows)
    sub_k = sum(r["sub_ok"] for r in rows)
    main_k = sum(r["main_ok"] for r in rows)
    nb = [r for r in rows if not r["boundary"]]
    nb_k = sum(r["sub_ok"] for r in nb)
    by = defaultdict(list)
    for r in rows:
        by[r["gold_sub"]].append(r)
    per_class = {
        g: {"n": len(rs), "correct": sum(x["sub_ok"] for x in rs)} for g, rs in sorted(by.items())
    }
    conf = Counter((r["gold_sub"], r["pred_sub"]) for r in rows if not r["sub_ok"])
    pi, po = E.PRICES.get(model, (0.0, 0.0))
    return {
        "n": n,
        "main_accuracy": round(main_k / n, 4),
        "main_correct": f"{main_k}/{n}",
        "sub_accuracy": round(sub_k / n, 4),
        "sub_ci95": E.wilson_ci(sub_k, n),
        "sub_correct": f"{sub_k}/{n}",
        "sub_excl_boundary": round(nb_k / len(nb), 4) if nb else 0.0,
        "sub_excl_boundary_correct": f"{nb_k}/{len(nb)}",
        "per_class": per_class,
        "confusion": [{"gold": g, "pred": p, "count": c} for (g, p), c in conf.most_common()],
        "misses": [
            {"id": r["id"], "gold": r["gold_sub"], "pred": r["pred_sub"], "boundary": r["boundary"]}
            for r in rows
            if not r["sub_ok"]
        ],
        "cost_usd": round(tin / 1e6 * pi + tout / 1e6 * po, 4),
    }


def mcnemar(base_res, other_res):
    """Точный двусторонний McNemar по дискордантным парам (одни и те же айтемы, парный тест).
    b = baseline верно / other неверно; c = baseline неверно / other верно (по sub-меткам).
    p — двусторонний точный биномиальный (X~Bin(b+c, 0.5)), без scipy."""
    bw = {m["id"] for m in base_res["misses"]}
    ow = {m["id"] for m in other_res["misses"]}
    b = len(ow - bw)  # был верно → стал неверно
    c = len(bw - ow)  # был неверно → стал верно
    n = b + c
    if n == 0:
        return {"b": 0, "c": 0, "n_discordant": 0, "p_two_sided": 1.0}
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5**n)
    return {"b": b, "c": c, "n_discordant": n, "p_two_sided": round(min(1.0, 2 * tail), 4)}


def subset_metrics(res, keep_ids):
    """sub-accuracy на подмножестве айтемов (для 71-new-only среза)."""
    wrong = sum(1 for m in res["misses"] if m["id"] in keep_ids)
    n = len(keep_ids)
    k = n - wrong
    return {"n": n, "correct": f"{k}/{n}", "accuracy": round(k / n, 4) if n else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=["dev", "heldout"], choices=list(SETS))
    ap.add_argument(
        "--variants", nargs="+", default=["baseline", "all", "ac"], choices=list(VARIANTS)
    )
    args = ap.parse_args()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY не задан.", file=sys.stderr)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    table = []
    for sname in args.sets:
        path = SETS[sname]
        if not path.exists():
            print(f"пропуск {sname}: нет {path.name}", file=sys.stderr)
            continue
        items = E.load_gold(path)
        for variant in args.variants:
            print(f"=== {sname} / {variant} (N={len(items)}) ===", file=sys.stderr)
            res = run_variant(items, variant)
            (OUT / f"{sname}_{variant}.json").write_text(
                json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            table.append((sname, variant, res))
            print(
                f"  sub {res['sub_correct']} ({res['sub_accuracy']:.0%}) CI {res['sub_ci95']} "
                f"| main {res['main_correct']} | nb {res['sub_excl_boundary_correct']} "
                f"| ${res['cost_usd']}",
                file=sys.stderr,
            )
    print("\n=== СВОДКА ===")
    print(f"{'set':<9}{'variant':<10}{'N':>4}  {'sub':>10} {'main':>8} {'nb':>8}")
    for sname, variant, r in table:
        print(
            f"{sname:<9}{variant:<10}{r['n']:>4}  {r['sub_correct']:>10} "
            f"{r['main_correct']:>8} {r['sub_excl_boundary_correct']:>8}"
        )

    # Парный McNemar baseline↔variant + срез 71-new-only (id-порог "h30" — новые айтемы).
    print("\n=== McNemar (baseline ↔ variant, парный, sub) ===")
    for sname in args.sets:
        results = {v: r for (s, v, r) in table if s == sname}
        base = results.get("baseline")
        if not base:
            continue
        for variant, r in results.items():
            if variant == "baseline":
                continue
            mc = mcnemar(base, r)
            verdict = (
                "improved лучше"
                if mc["c"] > mc["b"]
                else ("baseline лучше" if mc["b"] > mc["c"] else "ничья")
            )
            print(
                f"  {sname} baseline↔{variant}: b={mc['b']} c={mc['c']} "
                f"(дискорд {mc['n_discordant']}) p_two_sided={mc['p_two_sided']} — {verdict}"
            )

    # Срез "только НОВЫЕ" held-out (id-номер ≥ 30) — чистейший held-out под замороженные правила
    # (айтемы h01–h29 уже измерялись на предыдущем прогоне; новые h30+ — впервые).
    if "heldout" in args.sets and SETS["heldout"].exists():
        new_ids = {it["id"] for it in E.load_gold(SETS["heldout"]) if int(it["id"][1:]) >= 30}
        ho = {v: r for (s, v, r) in table if s == "heldout"}
        if new_ids and ho:
            print(f"\n=== held-out: только НОВЫЕ айтемы (N={len(new_ids)}) ===")
            for variant, r in ho.items():
                sm = subset_metrics(r, new_ids)
                print(f"  {variant}: sub {sm['correct']} ({sm['accuracy']:.0%})")


if __name__ == "__main__":
    main()
