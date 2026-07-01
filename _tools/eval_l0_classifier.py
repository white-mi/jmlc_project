"""
L0 Agent-1 (Classifier) eval harness — честный sanity-check классификации шоков.

ЧТО ДЕЛАЕТ: прогоняет реальный промпт Agent 1 (тот же, что в продакшене,
orchestrator.extract_prompt) на gold-set (_tools/data/l0_gold_set.json) через
Anthropic API, сравнивает предсказанные main_category / subcategory с эталоном,
считает accuracy и стоимость по токенам (response.usage).

ЧЕСТНЫЕ ОГОВОРКИ (важно для интерпретации):
  * Это SANITY-CHECK на N=15 синтетических публичных новостях, НЕ статистически
    значимый бенчмарк. Эталонные метки проставлены вручную по таксономии.
  * Промпт Agent 1 ссылается на справочник таксономии как на ФАЙЛ. В продакшене
    режим --llm-mode cli (claude -p внутри Claude Code) читает файл инструментом;
    голый API так не умеет, поэтому harness ИНЛАЙНИТ компактную таксономию в промпт
    (TAXONOMY_INLINE ниже) — это воспроизводит то, что cli-агент получает из файла.
  * Подкатегория (27-классовая) объективно труднее main_category (5-классовая);
    пограничные пункты (boundary=true, напр. ожидаемое снижение КС 4.2 vs 5.2)
    считаются отдельно.

ЗАПУСК (нужен ANTHROPIC_API_KEY в окружении; без него — graceful exit, CI-safe):
  cd _tools
  python eval_l0_classifier.py --model haiku           # дёшево
  python eval_l0_classifier.py --model sonnet           # сильнее
  python eval_l0_classifier.py --model opus --limit 5    # частичный прогон

Сырые результаты пишутся в output/l0_eval/<model>_results.json (output/ в .gitignore).
Закоммиченная витрина (для теста и docs) собирается отдельно в data/l0_eval_results.json.
"""

import argparse
import json
import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

TOOLS_DIR = Path(__file__).parent
sys.path.insert(0, str(TOOLS_DIR / "agents"))

GOLD_PATH = TOOLS_DIR / "data" / "l0_gold_set.json"
OUT_DIR = TOOLS_DIR / "output" / "l0_eval"

# Полные ID моделей (alias → id). Голый API не принимает короткие алиасы.
MODEL_IDS = {
    "haiku": "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-6",
    "opus": "claude-opus-4-8",
}

# Цены $/MTok (input, output) — для оценки стоимости прогона. Опубликованные тарифы.
PRICES = {
    "claude-haiku-4-5-20251001": (1.0, 5.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-opus-4-8": (15.0, 75.0),
}

# Компактная таксономия (27 подкатегорий) — инлайн вместо файловой ссылки в промпте.
TAXONOMY_INLINE = """\
ПОЛНАЯ ТАКСОНОМИЯ (5 категорий, 27 подкатегорий) — выбери ТОЧНУЮ подкатегорию:
1. ВОЙНА/Геополитика: 1.1 военный конфликт старт/эскалация; 1.2 деэскалация/завершение;
   1.3 точечные удары/диверсии; 1.4 новый санкционный пакет; 1.5 прокси-конфликт/расширение фронта.
2. КРИЗИС: 2.1 системный финансовый кризис; 2.2 корпоративный дефолт крупного игрока;
   2.3 эпидемия/пандемия; 2.4 кибератака/инфраструктурный сбой; 2.5 бюджетный кризис РФ/секвестр;
   2.6 корп.отчётность хуже консенсуса.
3. КЛИМАТ: 3.1 наводнение/паводок; 3.2 ураган/шторм; 3.3 землетрясение/геофизика;
   3.4 экстремальная жара/засуха; 3.5 пожары крупного масштаба.
4. СТАВКА ЦБ: 4.1 резкое повышение КС (>=+200бп); 4.2 резкое снижение КС (>=-200бп);
   4.3 ястребиная риторика без изменения; 4.4 голубиная риторика без изменения;
   4.5 изменение нормативов/макропруденц.; 4.6 инфляционный сюрприз; 4.7 валютная интервенция.
5. ИНФОРМАЦИОННЫЙ ШУМ (нет реального действия/новой информации): 5.1 риторика без действия;
   5.2 ожидаемое решение/уже в ценах; 5.3 громкая угроза/ультиматум без эскалации; 5.4 календарный артефакт.
ВАЖНО: ожидаемое, полностью учтённое в ценах решение (даже снижение/повышение ставки) — это 5.2, не 4.x."""

CODE_RE = re.compile(r"(\d)\.(\d)")


def load_gold():
    return json.loads(GOLD_PATH.read_text(encoding="utf-8"))["items"]


def build_prompt(item):
    """Строит продакшен-промпт Agent 1 + инлайн-таксономию (reuse orchestrator)."""
    import orchestrator

    agent_path = TOOLS_DIR / "agents" / "agent_1_classifier.md"
    template = orchestrator.extract_prompt(agent_path)
    prompt = orchestrator.fill_prompt(
        template,
        {
            "НОВОСТЬ": item["news"],
            "ИСТОЧНИК": item["source"],
            "ДАТА": item["date"],
            "VAULT_ROOT": "(см. встроенную таксономию ниже)",
            "RADAR_ROOT": "(см. встроенную таксономию ниже)",
        },
    )
    return prompt + "\n\n" + TAXONOMY_INLINE


def parse_pred(raw_json):
    """Достаёт (main, sub) из ответа классификатора. sub='1.1', main='1'."""
    import orchestrator

    try:
        obj = orchestrator.extract_json(raw_json)
    except Exception:
        return None, None, {}
    sub_raw = str(obj.get("subcategory", "") or "")
    main_raw = str(obj.get("main_category", "") or "")
    m_sub = CODE_RE.search(sub_raw)
    sub = f"{m_sub.group(1)}.{m_sub.group(2)}" if m_sub else None
    # main: первая цифра из main_category, иначе из subcategory
    m_main = re.search(r"\d", main_raw)
    main = m_main.group(0) if m_main else (sub.split(".")[0] if sub else None)
    return main, sub, obj


def run(model_alias, limit=None):
    import anthropic
    import os

    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        print("ANTHROPIC_API_KEY не задан — нечего прогонять (CI-safe exit).", file=sys.stderr)
        return None
    model = MODEL_IDS.get(model_alias, model_alias)
    pi, po = PRICES.get(model, (0.0, 0.0))
    client = anthropic.Anthropic(api_key=key)
    items = load_gold()
    if limit:
        items = items[:limit]

    rows, tin, tout = [], 0, 0
    for it in items:
        prompt = build_prompt(it)
        resp = client.messages.create(
            model=model,
            max_tokens=1200,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text
        tin += resp.usage.input_tokens
        tout += resp.usage.output_tokens
        main, sub, _ = parse_pred(raw)
        rows.append(
            {
                "id": it["id"],
                "boundary": it["boundary"],
                "gold_main": it["gold_main"],
                "gold_sub": it["gold_sub"],
                "pred_main": main,
                "pred_sub": sub,
                "main_ok": main == it["gold_main"],
                "sub_ok": sub == it["gold_sub"],
            }
        )
        flag = "✓" if sub == it["gold_sub"] else ("~" if main == it["gold_main"] else "✗")
        print(
            f"  {flag} {it['id']}: gold {it['gold_sub']:<4} pred {str(sub):<5} "
            f"(main {main}/{it['gold_main']})",
            file=sys.stderr,
        )

    cost = tin / 1e6 * pi + tout / 1e6 * po
    n = len(rows)
    main_acc = sum(r["main_ok"] for r in rows) / n
    sub_acc = sum(r["sub_ok"] for r in rows) / n
    nb = [r for r in rows if not r["boundary"]]
    sub_acc_nb = sum(r["sub_ok"] for r in nb) / len(nb) if nb else 0.0
    summary = {
        "model": model,
        "n": n,
        "main_category_accuracy": round(main_acc, 4),
        "subcategory_accuracy": round(sub_acc, 4),
        "subcategory_accuracy_excl_boundary": round(sub_acc_nb, 4),
        "tokens_in": tin,
        "tokens_out": tout,
        "cost_usd": round(cost, 6),
        "rows": rows,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    safe = model.replace("/", "_")
    (OUT_DIR / f"{safe}_results.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 60, file=sys.stderr)
    print(f"model={model}  N={n}", file=sys.stderr)
    print(
        f"  main-category accuracy : {main_acc:.0%} ({sum(r['main_ok'] for r in rows)}/{n})",
        file=sys.stderr,
    )
    print(
        f"  subcategory accuracy   : {sub_acc:.0%} ({sum(r['sub_ok'] for r in rows)}/{n})",
        file=sys.stderr,
    )
    print(
        f"  subcat excl. boundary  : {sub_acc_nb:.0%} ({sum(r['sub_ok'] for r in nb)}/{len(nb)})",
        file=sys.stderr,
    )
    print(f"  tokens: {tin} in / {tout} out   cost: ${cost:.4f}", file=sys.stderr)
    return summary


def main():
    ap = argparse.ArgumentParser(description="L0 Agent-1 classifier eval")
    ap.add_argument("--model", default="haiku", help="haiku|sonnet|opus или полный id")
    ap.add_argument("--limit", type=int, default=None, help="ограничить число пунктов")
    args = ap.parse_args()
    run(args.model, args.limit)


if __name__ == "__main__":
    main()
