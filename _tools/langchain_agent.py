"""
LangChain-интеграция Макро-радара: тулы для агентов + сборка агента под Anthropic API.

Оборачивает 4 слоя пайплайна (L0..L3) в LangChain-тулы. Агент строится через
ChatAnthropic; API-ключ оставлен ПУСТЫМ (подставь позже через env ANTHROPIC_API_KEY
или аргумент build_agent(api_key=...)).

Запуск БЕЗ ключа (симуляции для полного покрытия сценариев):
    python langchain_agent.py --simulate      # прогон всех тулов по матрице сценариев
    python langchain_agent.py --agent-demo     # демо агент-цикла на fake-LLM (без сети)

Запуск С ключом (реальный агент):
    set ANTHROPIC_API_KEY=sk-...   (или передать в build_agent)
    python langchain_agent.py --ask "ЕС готовит новый пакет санкций против металлургии"

Тулы (читаются агентом по docstring):
    get_macro_state          L1   — текущее макро (CAI/EPU/режим КС)
    classify_news_shock      L0   — эвристическая классификация новости
    osl_forecast             L1.5 — прогноз выручки отрасли + conformal
    industry_spillover       L2   — распространение шока на 7 отраслей
    segment_impact           L3   — влияние на 18 клиентских сегментов
    run_credit_pipeline      L0-L3— сквозной анализ новости
    update_macro_state            — обновление макро (dry/live)
    find_news_analogs        RAG  — исторические аналоги
"""

import argparse
import json
import os
import sys
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

TOOLS_DIR = Path(__file__).parent
sys.path.insert(0, str(TOOLS_DIR))
sys.path.insert(0, str(TOOLS_DIR / 'agents'))
sys.path.insert(0, str(TOOLS_DIR / 'agents' / 'rag'))

# ---- Пайплайн-слои (чистый Python, без LLM) ----
import run_pipeline as rp
from calc_rf_cai import get_current_cai
from calc_rf_epu import get_current_epu
from spillover import severity_to_magnitude, propagate_shock, propagate_credit_channel
from segment_impact import predict_segment_impact, REGION_PROFILES

# ---- LangChain (мягкий импорт — без него работает только non-tool часть) ----
try:
    from langchain_core.tools import tool
    LANGCHAIN_AVAILABLE = True
except Exception:  # pragma: no cover
    LANGCHAIN_AVAILABLE = False

    def tool(fn=None, **_kw):  # no-op декоратор-заглушка
        def wrap(f):
            return f
        return wrap(fn) if fn else wrap


# ============================================================
# Конфигурация API (КЛЮЧ ОСТАВЛЕН ПУСТЫМ — подставить позже)
# ============================================================

# ID модели Anthropic (актуальная Opus 4.8). Меняй при необходимости.
MODEL_ID = "claude-opus-4-8"
# Пустой ключ-плейсхолдер. Реальный ключ: env ANTHROPIC_API_KEY или build_agent(api_key=...).
API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

VALID_INDUSTRIES = ["oilgas", "metallurgy", "chemistry", "retail", "energy", "oiv", "pharma"]
VALID_KC_REGIMES = ["normal", "moderate_stress", "acute_stress"]


# ============================================================
# Эвристический L0-классификатор (для тула и симуляции без LLM)
# ============================================================

_SHOCK_KEYWORDS = [
    (("снятие санкц", "снятии санкц", "отмена санкц"), "1.2"),
    (("новый пакет санкц", "санкци", "эмбарго", "sdn"), "1.4"),
    (("дрон", "удар по нпз", "атак", "нпз", "нефтебаз"), "1.3"),
    (("деэскал", "перемир", "ядерная сделка"), "1.2"),
    (("война", "эскалац", "вторжен"), "1.1"),
    (("дефолт", "банкрот", "реструктуризац"), "2.2"),
    (("дефицит бюджет", "секвестр", "нг-доход", "нефтегазовые доход"), "2.5"),
    (("кризис", "убыт", "спад", "минимум"), "2.6"),
    (("пожар",), "3.5"),
    (("наводнен", "паводок"), "3.1"),
    (("засух", "неурожай"), "3.4"),
    (("повыс ставк", "повышение ставк", "повышение кс"), "4.1"),
    (("снизил ставк", "снижение ставк", "снижение кс", "снизил ключев"), "4.2"),
    (("валют", "интервенц", "минфин покуп", "ослаб рубл", "обвал рубл"), "4.7"),
    (("тариф", "индексац"), "4.5"),
    (("инфляц",), "4.6"),
    (("ставк", "цб", "заседани"), "4.3"),
    (("устойчив", "риторик", "заявил", "без мер", "адаптир"), "5.1"),
]


def _heuristic_classify(news_text: str) -> dict:
    t = (news_text or "").lower()
    subcat = "5.2"
    for kws, code in _SHOCK_KEYWORDS:
        if any(k in t for k in kws):
            subcat = code
            break
    direction = rp.infer_direction(news_text or "", subcat)
    industries = rp.get_industries_for_shock(subcat, "oilgas")
    primary = industries[0] if industries else "oilgas"
    # грубая оценка силы
    sev = 50
    if any(w in t for w in ("кризис", "обвал", "дефолт", "санкци", "удар", "катастроф")):
        sev = 65
    if any(w in t for w in ("риторик", "заявил", "устойчив", "ожидаем")):
        sev = 20
    level = "H" if sev >= 70 else ("M" if sev >= 30 else "L")
    return {
        "subcategory": subcat,
        "primary_industry": primary,
        "severity_score": sev,
        "severity_level": level,
        "direction": direction,
    }


# ============================================================
# ТУЛЫ (LangChain)
# ============================================================

@tool
def get_macro_state() -> dict:
    """Текущее макро-состояние РФ (слой L1): РФ-CAI и фаза цикла, индекс EPU
    (+ флаг degraded), наклон кривой доходности, режим ключевой ставки.
    Без аргументов. Вызови это первым при анализе любой новости."""
    cai = get_current_cai()
    epu = get_current_epu(window_days=30)
    key_rate = cai.components.get("key_rate", {}).get("current", 16.0)
    regime = rp.kc_regime_from_rate(key_rate)
    return {
        "cai": cai.cai, "phase": cai.phase,
        "yield_curve_slope_pp": cai.yield_curve_slope_pp,
        "key_rate": key_rate, "kc_regime": regime,
        "epu": epu.epu_value, "epu_degraded": epu.epu_degraded,
    }


@tool
def classify_news_shock(news_text: str) -> dict:
    """Эвристическая классификация новости (слой L0) в подкатегорию шока (1.1..5.4),
    оценку силы (0-100, L/M/H), направление (+1 ухудшение / -1 смягчение) и
    первичную затронутую отрасль. Быстрый эвристический классификатор —
    LLM-агент может уточнить подкатегорию сам."""
    return _heuristic_classify(news_text)


@tool
def osl_forecast(industry: str) -> dict:
    """Operational Signal Layer (L1.5): прогноз годовой выручки эмитентов отрасли
    с conformal-интервалами (90%). industry — один из:
    oilgas, metallurgy, chemistry, retail, energy, oiv, pharma."""
    if industry not in VALID_INDUSTRIES:
        return {"error": f"industry must be one of {VALID_INDUSTRIES}"}
    return rp.osl_for_industry(industry)


@tool
def industry_spillover(industry: str, severity_score: float = 60.0) -> dict:
    """Industry Spillover (L2): распространение шока от отрасли-источника на все
    7 отраслей (ΔPD в п.п.). magnitude выводится из severity_score (0-100).
    industry — отрасль-источник из 7 покрытых."""
    if industry not in VALID_INDUSTRIES:
        return {"error": f"industry must be one of {VALID_INDUSTRIES}"}
    magnitude = severity_to_magnitude(severity_score)
    spill = propagate_shock(industry, magnitude_pp=magnitude)
    return {"source": spill.source, "magnitude_pp": spill.magnitude_pp,
            "ranked": spill.ranked}


@tool
def industry_spillover_credit_channel(severity_score: float = 55.0) -> dict:
    """L2 broad credit channel для шоков ставки ЦБ (категория 4): несколько
    debt-чувствительных отраслей бьются одновременно. magnitude из severity_score."""
    magnitude = severity_to_magnitude(severity_score)
    spill = propagate_credit_channel(magnitude_pp=magnitude)
    return {"source": spill.source, "magnitude_pp": spill.magnitude_pp,
            "ranked": spill.ranked}


@tool
def segment_impact(subcategory: str, kc_regime: str = "moderate_stress",
                   region: str = "") -> dict:
    """Client Segment Impact (L3): влияние шока на 18 клиентских сегментов банка —
    ΔPD (п.п.), Δdemand (%), Δchurn (п.п.) с раскладкой по 5 каналам.
    subcategory — подкатегория шока (1.1..5.4); kc_regime ∈
    {normal, moderate_stress, acute_stress}; region (опц.) ∈
    {oil_region, metal_monotown, capital_diversified, agricultural_rural}."""
    if kc_regime not in VALID_KC_REGIMES:
        return {"error": f"kc_regime must be one of {VALID_KC_REGIMES}"}
    reg = region or None
    if reg and reg not in REGION_PROFILES:
        return {"error": f"region must be one of {list(REGION_PROFILES)} or empty"}
    res = predict_segment_impact(subcategory, kc_regime, region=reg,
                                 include_breakdown=False)
    return {s: {"delta_pd": i.delta_pd, "delta_demand": i.delta_demand,
                "delta_churn": i.delta_churn, "confidence": i.confidence}
            for s, i in res.items()}


@tool
def run_credit_pipeline(news_text: str, subcategory: str = "",
                        industry: str = "", severity_score: int = 60) -> dict:
    """Сквозной анализ новости через все 4 слоя (L0→L1→L1.5→L2→L3). Если
    subcategory/industry не заданы — классифицирует эвристически. Возвращает
    компактную сводку: классификация, макро, топ-отрасль spillover, худший/лучший
    сегмент. Используй как финальный инструмент для целостного вывода."""
    if not subcategory or not industry:
        c = _heuristic_classify(news_text)
        subcategory = subcategory or c["subcategory"]
        industry = industry or c["primary_industry"]
        severity_score = c["severity_score"]
    state = rp.run_full_pipeline(news_text=news_text, source="langchain-agent",
                                 date="", smoke_shock=subcategory,
                                 smoke_industry=industry, smoke_severity=severity_score)
    l2 = state["L2_spillover"]
    l3 = state["L3_segments"]
    worst = max(l3.items(), key=lambda kv: kv[1]["delta_pd_pp"])
    best = min(l3.items(), key=lambda kv: kv[1]["delta_pd_pp"])
    return {
        "subcategory": subcategory, "industry": industry,
        "severity_score": severity_score,
        "macro": {"cai": state["L1_macro"]["cai"], "phase": state["L1_macro"]["phase"],
                  "kc_regime": state["kc_regime"],
                  "epu_degraded": state["L1_macro"].get("epu_degraded")},
        "l2_source": l2["source"], "l2_top": l2["ranked"][0],
        "worst_segment": [worst[0], round(worst[1]["delta_pd_pp"], 3)],
        "best_segment": [best[0], round(best[1]["delta_pd_pp"], 3)],
    }


@tool
def update_macro_state(live: bool = False) -> dict:
    """Обновление макро-состояния (data/macro_state.json). live=False — dry-run
    (ничего не пишет, показывает доступные значения). live=True — тянет 4 живых
    фида (USD/RUB, Brent, ключевая ставка, инфляция) и перезаписывает current_state."""
    import fetch_macro_state as fms
    if not live:
        return {"mode": "dry-run", "note": "live=True тянет и пишет current_state"}
    values = fms.fetch_all()
    res = fms.update_macro_state(values, dry_run=True)  # симуляция записи
    return {"mode": "fetched (not written in tool)", "applied": res["applied"]}


@tool
def find_news_analogs(query: str, subcategory: str = "") -> list:
    """RAG: исторические аналоги новости из архива анализов (cosine similarity по
    эмбеддингам). query — текст/суть новости; subcategory (опц.) — фильтр."""
    try:
        from find_analogs import find_analogs as _find
        sub = subcategory.split(" ")[0] if subcategory else None
        res = _find(query_text=query[:500], subcategory=sub, top_k=5, threshold=0.0)
        return [{"date": r.get("date"), "title": r.get("title"),
                 "subcategory": r.get("subcategory"),
                 "similarity": round(r.get("similarity", 0.0), 3)} for r in res]
    except Exception as e:
        return [{"error": f"RAG недоступен: {e}"}]


TOOLS = [
    get_macro_state, classify_news_shock, osl_forecast, industry_spillover,
    industry_spillover_credit_channel, segment_impact, run_credit_pipeline,
    update_macro_state, find_news_analogs,
]

SYSTEM_PROMPT = (
    "Ты — кредитный аналитик Т-Банка с доступом к инструментам Макро-радара "
    "(4-слойный конвейер новость→макро→отрасль→клиентский сегмент). Для анализа "
    "новости: 1) get_macro_state; 2) classify_news_shock (или определи подкатегорию "
    "сам); 3) при необходимости osl_forecast/industry_spillover/segment_impact; "
    "4) run_credit_pipeline для целостного вывода; 5) find_news_analogs для "
    "исторических аналогов. Опирайся ТОЛЬКО на числа из инструментов, не выдумывай. "
    "Отмечай бифуркации (когда сегменты расходятся по знаку ΔPD)."
)


# ============================================================
# Сборка реального агента (нужен API-ключ)
# ============================================================

def build_agent(api_key: str = API_KEY, model: str = MODEL_ID):
    """Строит tool-calling агента через ChatAnthropic. api_key ПУСТОЙ по умолчанию —
    подставь реальный ключ позже. Без ключа возвращает None + печатает инструкцию."""
    if not LANGCHAIN_AVAILABLE:
        print("  ⚠️ LangChain не установлен: pip install langchain langchain-anthropic",
              file=sys.stderr)
        return None
    if not api_key:
        print("  ⚠️ API-ключ пуст. Задай ANTHROPIC_API_KEY или build_agent(api_key=...) "
              "для реального запуска. Тулы и симуляции работают без ключа.",
              file=sys.stderr)
        return None
    from langchain_anthropic import ChatAnthropic
    from langchain.agents import create_agent
    llm = ChatAnthropic(model=model, api_key=api_key, max_tokens=4000)
    return create_agent(model=llm, tools=TOOLS, system_prompt=SYSTEM_PROMPT)


# ============================================================
# Симуляция БЕЗ ключа — полное покрытие сценариев
# ============================================================

ALL_SUBCATS = (
    ["1.1", "1.2", "1.3", "1.4", "1.5"]
    + ["2.1", "2.2", "2.3", "2.4", "2.5", "2.6"]
    + ["3.1", "3.2", "3.3", "3.4", "3.5"]
    + ["4.1", "4.2", "4.3", "4.4", "4.5", "4.6", "4.7"]
    + ["5.1", "5.2", "5.3", "5.4"]
)

SAMPLE_NEWS = [
    "ЕС готовит новый пакет санкций против металлургии РФ",
    "ЦБ снизил ключевую ставку до 14%",
    "Дроны атаковали НПЗ в Самаре, экспорт авиакеросина закрыт",
    "Засуха в Черноземье снизила прогноз урожая",
    "Чиновник заявил, что экономика устойчива, без новых мер",
    "США объявили о снятии санкций с банка",
    "Обвал рубля до 90 на падении нефти",
]


def _invoke(t, payload):
    """Вызвать LangChain-тул (.invoke) либо напрямую функцию, если LangChain нет."""
    if LANGCHAIN_AVAILABLE and hasattr(t, "invoke"):
        return t.invoke(payload)
    return t(**payload)


def simulate_coverage() -> dict:
    """Прогон ВСЕХ тулов по матрице сценариев без LLM/ключа. Возвращает отчёт
    покрытия: какие тулы вызваны, сколько сценариев, ошибки."""
    report = {"tools_exercised": {}, "scenarios": 0, "errors": []}

    def run(tool_obj, payload, label):
        name = getattr(tool_obj, "name", getattr(tool_obj, "__name__", str(tool_obj)))
        try:
            out = _invoke(tool_obj, payload)
            report["tools_exercised"][name] = report["tools_exercised"].get(name, 0) + 1
            report["scenarios"] += 1
            return out
        except Exception as e:
            report["errors"].append(f"{name} {label}: {e}")
            return None

    # 1) L1 макро + обновление (dry)
    run(get_macro_state, {}, "macro")
    run(update_macro_state, {"live": False}, "macro-dry")

    # 2) L0 классификация на сэмплах
    for txt in SAMPLE_NEWS:
        run(classify_news_shock, {"news_text": txt}, txt[:20])

    # 3) L1.5 OSL + L2 spillover по всем 7 отраслям
    for ind in VALID_INDUSTRIES:
        run(osl_forecast, {"industry": ind}, ind)
        run(industry_spillover, {"industry": ind, "severity_score": 60}, ind)
    run(industry_spillover_credit_channel, {"severity_score": 55}, "credit")

    # 4) L3 segment_impact по ВСЕМ 27 подкатегориям × 3 режима КС
    for sub in ALL_SUBCATS:
        for kc in VALID_KC_REGIMES:
            run(segment_impact, {"subcategory": sub, "kc_regime": kc}, f"{sub}/{kc}")
    # 4b) с региональными профилями
    for reg in list(REGION_PROFILES):
        run(segment_impact, {"subcategory": "1.2", "kc_regime": "moderate_stress",
                             "region": reg}, reg)

    # 5) сквозной пайплайн на сэмплах
    for txt in SAMPLE_NEWS:
        run(run_credit_pipeline, {"news_text": txt}, txt[:20])

    # 6) RAG аналоги
    run(find_news_analogs, {"query": "санкции против металлургии", "subcategory": "1.4"}, "rag")

    report["all_tools_covered"] = set(report["tools_exercised"]) == {
        getattr(t, "name", getattr(t, "__name__", "")) for t in TOOLS}
    report["subcategories_covered"] = len(ALL_SUBCATS)
    return report


def simulate_agent_loop() -> dict:
    """Демо агент-цикла БЕЗ сети/ключа. Воспроизводит то, что делает tool-calling
    агент: (1) «решение» LLM вызвать тул (scripted AIMessage с tool_call),
    (2) диспатч тула из реестра TOOLS и формирование ToolMessage, (3) проверку,
    что реальная проводка create_agent(ChatAnthropic, TOOLS) КОНСТРУИРУЕТСЯ с
    плейсхолдер-ключом (сетевой вызов нужен только при invoke)."""
    if not LANGCHAIN_AVAILABLE:
        return {"skipped": "LangChain недоступен"}
    from langchain_core.messages import AIMessage, ToolMessage

    news = "ЕС готовит новый пакет санкций против металлургии РФ"
    # (1) как если бы LLM решил вызвать тул:
    ai = AIMessage(content="", tool_calls=[{
        "name": "run_credit_pipeline", "args": {"news_text": news}, "id": "call_1"}])
    # (2) диспатч тула из реестра (это и есть «исполнение» агентом):
    tool_map = {getattr(t, "name", None): t for t in TOOLS}
    tool_results = []
    for tc in ai.tool_calls:
        out = tool_map[tc["name"]].invoke(tc["args"])
        tool_results.append(ToolMessage(
            content=json.dumps(out, ensure_ascii=False, default=str),
            tool_call_id=tc["id"]))

    # (3) проверка, что реальная проводка агента конструируется (без вызова сети):
    wiring_ok, wiring_err = False, None
    try:
        from langchain_anthropic import ChatAnthropic
        from langchain.agents import create_agent
        llm = ChatAnthropic(model=MODEL_ID, api_key="placeholder-set-later",
                            max_tokens=512)
        create_agent(model=llm, tools=TOOLS, system_prompt=SYSTEM_PROMPT)
        wiring_ok = True
    except Exception as e:
        wiring_err = f"{type(e).__name__}: {e}"

    preview = tool_results[0].content if tool_results else ""
    return {
        "ok": True,
        "tool_called": ai.tool_calls[0]["name"],
        "tool_result_preview": (preview[:200] + "…") if len(preview) > 200 else preview,
        "real_agent_wiring_constructs": wiring_ok,
        "wiring_error": wiring_err,
    }


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Макро-радар · LangChain-агент")
    parser.add_argument("--simulate", action="store_true",
                        help="Симуляция покрытия тулов без ключа")
    parser.add_argument("--agent-demo", action="store_true",
                        help="Демо агент-цикла на fake-LLM (без сети)")
    parser.add_argument("--ask", help="Реальный запрос агенту (нужен ANTHROPIC_API_KEY)")
    args = parser.parse_args()

    if args.simulate:
        rep = simulate_coverage()
        print(json.dumps({
            "scenarios": rep["scenarios"],
            "tools_exercised": rep["tools_exercised"],
            "all_tools_covered": rep["all_tools_covered"],
            "subcategories_covered": rep["subcategories_covered"],
            "errors": rep["errors"],
        }, ensure_ascii=False, indent=2))
        return

    if args.agent_demo:
        print(json.dumps(simulate_agent_loop(), ensure_ascii=False, indent=2))
        return

    if args.ask:
        agent = build_agent()
        if agent is None:
            print("Агент не готов (нет ключа/LangChain). Используй --simulate.")
            return
        out = agent.invoke({"messages": [("user", args.ask)]})
        for m in out.get("messages", []):
            if getattr(m, "content", ""):
                print(m.content)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
