"""
Multi-Agent Pipeline Orchestrator (v0.7).

Прогон 5 агентов на одной новости через Anthropic SDK:
  Agent 1 (Classifier) → Agent 2 (Context-RAG) → Agent 3 (Backtest-Analog)
  → Agent 4 (Impact) → Agent 5 (Summary)

Особенности:
  - JSON-state передаётся между агентами
  - Перед Agent 3 вызывается RAG find_analogs() и аналоги inject в промпт
  - shortcut=true (информационный шум) пропускает Agent 2-4
  - Финальный markdown сохраняется в _Анализы/<YYYY-MM-DD> — <slug>.md
  - После сохранения автоматически вызывается index_news для повторной индексации

Использование:
  python orchestrator.py --news-file news.txt --source "ТАСС" --date 2026-04-26
  cat news.txt | python orchestrator.py --source "ТАСС" --date 2026-04-26
  python orchestrator.py --news-file news.txt --bifurcation --dry-run
"""

import argparse
import json
import os
import re
import sys
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

AGENTS_DIR = Path(__file__).parent
TOOLS_DIR = AGENTS_DIR.parent  # _tools/
RAG_DIR = AGENTS_DIR / "rag"
RADAR_ROOT = AGENTS_DIR.parent.parent  # _tools/agents/orchestrator.py → Макро-радар/
ANALYSES_DIR = RADAR_ROOT / "_Анализы"

# S2.4: путь к _tools для импорта run_pipeline (реальные OSL-числа для Agent 4)
sys.path.insert(0, str(TOOLS_DIR))

DEFAULT_MODEL = "opus"  # alias подходит и для CLI (--model opus), и для SDK
TIMEOUT_SECONDS = 300
DEFAULT_LLM_MODE = "cli"  # 'cli' | 'sdk' | 'dry-run'

# S1.4: минимальное косинусное сходство для RAG-аналогов. Было 0.0 (фильтр
# отключён → вся БД уходила в Agent 3). TF-IDF даёт низкие cos, поэтому порог
# умеренный; при переходе на нейроэмбеддинги (S3.3) можно поднять до ~0.5.
RAG_MIN_SIMILARITY = 0.15
RAG_MAX_ANALOGS = 5  # cap на число инжектируемых аналогов (защита контекста)

# Подключаем RAG-модуль для прямого Python-импорта
sys.path.insert(0, str(RAG_DIR))


# ============================================================
# Промпт-парсер
# ============================================================


def extract_prompt(agent_md_path: Path) -> str:
    """Извлекает блок Промпта из markdown-файла агента.
    Блок — между ## Промпт и закрывающим ``` (тройной бэктик)."""
    text = agent_md_path.read_text(encoding="utf-8")
    # Ищем секцию ## Промпт ... ```...```
    match = re.search(r"##\s+Промпт\s*\n+```(?:\w+)?\s*\n(.*?)```", text, re.DOTALL)
    if not match:
        raise ValueError(f"Не нашёл блок ## Промпт в {agent_md_path.name}")
    return match.group(1).strip()


def fill_prompt(template: str, substitutions: dict) -> str:
    """Простая подстановка <ключ> → значение."""
    out = template
    for k, v in substitutions.items():
        if isinstance(v, (dict, list)):
            v = json.dumps(v, ensure_ascii=False, indent=2)
        out = out.replace(f"<{k}>", str(v))
    return out


def _balance_braces(candidate: str) -> str:
    """Достраивает закрывающие скобки/кавычки для обрезанного JSON (S1.5).

    Идёт по строке, учитывая строковые литералы и экранирование, и добавляет
    недостающие ", ] и } в конец — чтобы каскадно не падать на truncated-ответе LLM.
    """
    in_str = False
    esc = False
    depth_obj = 0
    depth_arr = 0
    for ch in candidate:
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth_obj += 1
        elif ch == "}":
            depth_obj -= 1
        elif ch == "[":
            depth_arr += 1
        elif ch == "]":
            depth_arr -= 1
    tail = ""
    if in_str:
        tail += '"'
    tail += "]" * max(0, depth_arr)
    tail += "}" * max(0, depth_obj)
    return candidate + tail


def extract_json(text: str) -> dict:
    """Извлекает JSON из ответа LLM (может быть обёрнут в ```json ... ```).

    При обрезанном/неполном JSON пытается достроить закрывающие скобки
    через _balance_braces, чтобы не давать каскадный отказ (S1.5)."""
    # 1) ```json ... ``` — допускаем и незакрытый code-fence
    match = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
    if match:
        candidate = match.group(1).strip()
    else:
        match = re.search(r"```(?:json)?\s*\n(.*)", text, re.DOTALL)
        candidate = match.group(1).strip() if match else text.strip()
    # 2) Найти первую { и последнюю }
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start >= 0 and end > start:
        candidate = candidate[start : end + 1]
    elif start >= 0:
        candidate = candidate[start:]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return json.loads(_balance_braces(candidate))


# ============================================================
# Anthropic SDK wrapper
# ============================================================


def _stub_response() -> str:
    return (
        '{"dry_run": true, "WHAT": "stub", "main_category": "1.1", '
        '"subcategory": "1.1", "severity_score": 50, "shortcut": false}'
    )


def call_llm_cli(prompt: str, model: str = DEFAULT_MODEL) -> str:
    """Вызов Claude через `claude -p` CLI (использует подписку, не API key)."""
    cmd = ["claude", "-p", prompt, "--model", model, "--output-format", "text"]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
            encoding="utf-8",
        )
    except FileNotFoundError:
        raise EnvironmentError(
            "`claude` CLI не найден в PATH. Установите Claude Code "
            "или используйте --llm-mode sdk c ANTHROPIC_API_KEY."
        )
    except subprocess.TimeoutExpired:
        raise TimeoutError(f"`claude` CLI превысил {TIMEOUT_SECONDS}s")
    if result.returncode != 0:
        raise RuntimeError(
            f"`claude` CLI failed (code {result.returncode}): " f"{result.stderr[:500]}"
        )
    return result.stdout


def call_llm_sdk(prompt: str, model: str = DEFAULT_MODEL, max_tokens: int = 8000) -> str:
    """Прямой вызов Anthropic API через SDK (требует ANTHROPIC_API_KEY)."""
    try:
        import anthropic
    except ImportError:
        raise ImportError("anthropic SDK не установлен. Run: pip install anthropic")
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY не задан. Используйте --llm-mode cli "
            "(работает через подписку Claude Code) либо задайте API key."
        )
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def call_llm(
    prompt: str, model: str = DEFAULT_MODEL, mode: str = DEFAULT_LLM_MODE, max_tokens: int = 8000
) -> str:
    """Унифицированный вызов LLM. mode: 'cli' | 'sdk' | 'dry-run'."""
    if mode == "dry-run":
        return _stub_response()
    if mode == "cli":
        return call_llm_cli(prompt, model)
    if mode == "sdk":
        return call_llm_sdk(prompt, model, max_tokens)
    raise ValueError(f"Unknown llm mode: {mode}")


# ============================================================
# RAG интеграция
# ============================================================


def fetch_analogs(
    query_text: str, top_k: int = 5, subcategory_filter: Optional[str] = None
) -> list[dict]:
    """Прямой Python-импорт find_analogs из rag/ (быстрее subprocess)."""
    try:
        from find_analogs import find_analogs as _find  # noqa
    except ImportError:
        return []
    try:
        results = _find(
            query_text=query_text[:500],
            subcategory=subcategory_filter,
            top_k=min(top_k, RAG_MAX_ANALOGS),
            threshold=RAG_MIN_SIMILARITY,  # S1.4: фильтрация включена
        )
        # Чистим непригодные для JSON-сериализации поля
        clean = []
        for r in results:
            clean.append(
                {
                    k: v
                    for k, v in r.items()
                    if isinstance(v, (str, int, float, bool, type(None), list, dict))
                }
            )
        return clean
    except Exception as e:
        print(f"    ⚠️ RAG fetch error: {e}", file=sys.stderr)
        return []


def reindex_rag(analysis_path: Path) -> bool:
    """Запускает index_news.py для свежесозданного анализа."""
    index_script = RAG_DIR / "index_news.py"
    if not index_script.exists():
        return False
    try:
        result = subprocess.run(
            [sys.executable, str(index_script), "--file", str(analysis_path)],
            capture_output=True,
            text=True,
            timeout=60,
            encoding="utf-8",
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


# ============================================================
# Pipeline
# ============================================================


def compute_osl_results(subcat: Optional[str]) -> dict:
    """S2.4: реальные OSL/conformal-числа для затронутых отраслей (для Agent 4).

    subcat — подкатегория шока (например '1.2'). Переиспользует маршрутизацию и
    osl_for_industry из run_pipeline. При сбоях возвращает {'error': ...},
    не валя весь конвейер.
    """
    try:
        import run_pipeline as rp
    except Exception as e:
        return {"error": f"osl import failed: {e}"}
    sc = (subcat or "5.1").split(" ")[0]
    try:
        industries = rp.get_industries_for_shock(sc, "oilgas")
    except Exception:
        industries = ["oilgas"]
    out = {"subcategory": sc, "industries": industries}
    for ind in industries[:3]:
        try:
            out[ind] = rp.osl_for_industry(ind)
        except Exception as e:
            out[ind] = {"error": str(e)}
    if "oilgas" in industries:
        scen = rp.BRENT_SCENARIOS.get(sc, rp.BRENT_SCENARIOS.get("default", {}))
        pre = scen.get("pre") or 78.0
        post = scen.get("post") or 78.0
        try:
            out["oilgas_forward"] = rp.osl_oilgas_forward_scenarios(pre, post)
        except Exception as e:
            out["oilgas_forward"] = {"error": str(e)}
    return out


def run_agent(
    agent_num: int,
    substitutions: dict,
    model: str,
    llm_mode: str = DEFAULT_LLM_MODE,
    parse_json: bool = True,
) -> Any:
    """Запуск одного агента: загрузка промпта + подстановка + вызов LLM."""
    agent_files = {
        1: "agent_1_classifier.md",
        2: "agent_2_context_rag.md",
        3: "agent_3_backtest_analog.md",
        4: "agent_4_impact.md",
        5: "agent_5_summary.md",
    }
    agent_path = AGENTS_DIR / agent_files[agent_num]
    template = extract_prompt(agent_path)
    # S1.1: подставляем корни хранилища вместо захардкоженных путей в промптах.
    # <VAULT_ROOT> = D:\...\claudeT, <RADAR_ROOT> = .../Макро-радар.
    substitutions = {
        "VAULT_ROOT": str(RADAR_ROOT.parent),
        "RADAR_ROOT": str(RADAR_ROOT),
        **substitutions,
    }
    prompt = fill_prompt(template, substitutions)

    print(f"  -> Agent {agent_num} ({agent_files[agent_num]})...", file=sys.stderr)
    response = call_llm(prompt, model=model, mode=llm_mode)
    if parse_json:
        try:
            return extract_json(response)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"    [WARN] JSON parse error: {e}", file=sys.stderr)
            # S1.5: явный флаг сбоя вместо тихой передачи битого state дальше.
            return {"_raw": response, "_error": str(e), "_l0_failed": True}
    return response


def run_pipeline(
    news_text: str,
    source: str,
    date: str,
    bifurcation: bool = False,
    model: str = DEFAULT_MODEL,
    llm_mode: str = DEFAULT_LLM_MODE,
    dry_run: bool = False,
) -> tuple[dict, str]:
    """Полный pipeline. Возвращает (final_state_json, markdown_text).

    llm_mode: 'cli' (default — через `claude -p`) | 'sdk' | 'dry-run'
    dry_run: legacy флаг, эквивалентно llm_mode='dry-run'.
    """
    if dry_run:
        llm_mode = "dry-run"

    # Agent 1
    state = run_agent(
        1,
        {
            "НОВОСТЬ": news_text,
            "ИСТОЧНИК": source,
            "ДАТА": date,
        },
        model=model,
        llm_mode=llm_mode,
    )

    if state.get("shortcut"):
        print("  -> shortcut=true: skip Agent 2-4", file=sys.stderr)
        markdown = run_agent(
            5,
            {"STATE_JSON": state, "MODE": "shortcut"},
            model=model,
            llm_mode=llm_mode,
            parse_json=False,
        )
        return state, markdown

    # Agent 2
    state = run_agent(2, {"STATE_JSON": state}, model=model, llm_mode=llm_mode)

    # Agent 3: pre-fetch RAG
    query_text = state.get("WHAT", news_text[:200])
    subcat = state.get("subcategory", "").split(" ")[0] if state.get("subcategory") else None
    analogs = fetch_analogs(query_text, top_k=5, subcategory_filter=subcat)
    state["rag_analogs"] = analogs
    print(f"    RAG analogs found: {len(analogs)}", file=sys.stderr)

    state = run_agent(
        3, {"STATE_JSON": state, "RAG_ANALOGS": analogs}, model=model, llm_mode=llm_mode
    )

    # Agent 4: S2.4 — считаем реальные OSL/conformal-числа и инжектируем в промпт,
    # чтобы LLM использовал расчёты, а не фабриковал прогнозы выручки.
    state["_bifurcation_mode"] = bifurcation
    osl_results = compute_osl_results(subcat)
    state["osl_results"] = osl_results
    state = run_agent(
        4,
        {"STATE_JSON": state, "BIFURCATION": bifurcation, "OSL_RESULTS": osl_results},
        model=model,
        llm_mode=llm_mode,
    )

    # Agent 5
    markdown = run_agent(5, {"STATE_JSON": state}, model=model, llm_mode=llm_mode, parse_json=False)

    return state, markdown


# ============================================================
# Slug + сохранение
# ============================================================


def slugify(text: str, max_len: int = 50) -> str:
    """Простой slug для имени файла."""
    text = re.sub(r"[^\w\sА-Яа-яЁё-]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_len]


def save_analysis(markdown: str, state: dict, date: str) -> Path:
    """Сохраняет анализ в _Анализы/<date> — <slug>.md."""
    ANALYSES_DIR.mkdir(exist_ok=True)
    title = state.get("WHAT", "analysis")[:80]
    slug = slugify(title)
    file_path = ANALYSES_DIR / f"{date} — {slug}.md"
    counter = 1
    while file_path.exists():
        file_path = ANALYSES_DIR / f"{date} — {slug} ({counter}).md"
        counter += 1
    file_path.write_text(markdown, encoding="utf-8")
    return file_path


# ============================================================
# CLI
# ============================================================


def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Pipeline для Макро-радара (v0.7)")
    parser.add_argument("--news-file", help="Файл с текстом новости")
    parser.add_argument("--source", required=True, help="Источник новости (URL или издание)")
    parser.add_argument(
        "--date", default=datetime.now().strftime("%Y-%m-%d"), help="Дата новости (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--bifurcation",
        action="store_true",
        help="Региональная бифуркация (например, Краснодарский край)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model id или alias (default: {DEFAULT_MODEL}). "
        f"Для CLI: opus|sonnet|haiku или полное имя.",
    )
    parser.add_argument(
        "--llm-mode",
        choices=["cli", "sdk", "dry-run"],
        default=DEFAULT_LLM_MODE,
        help="Способ вызова LLM: "
        "cli (default) — через `claude -p` (подписка), "
        "sdk — через Anthropic API (нужен ANTHROPIC_API_KEY), "
        "dry-run — без API.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Алиас --llm-mode dry-run (legacy)")
    parser.add_argument("--no-save", action="store_true", help="Не сохранять результат в _Анализы/")
    parser.add_argument(
        "--no-reindex", action="store_true", help="Не запускать index_news после сохранения"
    )
    args = parser.parse_args()

    # Чтение новости
    if args.news_file:
        news_text = Path(args.news_file).read_text(encoding="utf-8").strip()
    else:
        news_text = sys.stdin.read().strip()
    if not news_text:
        sys.exit("ERROR: пустая новость")

    llm_mode = "dry-run" if args.dry_run else args.llm_mode
    print("=" * 70, file=sys.stderr)
    print(f"  Multi-Agent Pipeline · model={args.model} · llm_mode={llm_mode}", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    print(f"  Источник: {args.source}", file=sys.stderr)
    print(f"  Дата: {args.date}", file=sys.stderr)
    print(f"  Длина текста: {len(news_text)} символов", file=sys.stderr)
    if args.bifurcation:
        print("  ⚡ Bifurcation mode", file=sys.stderr)
    print("-" * 70, file=sys.stderr)

    state, markdown = run_pipeline(
        news_text,
        args.source,
        args.date,
        bifurcation=args.bifurcation,
        model=args.model,
        llm_mode=llm_mode,
    )

    print("-" * 70, file=sys.stderr)
    if not args.no_save:
        path = save_analysis(markdown, state, args.date)
        print(f"  ✅ Сохранено: {path.relative_to(RADAR_ROOT)}", file=sys.stderr)
        if not args.no_reindex:
            ok = reindex_rag(path)
            print(f'  RAG re-index: {"✅" if ok else "⚠️ skipped"}', file=sys.stderr)
    else:
        # Печатаем markdown в stdout
        print(markdown)


if __name__ == "__main__":
    main()
