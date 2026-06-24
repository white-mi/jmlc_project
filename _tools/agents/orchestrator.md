---
type: orchestrator
agents: 5
---

# Orchestrator — координация Multi-Agent Pipeline

> [[README|← README]] · [[../../Промпт — Анализ новости|← Одиночный промпт v1.1]]

## Назначение

Координировать последовательный запуск 5 агентов на одной новости. Используется **вручную** через Claude Code в v1.0; в v2.0 — автоматизированный Python-скрипт через Anthropic SDK.

## Workflow (v1.0 ручной режим)

### Шаг 0 — Решение использовать ли pipeline

```
Если новость:
  - Содержит композитные шоки (>1 категория) → ✅ pipeline
  - Касается M+/L клиентов / системных эффектов → ✅ pipeline
  - Сила шока ожидается ≥ M (≥30/100) → ✅ pipeline
  - Иначе → одиночный промпт v1.1 (быстрее и дешевле)
```

### Шаг 1 — Запустить Agent 1 (Classifier)

```
В Claude Code:
1. Открой agent_1_classifier.md
2. Скопируй блок "Промпт"
3. Заполни ВХОД: НОВОСТЬ, ИСТОЧНИК, ДАТА
4. Запусти
5. Сохрани JSON-ответ в clipboard
```

Если в ответе `"shortcut": true` → пропусти к шагу 5 (информационный шум).

### Шаг 2 — Запустить Agent 2 (Context-RAG)

```
1. Открой agent_2_context_rag.md
2. Подставь JSON от Agent 1 в ВХОД
3. Запусти — Agent 2 прочитает отраслевые отчёты vault
4. Сохрани расширенный JSON
```

### Шаг 3 — Запустить Agent 3 (Backtest-Analog)

```
1. Открой agent_3_backtest_analog.md
2. Подставь JSON от Agent 2
3. Запусти — Agent 3 прочитает _Анализы/ и журналы отраслей
4. Сохрани расширенный JSON
```

### Шаг 4 — Запустить Agent 4 (Impact)

```
1. Открой agent_4_impact.md
2. Подставь JSON от Agent 3
3. Запусти — Agent 4 применит карту L1/L1.5/L2/L3
4. Сохрани финальный JSON
```

### Шаг 5 — Запустить Agent 5 (Summary)

```
1. Открой agent_5_summary.md
2. Подставь финальный JSON от Agent 4
3. Запусти — Agent 5 создаст markdown-документ
4. Сохрани в _Анализы/<YYYY-MM-DD> — <короткое название>.md
```

## Workflow (v2.0 автоматизированный — реализован в `orchestrator.py`)

Реализация: `_tools/agents/orchestrator.py`

**Способы вызова LLM (`--llm-mode`):**

| Mode | Кто вызывает | Платёж | Когда использовать |
|---|---|---|---|
| **`cli`** (default) | `claude -p` (Claude Code) | подписка Pro/Max | когда вы залогинены в Claude Code |
| `sdk` | `anthropic` Python SDK | API platform billing | когда нужен прямой API + ANTHROPIC_API_KEY |
| `dry-run` | stub-ответы | бесплатно | smoke-тесты, отладка цепочки |

**Использование:**

```bash
# По умолчанию через подписку Claude Code (никаких API keys)
echo "<текст новости>" | python orchestrator.py --source "ТАСС" --date 2026-04-26

# Из файла
python orchestrator.py --news-file news.txt --source "Коммерсантъ" --date 2026-04-26

# Через прямой API (если есть ANTHROPIC_API_KEY)
python orchestrator.py --news-file news.txt --source "..." --date 2026-04-26 --llm-mode sdk

# Бифуркация для региональных шоков (Туапсе≠Сочи)
python orchestrator.py --news-file news.txt --source "..." --date 2026-04-26 --bifurcation

# Smoke-тест без вызовов LLM (stub-ответы)
python orchestrator.py --news-file news.txt --source "test" --date 2026-04-26 --llm-mode dry-run --no-save
```

**Что делает скрипт:**
1. Парсит блок `## Промпт ... ` из каждого `agent_*.md` (extract via regex)
2. Передаёт state-JSON между агентами (выход агента N → вход агента N+1)
3. Перед Agent 3 вызывает `find_analogs()` из `rag/` (top_k=5, threshold=0.0) и подсовывает аналоги в промпт
4. Если Agent 1 вернул `shortcut=true` — пропускает агенты 2-4, сразу к Agent 5
5. Сохраняет результат в `_Анализы/<date> — <slug>.md`
6. Запускает `index_news.py` для повторной индексации в RAG

**Зависимости:**
- Для `--llm-mode cli` (default): установленный `claude` CLI (Claude Code), залогиненный аккаунт.
- Для `--llm-mode sdk`: `pip install anthropic` + `ANTHROPIC_API_KEY` в окружении.
- RAG модуль (`rag/find_analogs.py`, `rag/index_news.py`) — опционально (gracefully degrades).

**Параметры командной строки:**
- `--news-file PATH` / stdin
- `--source TEXT` (обязательный)
- `--date YYYY-MM-DD` (default: today)
- `--bifurcation` — региональная разбивка
- `--model ID` (default: `opus`; alias подходит и для CLI, и для SDK)
- `--llm-mode {cli,sdk,dry-run}` — способ вызова LLM (default: `cli`)
- `--dry-run` — алиас `--llm-mode dry-run`
- `--no-save` — не сохранять в `_Анализы/`
- `--no-reindex` — не вызывать `index_news.py`

## Версионирование

| Версия | Дата | Изменения |
|---|---|---|
| v1.0 | 2026-04-25 | Базовая архитектура 5 агентов; ручной режим через Claude Code |
| v2.0 | 2026-04-25 | ✅ Python orchestrator + Anthropic SDK + автоиндексация RAG |
| v2.1 (план) | октябрь 2026 | Параллельный запуск Agent 2 и 3 |
| v2.2 (план) | 2027 | Замена TF-IDF на multilingual-e5-small для лучшего semantic match |
| v3.0 (план) | 2027 | Self-критика: Agent 4.5 ревьюит выводы Agent 4 |

---

*Orchestrator v1.0 · 2026-04-25 · Ручной workflow для Multi-Agent Pipeline*
