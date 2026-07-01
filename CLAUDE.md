# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> Гайд по работе с кодовой базой Макро-радара (claude.ai/code).

---

## Что это

Аналитический пайплайн «новость → макро → отрасль → клиентский сегмент банка».
Чистый Python в `_tools/` + markdown-промпты агентов + Obsidian-заметки вокруг.
Текущая версия **v0.9.x**, `pipeline_version='0.9'`.

**Данные:** только публичные/иллюстративные источники; клиентских и портфельных данных банка в репозитории нет.

---

## Команды

Весь Python живёт в `_tools/`. **Сначала `cd _tools`** — скрипты делают
`sys.path.insert` относительно своего расположения и грузят `data/*.json` по
относительным путям.

```bash
cd _tools

# Тесты (269 собирается, 0 skipped)
python -m pytest tests/ -v
python -m pytest tests/test_v09.py -v                          # один файл
python -m pytest tests/test_l1_l2_l3.py::test_churn_always_positive -v   # один тест
python -m pytest tests/ -k bifurcation -v                     # по подстроке имени

# Линтеры (мягкий режим: E501/E741/W605 игнорятся, line-length 100)
ruff check .
black --check .          # black . чтобы применить

# OSL по отрасли (7 модулей; тиринг по доступности данных — 4 валидированы / 3 иллюстративны,
# см. docs/COVERAGE_TIERS.md: metallurgy/oilgas/chemistry/energy валидированы; pharma/retail/oiv — нет)
python osl_metallurgy.py
python osl_oilgas.py --company Газпром

# Макро-индексы и состояние
python fetch_macro_state.py        # перезаписывает ТОЛЬКО current_state (4 живых фида)
python calc_rf_cai.py
python calc_rf_epu.py              # по корпусу _Анализы/
python calc_rf_epu.py --source fred

# Conformal-интервалы / калибровка
python conformal_prediction.py --industry all
python osl_calibrator.py --module all      # регламент: _tools/CALIBRATION_GUIDE.md

# Полный пайплайн
python run_pipeline.py --smoke-shock 4 --smoke-industry oilgas   # без LLM (smoke)
python run_pipeline.py --news-file news.txt --source "ТАСС" --date 2026-06-24
python batch_run.py --date 2026-06-14                            # пакетный прогон

# Multi-agent оркестратор (L0). По умолчанию --llm-mode cli (через claude CLI, без API key)
echo "<текст новости>" | python agents/orchestrator.py --source "ТАСС" --date 2026-06-24
python agents/orchestrator.py --news-file news.txt --source "..." --date 2026-06-24 --llm-mode dry-run --no-save
```

CI: `.github/workflows/test.yml` (в корне репозитория). Триггер на `_tools/**`
(+ `requirements.lock`, `Dockerfile`, сам workflow). 4 job с `RADAR_RAG_USE_ST=0`
(TF-IDF, без сети и тяжёлых моделей): **tests** (матрица py3.11/3.12 — ruff-гейт +
black-гейт + `pytest --cov` с порогом ≥60%), **smoke** (e2e L0→L3), **docker**
(clean-clone build), **security** (gitleaks secret-scan + pip-audit).

---

## Архитектура

Пять слоёв (L0→L3, L1.5 между L1 и L2). State — это **dict-JSON, который растёт от
слоя к слою**; `run_pipeline.run_full_pipeline()` оркестрирует, `render_markdown()`
рендерит итог в `_Анализы/`.

| Слой | Файлы | Что делает | Ключевые данные |
|---|---|---|---|
| **L0** Фильтр новостей | `agents/orchestrator.py`, `agents/agent_*.md`, `agents/rag/` | 5 LLM-агентов (Classifier→Context-RAG→Backtest-Analog→Impact→Summary); классификация в 27 подкатегорий шоков | `data/shock_to_industries.json` |
| **L1** Макро-состояние | `calc_rf_cai.py`, `calc_rf_epu.py`, `fetch_macro_state.py` | РФ-CAI (z-score 6 показателей), EPU (Baker-Bloom-Davis по корпусу), режим КС | `data/macro_state.json` |
| **L1.5** Опер-сигнал (OSL) | `osl_*.py` (7), `osl_common.py`, `conformal_prediction.py`, `osl_calibrator.py` | Прогноз годовой выручки эмитентов из физданных×цены; опережает МСФО на 2–3 мес.; conformal-интервалы | `calibration/*_calibrated.json` |
| **L2** Spillover | `spillover.py` | Распространение шока по матрице отраслей 7×7 (Fialkowski); credit-channel для шоков ЦБ | `data/spillover_matrix.json` |
| **L3** Сегменты | `segment_impact.py` | ΔPD/Δdemand/Δchurn по 10 сегментам через **5 каналов** (consumer/oil_revenue/fiscal/fx/supply_chain) × КС-amplifier; региональные профили | `data/segment_impact_table_v0_8.json` |

`langchain_agent.py` — альтернативная обёртка тех же слоёв в LangChain-тулы (опц.,
extra `agent`); работает без ключа в `--simulate`/`--agent-demo`.

### Поток данных (важные инварианты)

- **L3 — channel decomposition, не глобальный direction±1.** Один шок может
  одновременно ухудшать одни сегменты и улучшать другие (бифуркация): деэскалация
  Brent↓ улучшает потребсегменты, но **ухудшает** нефтегаз-корпов и нефтегаз-регионы.
  Тесты `test_bifurcation_*` это закрепляют — не «чините» их в сторону «все ΔPD одного знака».
- **Активная таблица сегментов — `segment_impact_table_v0_8.json`** (`DATA_PATH`).
  `segment_impact_table.json` — legacy-совместимость (`LEGACY_DATA_PATH`).
- **Conformal-покрытие — IN-SAMPLE.** Считается на захардкоженных `ACTUAL_*` в
  `osl_*.py`, не на out-of-sample. Не утверждать «out-of-sample 96%». Честный
  hold-out — единственный `skipped` тест (ждёт независимых 9M actuals).
- **Маршрутизация шоков — данные, не код.** Все 27 подкатегорий → отрасли в
  `data/shock_to_industries.json`. Не зашивать в inline-словарь.

### Кросс-слойные правила (легко сломать)

- **`run_pipeline.py --json`: stdout должен быть чистым JSON.** Любой import-time
  `print` обязан идти в **stderr** (см. `conformal_prediction.py`). Есть тест-контракт
  `test_run_pipeline_json_stdout_is_clean`.
- **Кириллица в Windows-shell:** скрипты делают `sys.stdout.reconfigure(encoding='utf-8')`
  в начале — сохранять при добавлении новых entry-points, иначе кракозябры/падение.
- **Промпты агентов парсятся из markdown.** `orchestrator.extract_prompt()` берёт блок
  между `## Промпт` и закрывающим ``` в `agents/agent_*.md`. Менять промпт = править
  этот блок, а не Python.
- **Пути в промптах — плейсхолдер `<VAULT_ROOT>`** (подставляется в `run_agent()`).
  Никаких абсолютных `C:\Users\...` в `agent_*.md`.
- **`fetch_macro_state.py` пишет ТОЛЬКО `current_state`** в `macro_state.json` (4 фида:
  USD/RUB cbr-xml-daily, Brent Yahoo `BZ=F`, KeyRate CBR SOAP, inflation World Bank);
  `baseline_state` не трогает; источник недоступен → значение не меняется (graceful).
- **RAG-эмбеддер выбирается одним env-флагом `RADAR_RAG_USE_ST`** (default `0` = TF-IDF;
  `1` = sentence-transformers e5-small). Индексация и запрос должны быть в одном
  пространстве — менять флаг = реиндексировать. `index_single()` делает UPSERT (не
  стирает корпус). Порог сходства — `RAG_MIN_SIMILARITY` (0.15 для TF-IDF).

### Добавление новой OSL-отрасли

1. `osl_<отрасль>.py` (импорт `RevenuePredict`/`FXRate`/`mae_pct` из `osl_common.py`,
   не копировать схему).
2. Зарегистрировать модуль в `pyproject.toml → [tool.setuptools] py-modules`.
3. Добавить колонку/строку в `data/spillover_matrix.json` (L2) и каналы в
   `segment_impact_table_v0_8.json` при необходимости.
4. Калибровка → `calibration/osl_<отрасль>_calibrated.json` (`osl_calibrator.py`).
5. Smoke-тест в `tests/` (см. `test_osl_smoke.py` как шаблон).

---

## Документация для углубления

- `_tools/README.md` — source of truth по модулям/командам/статусу (держать в синхроне).
- `_tools/CALIBRATION_GUIDE.md` — регламент калибровки OSL.
- `CHANGELOG.md` — версии промпта и кода (v1.0→v0.9); читать перед изменением логики слоя.
- `docs/4_Tech_Architecture.md` — детальная архитектура; `docs/3_User_Guide_Analyst.md` — для аналитика.
- `Макро-радар — Хаб.md` — навигатор проекта и статус фаз.

## Ручной разбор новости (без кода)

Промпт готов в `Промпт — Анализ новости.md`. Результат сохранять как
`_Анализы/YYYY-MM-DD — <короткое название>.md` (батч-прогоны → `_Анализы/_batch/`).
