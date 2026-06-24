---
tags: [макро-радар, docs, architecture, для-IT]
дата: "2026-06-15"
аудитория: "Tech lead / архитектор / новый разработчик"
---

# 📙 Техническая архитектура проекта

> [[../Макро-радар — Хаб|← Хаб]] · [[1_Macro_Knowledge_for_IT|← Макро-знания]] · [[2_Tech_Knowledge_for_IT|← Тех-знания]] · [[3_User_Guide_Analyst|← User Guide]]

Полная техническая документация проекта «Макро-радар». Архитектура, модули, API, схемы данных, deployment, тестирование.

---

## 1. Высокоуровневая архитектура

```
┌────────────────────────────────────────────────────────────┐
│                       USER (Analyst)                        │
│  Open: Промпт — Анализ новости.md                          │
│  Provide: новость + источник + дата                        │
└────────────────────────┬───────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────────┐
│                 CLAUDE CODE (LLM)                           │
│  Reads:                                                     │
│    - Промпт — Анализ новости.md                            │
│    - _Справочники/* (таксономия + сегменты)                │
│    - _tools/agents/* (5 агентов)                            │
│  Calls (sub-tools):                                         │
│    - python osl_*.py (7 OSL-скриптов)                      │
│    - python conformal_prediction.py                         │
│    - python find_analogs.py (RAG)                          │
└────────────────────────┬───────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────────┐
│              KNOWLEDGE BASE (markdown + Python)             │
│                                                             │
│  ├── _Анализы/        (history of analyses)                │
│  ├── _Справочники/    (taxonomy + segments)                │
│  ├── _tools/                                                │
│  │   ├── osl_*.py          (7 industry models)             │
│  │   ├── conformal_*.py    (prediction intervals)          │
│  │   ├── osl_calibrator.py (auto-calibration)              │
│  │   ├── calibration/*.json (calibrated params)            │
│  │   └── agents/                                            │
│  │       ├── *.md            (5 prompts)                   │
│  │       └── rag/                                           │
│  │           ├── *.py                                       │
│  │           └── radar_rag.db (SQLite + sqlite-vec)        │
│  └── docs/             (this documentation)                │
└────────────────────────┬───────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────────┐
│                    OUTPUT (markdown)                        │
│  _Анализы/<DATE> — <название>.md                           │
└────────────────────────────────────────────────────────────┘
```

---

## 2. Структура папок

```
Макро-радар/
├── Макро-радар — Хаб.md                  (entry point)
├── Архитектура — План v1.md              (design doc)
├── Промпт — Анализ новости.md            (user-facing prompt)
├── CHANGELOG.md                           (version history)
├── Свежие модели — 12 кандидатов v1.md   (research)
├── docs/                                  (4 documentation files)
├── _Справочники/
│   ├── Таксономия шоков.md
│   └── Сегменты клиентов.md
├── _Анализы/                              (output of analyses)
│   ├── 2026-03-27 — Иран ...
│   └── ...
└── _tools/
    ├── osl_common.py            (общие структуры, дедуп 7 OSL-модулей)
    ├── osl_metallurgy.py
    ├── osl_oilgas.py
    ├── osl_chemistry.py
    ├── osl_pharma.py
    ├── osl_retail.py
    ├── osl_energy.py
    ├── osl_oiv.py
    ├── osl_calibrator.py
    ├── conformal_prediction.py
    ├── fetch_macro_state.py     (4 живых макрофида: USD/RUB, Brent, КС ЦБ, инфляция)
    ├── batch_run.py             (пакетный прогон новостей)
    ├── spillover.py             (L2 межотраслевой spillover)
    ├── segment_impact.py        (L3 воздействие на сегменты)
    ├── run_pipeline.py          (end-to-end)
    ├── pyproject.toml           (v0.9 — зависимости + ruff/black)
    ├── README.md
    ├── CALIBRATION_GUIDE.md
    ├── data/
    │   ├── shock_to_industries.json  (маршрутизация 27 подкатегорий)
    │   ├── brent_scenarios.json
    │   ├── spillover_matrix.json
    │   └── segment_impact_table.json
    ├── tests/                   (127 pytest зелёных, 1 skipped)
    ├── calibration/
    │   ├── osl_metallurgy_calibrated.json
    │   └── ... (7 JSON + multi-param/multi-period)
    └── agents/
        ├── README.md
        ├── orchestrator.md
        ├── agent_1_classifier.md
        ├── agent_2_context_rag.md
        ├── agent_3_backtest_analog.md
        ├── agent_4_impact.md
        ├── agent_5_summary.md
        ├── rag_template.md
        └── rag/
            ├── __init__.py
            ├── init_db.py
            ├── embeddings.py
            ├── index_news.py
            ├── find_analogs.py
            └── radar_rag.db
```

---

## 3. Компоненты и их ответственность

### 3.1. OSL-модули (7 штук)

**Назначение:** прогноз выручки эмитентов по физическим объёмам и ценам.

**Контракт каждого модуля (де-факто):**
```python
# Глобальные
PROFILES: dict[str, CompanyProfile]
PRICES_*: dict | dataclass         # цены сырья
ACTUAL_REVENUE_*: dict              # факт МСФО
FX_*: dataclass                     # курс USD/RUB

# Функция
def predict_revenue(company: str) -> RevenuePredict:
    """Возвращает RevenuePredict с predicted_rub_bn."""

def main():
    """CLI entrypoint для запуска бэк-теста."""
```

### 3.2. Conformal Prediction

**`_tools/conformal_prediction.py`:**
- Generic: `make_interval_generic()` — для модулей с `PRICES_*`
- Специальные: `make_interval_retail`, `make_interval_pharma`, `make_interval_energy`, `make_interval_oiv` — для специфичных структур
- Auto-applies calibrations при импорте

### 3.3. Auto-Calibrator

**`_tools/osl_calibrator.py`:**
- `tune_param()` — grid search + binary refine
- `calibrate_<industry>()` — для каждой отрасли
- `apply_calibration()` — загружает JSON в profile
- `apply_all_calibrations()` — для всех 7 модулей
- `drift_check()` — мониторинг

### 3.4. RAG (Agent 3)

**`_tools/agents/rag/`:**
- `init_db.py` — SQLite + sqlite-vec, dim=384
- `embeddings.py` — TfidfEmbedder + SentenceTransformerEmbedder
- `index_news.py` — парсит markdown анализы и индексирует
- `find_analogs.py` — cosine similarity поиск top-K

**Database schema:**
```sql
CREATE TABLE news_analyses (
    id INTEGER PRIMARY KEY,
    file_path TEXT UNIQUE,
    date TEXT,
    title TEXT,
    main_category TEXT,
    subcategory TEXT,
    severity_score INTEGER,
    severity_level TEXT,
    macro_region TEXT,
    micro_region TEXT,
    industries TEXT,
    shock_summary TEXT,
    actual_outcome_summary TEXT,
    created_at TIMESTAMP
);

CREATE VIRTUAL TABLE news_embeddings USING vec0(
    news_id INTEGER PRIMARY KEY,
    title_embedding FLOAT[384],
    what_embedding FLOAT[384]
);
```

### 3.5. Multi-Agent Pipeline

**5 агентов как markdown-файлы** в `_tools/agents/`:
- Каждый = независимый промпт с контрактом input/output JSON
- Управление через `orchestrator.md` (ручной запуск через Claude Code)
- В v2.0 — Python orchestrator через Anthropic SDK

### 3.6. Общие модули и данные (v0.8–v0.9)

- **`osl_common.py`** — общие dataclass-структуры и хелперы; устраняет дублирование между 7 OSL-модулями.
- **`fetch_macro_state.py`** — 4 живых макрофида: USD/RUB, Brent (Yahoo), КС ЦБ (CBR SOAP), инфляция (World Bank). Пишет `data/macro_state.json`.
- **`batch_run.py`** — пакетный прогон набора новостей через пайплайн.
- **`data/shock_to_industries.json`** — маршрутизация 27 подкатегорий шоков на отрасли (вход для L2).
- **`data/brent_scenarios.json`** — сценарные траектории Brent.

---

## 4. Потоки данных

### 4.1. Поток анализа новости (Multi-Agent v1.0)

```
User
  └─ копирует Промпт — Анализ новости.md в Claude
     └─ Agent 1 (Classifier)
        └─ JSON: {classified_event}
           └─ Agent 2 (Context-RAG)
              ├─ читает: отраслевые отчёты, Хаб
              └─ JSON: {classified_event + context}
                 └─ Agent 3 (Backtest-Analog)
                    ├─ вызов: find_analogs.py "WHAT"
                    └─ JSON: {... + historical_analogs}
                       └─ Agent 4 (Impact)
                          ├─ вызов: osl_*.py для затронутых отраслей
                          ├─ вызов: conformal_prediction.py
                          └─ JSON: {... + impact_analysis}
                             └─ Agent 5 (Summary)
                                └─ Markdown → _Анализы/<DATE>.md
```

### 4.2. Поток калибровки

```
Quarterly trigger (manual)
  └─ Update ACTUAL_REVENUE_* in osl_*.py (paste new MSFO)
     └─ python osl_calibrator.py --module all
        ├─ для каждого эмитента: tune_param() через grid search
        └─ save: calibration/osl_*_calibrated.json
           └─ next time conformal_prediction.py imports:
              └─ apply_all_calibrations() reads JSON
                 └─ profile.X = calibrated_value (in-memory)
```

### 4.3. Поток индексации RAG

```
New analysis saved (manual: _Анализы/<DATE>.md)
  └─ python index_news.py
     ├─ parse markdown frontmatter
     ├─ extract title + WHAT + full_text
     ├─ embedder.encode() через multilingual-e5-small
     └─ INSERT INTO news_analyses + news_embeddings
```

---

## 5. API и контракты

### 5.1. RevenuePredict (OSL)

```python
@dataclass
class RevenuePredict:
    company: str
    period: str                          # "12M2025"
    predicted_usd_bn: float
    predicted_rub_bn: float
    breakdown_rub_bn: dict[str, float]   # сегменты revenue
    actual_rub_bn: float | None
    mae_pct: float | None
```

### 5.2. PredictionInterval (Conformal)

```python
@dataclass
class PredictionInterval:
    company: str
    predicted_base: float
    predicted_low: float        # 5% quantile
    predicted_high: float       # 95% quantile
    interval_width_pct: float
    actual: float | None
    actual_in_interval: bool | None
    coverage_metric: str | None  # "BELOW" / "INSIDE" / "ABOVE"
```

> ⚠️ **Покрытие — in-sample, не out-of-sample.** Conformal калибруется на тех же захардкоженных `ACTUAL`, поэтому метрика покрытия отражает остроту интервалов, а не обобщающую способность. Честный out-of-sample (независимые 9M actuals) — в планах.

### 5.3. Frontmatter анализа (YAML)

```yaml
tags: [макро-радар, анализ, <category>]
дата_новости: "YYYY-MM-DD"
дата_анализа: "YYYY-MM-DD"
источник_новости: "URL"
шок_категория: "X.Y Подкатегория"
сила_шока: "L|M|H · NN/100"
multi_agent: bool
```

### 5.4. Calibration JSON

```json
{
  "<company>": {
    "<param_name>": float,
    "mae_pct": float,
    "actual_rub_bn": float
  }
}
```

---

## 6. Критические инварианты

> Список инвариантов которые ВСЕГДА должны соблюдаться. Их нарушение = баг.

1. **Только публичные/иллюстративные данные**
   - В репозитории нет клиентских/портфельных данных банка
   - L3-сегменты не калиброваны на реальных PD банка (`confidence='low'`)

2. **Marp-стили презентаций**
   - `theme: default` (не uncover)
   - `letter-spacing: 0` для русских шрифтов
   - `--html --allow-local-files` обязательно

3. **R1-R8 правила (для отраслевых отчётов)**
   - R1: нет рекомендаций по ценообразованию
   - R3: нет многопараметрических сценарных таблиц
   - R6: snapshot-diff между прогонами
   - R7: нет анализа дивидендов
   - R8: нет хронологий санкций

4. **Frontmatter для всех markdown в `_Анализы/`** — обязательно YAML с tags, датой, источником

5. **Auto-calibration не должна быть destructive** — старые JSON версионируются, можно откатиться

6. **Conformal interval должен покрывать actual в ≥80% случаев** — если ниже, либо base модель некалибрована, либо perturbation узкий. Метрика in-sample (калибровка на тех же ACTUAL).

---

## 7. Тестирование (текущее + roadmap)

### Сейчас
- ✅ **127 pytest зелёных (1 skipped)** — `cd _tools && python -m pytest tests/ -v`
- ✅ **CI: GitHub Actions** (`.github/workflows/test.yml`) — pytest + ruff + black на каждый push
- ✅ pytest suite для всех 7 OSL модулей + conformal + RAG + L1/L2/L3
- ✅ Бэк-тест встроен в каждый OSL — `predict_revenue()` сравнивается с `ACTUAL_REVENUE_*`
- ✅ Conformal `--industry all` показывает coverage 90% interval (in-sample, см. ниже)
- ✅ Auto-calibrator проверяет MAE через grid search

> Единственный skip — честный out-of-sample skip: нет независимых 9M-фактов для проверки обобщения conformal.

### Roadmap
- 📋 Integration test Multi-Agent (на synthetic новости)
- 📋 Out-of-sample conformal на независимых 9M actuals

---

## 8. Deployment

### Текущее (single-user, local)

```bash
# 1. Установить зависимости (есть pyproject.toml, v0.9)
pip install -e .              # numpy/scipy/sklearn/pyyaml; опц. extras + dev (pytest/ruff/black)
# либо вручную:
pip install pyyaml numpy scipy scikit-learn sqlite-vec sentence-transformers
npm install -g @marp-team/marp-cli

# 2. Запуск (любая команда)
cd "<path>/Макро-радар/_tools"
python osl_calibrator.py --module all
python conformal_prediction.py --industry all
python agents/rag/find_analogs.py "<query>"
```

### v2.0 (планируемый — multi-user)

- Docker container с предустановленными зависимостями
- API-обёртка через FastAPI
- Web UI для аналитика (Streamlit?) — упрощает запуск multi-agent
- Postgres вместо SQLite если нужен concurrent access
- Auth через OAuth для команды

---

## 9. Безопасность и приватность

### Текущее
- ✅ Только публичные/иллюстративные данные — без клиентских списков и портфельных данных банка
- ✅ Локальный запуск, нет внешних API кроме Claude (через Claude Code)

### Roadmap
- 📋 PII-санитизация если будет интеграция с внутренними данными банка (Слой 3)
- 📋 Differential privacy при работе с агрегатами портфеля
- 📋 Encrypted-at-rest для radar_rag.db
- 📋 Audit log калибровок (who/when/what changed)

---

## 10. Метрики и мониторинг

| Метрика | Цель | Текущее v0.9 |
|---|---|---|
| OSL MAE по 28 эмитентам | ≤ 5% (среднее) | ~3% (после auto-calibration) |
| Conformal coverage (90% interval) | ≥ 90% | 82% (23/28) |
| Время Multi-Agent Pipeline | ≤ 5 мин | 3-5 мин |
| RAG поиск top-5 | ≤ 1 сек | ~200 мс |
| Auto-calibration всех 7 модулей | ≤ 1 мин | ~30 сек |

**Drift monitor** (`python osl_calibrator.py --module drift`):
- Triggered weekly
- Flag NEEDS_RECALIBRATION при drift >5 п.п.

---

## 11. Команды разработчика

```bash
# Полный цикл
cd Макро-радар/_tools

# 1. Калибровка (после новых ACTUAL_REVENUE)
python osl_calibrator.py --module all

# 2. Бэк-тест
python conformal_prediction.py --industry all

# 3. Drift check
python osl_calibrator.py --module drift

# 4. RAG re-indexing (после новых анализов)
cd agents/rag
python index_news.py

# 5. Бэк-тест отдельных модулей
python osl_metallurgy.py
python osl_oilgas.py --company ЛУКОЙЛ
```

---

## 12. Roadmap (приоритизированный)

Статусы на v0.9.2 (июнь 2026):

| Версия | Содержание | Статус |
|---|---|---|
| v0.7–0.9 | energy refactor + multi-period validation | ✅ done |
| v0.7–0.9 | pytest suite (127 зелёных) + CI/линтеры (ruff/black) | ✅ done |
| v0.7–0.9 | `pyproject.toml` (зависимости) | ✅ done |
| v0.8–0.9 | `fetch_macro_state.py` — живые макрофиды | ✅ done |
| v0.8–0.9 | маршрутизация подкатегорий (`shock_to_industries.json`) | ✅ done |
| v0.8–0.9 | multi-source / credit-channel spillover (L2) | ✅ done |
| — | API connectors LME/LBMA/Минэнерго | 📋 план |
| — | OSLModel унифицированный интерфейс | 📋 план |
| — | Python orchestrator для Multi-Agent | 📋 план |
| — | out-of-sample conformal (независимые 9M actuals) | 📋 план |
| — | L3-калибровка на данных банка; Diebold-Yilmaz / DebtRank | 📋 план |
| v2.0 | Web UI + multi-user + Postgres | 📋 план |

---

## 13. Контакты и ответственные

- **Owner:** Кредитный департамент Т-Банка
- **Tech lead:** см. CHANGELOG.md
- **Bug reports:** через Issues или прямой канал
- **Pull requests:** в личной ветке + ревью перед merge

---

*Техническая документация · v0.9.2 · 2026-06-15*
