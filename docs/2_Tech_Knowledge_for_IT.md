---
tags: [макро-радар, docs, knowledge, tech, для-IT]
дата: "2026-06-15"
аудитория: "IT-специалист готовый разрабатывать"
---

# 📗 Технический стек и решения — для IT-специалиста

> [[../Макро-радар — Хаб|← Хаб]] · [[1_Macro_Knowledge_for_IT|← Макро-знания]] · [[3_User_Guide_Analyst|→ User Guide]] · [[4_Tech_Architecture|→ Тех. документация]]

Обзор технологий, паттернов и подходов, применённых в проекте «Макро-радар». Документ для разработчика, который собирается работать с кодом или расширять систему.

---

## 1. Стек

| Слой | Технология | Версия | Зачем |
|---|---|---|---|
| **Язык** | Python | 3.10+ | Расчёты OSL, Conformal, Calibrator |
| **Эмбеддинги** | sentence-transformers | 5.4.1 | Семантический поиск аналогов |
| **Модель** | multilingual-e5-small | — | 384d, поддерживает русский |
| **Вектор-БД** | sqlite-vec | 0.1.9 | Быстрый cosine similarity на SQLite |
| **TF-IDF fallback** | scikit-learn | — | Когда модели недоступны |
| **Frontmatter** | PyYAML | 6.0.3 | Парсинг markdown с meta |
| **Документы** | Obsidian + Markdown | — | Хранилище анализов |
| **Презентации** | Marp CLI | — | Markdown → PDF слайды |
| **Графики** | matplotlib | — | PNG для презентаций |
| **Web** | claude_code WebSearch | — | Поиск свежих новостей |
| **LLM** | Claude (Anthropic) | claude-opus-4-7 | Multi-Agent Pipeline |

**Зависимости легковесные** — нет тяжёлых фреймворков (Django, Pandas обязательно нет, NumPy опционально).

---

## 2. Архитектурные паттерны

### 2.1. Module-per-domain (один файл на отрасль)

```
_tools/
├── osl_metallurgy.py      # один файл = одна отрасль
├── osl_oilgas.py
├── osl_chemistry.py
├── osl_pharma.py
├── osl_retail.py
├── osl_energy.py
└── osl_oiv.py
```

**Почему так:** OSL для каждой отрасли имеет уникальную логику (хочет разные параметры). Унифицированный API через базовый класс был бы overengineered на текущем этапе. **При v0.7+** — рефакторинг в `OSLModel` интерфейс уместен.

### 2.2. Calibration — JSON-конфиг отдельно от кода

```
calibration/
├── osl_metallurgy_calibrated.json
├── osl_oilgas_calibrated.json
└── ...
```

`osl_calibrator.py` подбирает параметры через grid search → сохраняет в JSON → `apply_all_calibrations()` применяет при импорте Conformal.

**Преимущество:** код стабилен, калибровки версионируются отдельно. **Раз в квартал** одна команда обновляет все 7 модулей.

### 2.3. Multi-Agent Pipeline — markdown-based prompts

5 агентов как **отдельные markdown-файлы** в `_tools/agents/`:
- `agent_1_classifier.md` — классификация события
- `agent_2_context_rag.md` — RAG-контекст
- `agent_3_backtest_analog.md` — поиск исторических аналогов
- `agent_4_impact.md` — применение карты L1/L2/L3
- `agent_5_summary.md` — финальный документ

**Почему markdown:** промпты — это текст, а не код. Версионирование, ревью, изменения через git diff — всё работает естественно.


## 3. Ключевые алгоритмы

### 3.1. OSL предсказание выручки

```python
# Простейшая версия
def predict_revenue(company):
    profile = PROFILES[company]
    production = PRODUCTION[company]
    revenue = sum(p.volume * PRICES[p.metal] for p in production)
    revenue *= FX_USD_RUB
    revenue *= (1 + profile.other_income_pct)
    return revenue
```

Для гибридных моделей (например, Северсталь):
```python
# export по FOB + domestic с премией
revenue = (q × export_share × FOB_price × FX
         + q × (1-export_share) × domestic_price_RUB)
```

### 3.2. Conformal Prediction (perturbation-based)

```python
def conformal_predict(predictor_fn, base_params, n=200, conf=0.90):
    base = predictor_fn(base_params)
    predictions = []
    for _ in range(n):
        perturbed = perturb(base_params)  # ±5% prices, ±3% volumes
        predictions.append(predictor_fn(perturbed))
    alpha = 1 - conf
    return {
        "base": base,
        "low":  np.quantile(predictions, alpha/2),
        "high": np.quantile(predictions, 1-alpha/2),
    }
```

> ⚠️ **Покрытие — in-sample, не out-of-sample.** Калибровка идёт на тех же захардкоженных `ACTUAL`, поэтому «покрытие» — это мера остроты интервалов, а не обобщающей способности. Честный out-of-sample требует независимых 9M-фактов (в планах).

### 3.3. Auto-calibration (grid search + binary refine)

```python
def tune_param(predict_fn, target, p_min, p_max, n_steps=30):
    best_param, best_err = None, inf
    # Grid search
    for i in range(n_steps + 1):
        p = p_min + i * (p_max - p_min) / n_steps
        err = abs(predict_fn(p) - target) / target
        if err < best_err:
            best_param, best_err = p, err
    # Binary refine ±3%
    for _ in range(8):
        for c in [best_param * 0.97, best_param * 1.03]:
            err = abs(predict_fn(c) - target) / target
            if err < best_err:
                best_param, best_err = c, err
    return best_param, best_err
```

Это **не настоящая оптимизация** (нет градиентов), но для одно-параметрической калибровки — достаточно. Для multi-param tune (v0.7) → `scipy.optimize.differential_evolution`.

### 3.4. RAG поиск аналогов

```python
def find_analogs(query_text, top_k=5):
    query_emb = embedder.encode(query_text)
    # SQLite + sqlite-vec
    results = conn.execute("""
        SELECT n.*, vec_distance_cosine(e.what_embedding, ?) as dist
        FROM news_analyses n
        JOIN news_embeddings e ON n.id = e.news_id
        ORDER BY dist ASC LIMIT ?
    """, (query_emb.tobytes(), top_k)).fetchall()
    return results
```

Embeddings генерируются через **multilingual-e5-small** (384d). При отсутствии sentence-transformers — fallback на **TF-IDF + TruncatedSVD** в 384 dim.

---

## 4. Структуры данных

### 4.1. Frontmatter анализа

```yaml
---
tags: [макро-радар, анализ, 4.7]
дата_новости: "2026-04-23"
дата_анализа: "2026-04-25"
источник_новости: "URL"
шок_категория: "4.7 Валютная интервенция"
сила_шока: "M · 45/100"
multi_agent: true
---
```

PyYAML парсит → передаётся в RAG индексатор и Conformal.

### 4.2. JSON калибровки

```json
{
  "Норникель": {
    "other_income_pct": 0.140,
    "mae_pct": 0.0,
    "actual_rub_bn": 1225
  },
  "Северсталь": { ... }
}
```

### 4.3. Pipeline JSON между агентами

```json
{
  "WHAT": "...",
  "main_category": "1.1",
  "severity_score": 90,
  "context": { "relevant_industries": [...] },
  "historical_analogs": [{ "title": "...", "similarity": 0.907 }],
  "forecast_from_analog": { "expected_trajectory": "..." },
  "impact_analysis": {
    "L1_macro": [...],
    "L1_5_osl": [...],
    "L2_industry": [...],
    "L3_segments": [...]
  }
}
```

---

## 5. Производительность

| Операция | Время |
|---|---|
| 1 запуск OSL для отрасли (5 эмитентов) | ~50 мс |
| 1 Conformal prediction (n=200 simulations) | ~5 сек |
| Auto-calibration всех 7 модулей | ~30 сек |
| RAG поиск top-5 аналогов | ~20 мс (TF-IDF) / ~200 мс (ST) |
| Multi-Agent Pipeline (5 агентов) | ~3-5 минут |
| Marp PDF (15 слайдов) | ~10 сек |

Не нужны GPU, оптимизации. Всё умещается в 1 CPU-минуту, кроме Multi-Agent (там Claude API вызовы).

---

## 6. Тестирование (текущее состояние)

**Юнит/интеграционные тесты:** 127 pytest зелёных (1 skipped), запуск `cd _tools && python -m pytest tests/ -v`; CI через GitHub Actions.

> Единственный skip — честный out-of-sample skip (нет независимых 9M-фактов для проверки обобщения conformal).

**Quality assurance через бэк-тест:**
- Каждый OSL-скрипт имеет `ACTUAL_REVENUE_*_2025` → встроенная проверка predicted vs actual
- Conformal `--industry all` показывает покрытие 90% interval — это и есть quality gate
- Auto-calibrator проверяет MAE через grid search — само-проверка

**CI/линтеры (v0.9):**
- GitHub Actions (`.github/workflows/test.yml`) — pytest + ruff + black на каждый push

---

## 7. Принципы дизайна

### 7.1. Plain Python вместо фреймворков

Pandas / Django / FastAPI / SQLAlchemy — **не используются**. Только stdlib + numpy + scikit-learn где минимально нужно.

**Почему:** проект — **аналитический инструмент**, не сервис. Простота важнее производительности на масштабе.

### 7.2. Markdown-first

Все знания в markdown:
- Промпты — markdown
- Анализы — markdown
- Документация — markdown
- Калибровки — JSON (но всё рядом в markdown)

**Это даёт:**
- Версионирование через git
- Просмотр в Obsidian
- Конвертация в PDF через Marp

### 7.3. Изолированные модули

Каждая отрасль = один Python-файл. Можно править без риска сломать другие. Поэтому 7 OSL-модулей удобнее одного **OSLModel** базового класса (на текущем этапе).

### 7.4. Bias-detection через Conformal

Если actual выходит за 90% interval **постоянно** (несколько эмитентов) → модель **systematically biased**. Это сигнал для калибровки **до** того как ошибки станут критичными.

---

## 8. Что не сделано (intentional & roadmap)

### Не сделано осознанно
- ❌ FastAPI / REST API — проект для local use, не сервис
- ❌ Database migrations / Alembic — SQLite простой
- ❌ Docker — Python script достаточно
- ❌ Authentication — single-user

### Сделано в v0.8–v0.9
- ✅ pytest suite (127 зелёных, 1 skipped) + CI (GitHub Actions: pytest/ruff/black)
- ✅ `pyproject.toml` (v0.9, с ruff/black)
- ✅ Multi-period validation (energy refactor + multi-period калибровки)
- ✅ `fetch_macro_state.py` — живые макрофиды (USD/RUB, Brent, КС ЦБ, инфляция)
- ✅ `osl_common.py` — общие структуры, дедуп 7 OSL-модулей

### Roadmap v0.9+
- 📋 API connectors LME/LBMA/Минэнерго (auto-update PRICES_*)
- 📋 Python orchestrator для Multi-Agent (вместо ручного запуска 5 промптов)
- 📋 Унифицированный `OSLModel` интерфейс

---

## 9. Файлы зависимостей (нет requirements.txt)

```bash
# Минимально (всегда работает):
pip install pyyaml numpy scikit-learn

# Для RAG:
pip install sqlite-vec sentence-transformers

# Для Marp презентаций:
npm install -g @marp-team/marp-cli

# Для Multi-Agent:
# Claude Code (anthropic)
```

**`pyproject.toml` уже есть (v0.9):** numpy / scipy / scikit-learn / pyyaml + опц. `sentence-transformers` / `feedparser`; dev-зависимости — `pytest` / `ruff` / `black`.

---

## 10. Дальнейшее изучение

| Тема | Ссылка |
|---|---|
| Conformal Prediction теория | https://en.wikipedia.org/wiki/Conformal_prediction |
| sentence-transformers tutorial | https://www.sbert.net/ |
| Marp CLI docs | https://marp.app/ |
| sqlite-vec | https://github.com/asg017/sqlite-vec |
| Anthropic API | https://docs.anthropic.com/ |

---

*Технические знания · v0.9.2 · 2026-06-15*
