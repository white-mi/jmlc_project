---
tags: [russian-propagation, инструменты]
дата: "2026-06-15"
версия: "0.9"
---

# `_tools/` — Инструменты Russian Propagation

> **← Хаб**

Python-пакет для расчёта индикаторов и моделей всех 4 слоёв архитектуры. Запуск из командной строки или импорт как модулей.

**Состояние v0.9.2 (июнь 2026):**
- `pyproject` version = **0.9.0**, `pipeline_version='0.9'`.
- **349 pytest зелёных, 0 skipped** (включая DS-тесты: ОИВ region×year панель, нефтегаз, химия, EDA/DS-синтез).
- OSL покрывает **7 отраслей тирами по доступности данных**: **5 валидированы** (walk-forward +
  conformal + DS-отчёт), **2 иллюстративны** (нет публичного Q×P) — см. [`COVERAGE_TIERS`](../docs/COVERAGE_TIERS.md).
- `fetch_macro_state.py` тянет **4 живых макрофида**.

**DS-слой:** реальная панель FY2021–2025 +
сравнение 3 моделей + честная **out-of-sample** walk-forward + **split-conformal**. Глубоко
проработаны **пять отраслей**: металлургия ([отчёт](../docs/DS_REPORT.md), N=24), нефтегаз
([отчёт](../docs/DS_REPORT_OILGAS.md), N=18), химия ([отчёт](../docs/DS_REPORT_CHEMISTRY.md),
N=18; структурная подключена), энергетика ([отчёт](../docs/DS_REPORT_ENERGY.md), N=30;
двухкомпонентная структурная) и ОИВ ([отчёт](../docs/DS_REPORT_OIV.md),
N=24; фискальная панель region×year). DS-харнесс **industry-параметрический**
(один и тот же код на все 7 отраслей). Модули:

| Файл | Назначение |
|---|---|
| `data/panel/` | Панель эмитент×период (CSV) + схема + `SOURCES.md` (все цифры с цитатами) |
| `osl_panel.py` | stdlib-загрузчик панели (джойн цен, `to_matrix`) |
| `eda_osl.py` + `notebooks/eda_osl.ipynb` | EDA: 8 фигур + импликации (extra `[eda]`) |
| `osl_models.py` | 3 модели: StructuralOSL / ElasticNet-Ridge / HistGBM (единый fit/predict) |
| `osl_walkforward.py` | Expanding-window walk-forward + MAE/MAPE/RMSE/skill/Diebold–Mariano |
| `conformal_split.py` | Split/inductive conformal → OOS-покрытие (vs in-sample perturbation) |
| `backtest_analyses.py` | Продуктовый слой: воспроизводимая сводка корпуса `_Анализы/` (proxy-feedback), stdlib, read-only |

---

## Перечень модулей по слоям

### L0 — Фильтр новостей и оркестрация

| Файл | Назначение |
|---|---|
| `agents/orchestrator.py` | Оркестратор multi-agent пайплайна |
| `agents/rag/` | RAG-подсистема: индексация новостей, эмбеддинги, поиск аналогов (`index_news.py`, `embeddings.py`, `find_analogs.py`, `init_db.py`) |
| `batch_run.py` | Пакетный прогон нескольких новостей за один запуск |

### L1 — Макро-состояние

| Файл | Назначение |
|---|---|
| `calc_rf_cai.py` | Расчёт РФ-CAI (composite activity index) |
| `calc_rf_epu.py` | Расчёт индекса EPU; поддержка `--source fred` (FRED fallback), end_date-якорь, флаг `epu_degraded` |
| `fetch_macro_state.py` | Авто-обновление макро-состояния — 4 живых фида (см. ниже). Перезаписывает только `current_state` в `data/macro_state.json` |

### L1.5 — Операционный сигнал (OSL)

| Файл | Назначение |
|---|---|
| `osl_common.py` | Общие структуры: `RevenuePredict`, `FXRate`, `mae_pct` — единая схема для всех 7 OSL-модулей |
| `osl_metallurgy.py` | OSL для металлургии |
| `osl_oilgas.py` | OSL для нефтегаза |
| `osl_chemistry.py` | OSL для химии |
| `osl_energy.py` | OSL для энергетики |
| `osl_pharma.py` | OSL для фармацевтики |
| `osl_retail.py` | OSL для розницы непродовольственной |
| `osl_oiv.py` | OSL для ОИВ (региональные органы власти) |
| `conformal_prediction.py` | Inductive Conformal Prediction — интервалы доверия для OSL-прогнозов |
| `osl_calibrator.py` | Авто-калибровка OSL-моделей (см. `CALIBRATION_GUIDE.md`) |

### L2 — Отраслевой spillover

| Файл | Назначение |
|---|---|
| `spillover.py` | Межотраслевой spillover: `magnitude` из severity + `propagate_multi_source` (агрегация источников) + `propagate_credit_channel` (broad credit channel шоков ЦБ) |

### L3 — Поведение клиентов / сегменты

| Файл | Назначение |
|---|---|
| `segment_impact.py` | Воздействие на 10 сегментов клиентов; `REGION_PROFILES`; `confidence` как поле данных (`confidence_default`) |

### Оркестрация end-to-end

| Файл | Назначение |
|---|---|
| `run_pipeline.py` | Полный прогон L0→L3 за один вызов |

### Данные (`data/`)

| Файл | Назначение |
|---|---|
| `macro_state.json` | Текущее и базовое макро-состояние (`current_state` обновляет `fetch_macro_state.py`) |
| `shock_to_industries.json` | Маршрутизация шоков — все 27 подкатегорий → отрасли |
| `brent_scenarios.json` | Сценарии по Brent |
| `spillover_matrix.json` | Матрица зависимостей отраслей (Fialkowski 7×7) |
| `segment_impact_table.json` | Lookup-таблица коэффициентов воздействия на сегменты |

### Инфраструктура

| Файл | Назначение |
|---|---|
| `.github/workflows/test.yml` | CI: pytest + ruff + black (TF-IDF режим без сети) |
| `pyproject.toml` | Зависимости + конфигурация ruff/black/pytest |
| `tests/` | 349 тестов (0 skipped): юнит + property-based (hypothesis) + golden-снапшот пайплайна + голден-фикстуры LLM-ответов + регрессионные гейты качества ретрива |
| `eval_all.py` | сводная витрина метрик: значение · порог · чем воспроизвести → [docs/EVAL.md](../docs/EVAL.md) |
| `eval_rag.py` / `eval_l0_classifier.py` | retrieval-eval (два gold-set × два эмбеддера) / eval классификатора L0 |

---

## Как запускать

### OSL по отрасли

```bash
cd _tools
python osl_metallurgy.py              # все эмитенты отрасли
python osl_metallurgy.py --company Полюс
python osl_oilgas.py
python osl_oilgas.py --company Газпром
```

### Макро-состояние и индексы

```bash
python fetch_macro_state.py           # авто-обновление current_state (4 фида)
python calc_rf_cai.py
python calc_rf_epu.py                  # по корпусу _Анализы/
python calc_rf_epu.py --source fred    # FRED fallback (global uncertainty proxy)
```

### Conformal-интервалы и калибровка

```bash
python conformal_prediction.py --industry all
python osl_calibrator.py --module all   # см. CALIBRATION_GUIDE.md
```

### Spillover и сегменты

```bash
python spillover.py
python segment_impact.py
```

### Полный пайплайн и пакетный прогон

```bash
python run_pipeline.py                  # один прогон L0→L3
python batch_run.py                     # пакетный прогон нескольких новостей
```

### Тесты

```bash
cd _tools
python -m pytest tests/ -v              # 349 зелёных, 0 skipped
```

---

## Статус OSL (v0.9.2)

- **7 отраслей в продакшене:** металлургия, нефтегаз, химия, энергетика, фарма, розница, ОИВ.
- **Conformal-интервалы.** Perturbation-интервалы (`conformal_prediction.py`) — **IN-SAMPLE**. Честная **out-of-sample** валидация — `conformal_split.py` (split-conformal на независимой панели FY2021–2025, тест `test_holdout_coverage_metallurgy`). См. [DS-отчёт](../docs/DS_REPORT.md).
- **Нефтегаз — рабочий слой.** Pipeline v0.9 даёт числа; `fetch_macro_state.py` тянет живые макрофиды.

---

## Источники макро (`fetch_macro_state.py`)

4 из 4 живых фида, с тестируемыми парсерами и graceful degrade (пишет только `current_state`):

| Показатель | Источник |
|---|---|
| **USD/RUB** | cbr-xml-daily |
| **Brent** | Yahoo Finance (`BZ=F`) |
| **KeyRate** (КС ЦБ) | CBR KeyRate SOAP |
| **Inflation** | World Bank API |

---

## Зависимости

Из `pyproject.toml` (`requires-python = ">=3.11"`):

**Runtime:**
- `numpy`, `scipy`, `scikit-learn` — численные расчёты и модели
- `pyyaml` — конфигурация / снапшоты
- `sqlite-vec`, `sentence-transformers` — векторный поиск RAG (L0)
- `anthropic` — LLM-агенты
- `feedparser` — RSS-корпус новостей

**Dev:**
- `pytest`, `pytest-cov` — тесты и покрытие
- `ruff`, `black` — линтинг и форматирование

> Пакет опирается на научный стек и LLM-SDK — не только на stdlib.

---

## Roadmap инструментов

Статус v0.9.x:

### ✅ Реализовано

| Возможность | Чем закрыта |
|---|---|
| Авто-обновление макро | `fetch_macro_state.py` (4 живых фида) |
| FRED fallback для EPU | `calc_rf_epu.py --source fred`; end_date-якорь + `epu_degraded` |
| Маршрутизация шоков | все 27 подкатегорий (`data/shock_to_industries.json`) |
| Multi-source spillover | `propagate_multi_source` + `propagate_credit_channel` |
| Out-of-sample conformal | `conformal_split.py` + панель FY2021–2025 |
| CI на GitHub Actions | `.github/workflows/test.yml`: pytest + ruff + black |
| Конфигурация линтеров | ruff/black в `pyproject.toml` |
| Тестовое покрытие | 349 зелёных (0 skipped) |
| Единая схема OSL | `osl_common.py` — `RevenuePredict`/`FXRate`/`mae_pct` для 7 модулей; `batch_run.py` — пакетный прогон |

### ⏳ Осознанно не закрыто (нет данных)

| Пункт | Почему |
|---|---|
| **L3-калибровка на данных банка** | Нет доступа к внутренним данным портфеля |
| **Diebold-Yilmaz / DebtRank** | Нужно 3+ года истории revenue и balance sheets топ-заёмщиков |

---

## Структура папки

```
_tools/
├── README.md                  ← этот файл
├── pyproject.toml             ← зависимости + ruff/black/pytest
├── CALIBRATION_GUIDE.md       ← регламент калибровки OSL
├── osl_common.py              ← общие структуры OSL
├── osl_metallurgy.py / osl_oilgas.py / osl_chemistry.py
├── osl_energy.py / osl_pharma.py / osl_retail.py / osl_oiv.py
├── conformal_prediction.py    ← интервалы доверия
├── osl_calibrator.py          ← авто-калибровка
├── calc_rf_cai.py / calc_rf_epu.py
├── fetch_macro_state.py       ← авто-обновление макро
├── spillover.py / segment_impact.py
├── run_pipeline.py / batch_run.py
├── agents/                    ← orchestrator.py + rag/
├── data/                      ← macro_state, shock_to_industries, brent_scenarios, ...
├── calibration/               ← <module>_calibrated.json (7 шт.)
└── tests/                     ← 349 тестов (0 skipped)
```

---

*Документация v0.9 · 2026-06-15 · Обновляется с новыми скриптами*
