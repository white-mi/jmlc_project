---
tags: [макро-радар, инструменты]
дата: "2026-06-15"
версия: "0.9"
---

# `_tools/` — Инструменты Макро-радара

> **← Хаб**

Python-пакет для расчёта индикаторов и моделей всех 4 слоёв архитектуры. Запуск из командной строки или импорт как модулей.

**Состояние v0.9.2 (июнь 2026):**
- `pyproject` version = **0.9.0**, `pipeline_version='0.9'`.
- **254 pytest зелёных, 0 skipped** (+10 DS-нефтегаз; +9 DS-химия; +EDA-параметризация и DS-синтез).
- Архитектурная оценка независимым ревьюером — **9.2/10**.
- OSL покрывает **7 отраслей тирами по доступности данных**: **4 валидированы** (walk-forward +
  conformal + DS-отчёт), **3 иллюстративны** (нет публичного Q×P) — см. [`COVERAGE_TIERS`](../docs/COVERAGE_TIERS.md).
- `fetch_macro_state.py` тянет **4 живых макрофида**.

**DS-слой (доработка для Junior ML Contest, июнь 2026):** реальная панель FY2021–2025 +
сравнение 3 моделей + честная **out-of-sample** walk-forward + **split-conformal**. Глубоко
проработаны **четыре отрасли**: металлургия ([отчёт](../docs/DS_REPORT.md), N=24), нефтегаз
([отчёт](../docs/DS_REPORT_OILGAS.md), N=18), химия ([отчёт](../docs/DS_REPORT_CHEMISTRY.md),
N=18; структурная подключена) и энергетика ([отчёт](../docs/DS_REPORT_ENERGY.md), N=30;
двухкомпонентная структурная + урок про честность). DS-харнесс **industry-параметрический**
(разовая инвестиция на все 7 отраслей). Модули:

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
| `batch_run.py` | **(новый в v0.9)** Пакетный прогон нескольких новостей за один запуск |

### L1 — Макро-состояние

| Файл | Назначение |
|---|---|
| `calc_rf_cai.py` | Расчёт РФ-CAI (composite activity index) |
| `calc_rf_epu.py` | Расчёт индекса EPU; поддержка `--source fred` (FRED fallback), end_date-якорь, флаг `epu_degraded` |
| `fetch_macro_state.py` | **(новый в v0.9)** Авто-обновление макро-состояния — 4 живых фида (см. ниже). Перезаписывает только `current_state` в `data/macro_state.json` |

### L1.5 — Операционный сигнал (OSL)

| Файл | Назначение |
|---|---|
| `osl_common.py` | **(новый в v0.9)** Общие структуры: `RevenuePredict`, `FXRate`, `mae_pct` — устранено дублирование по 7 модулям |
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
| `tests/` | 254 теста (0 skipped) |

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
python -m pytest tests/ -v              # 254 зелёных, 0 skipped
```

---

## Статус OSL (v0.9.2)

- **7 отраслей в продакшене:** металлургия, нефтегаз, химия, энергетика, фарма, розница, ОИВ.
- **Conformal-интервалы работают.** Perturbation-интервалы (`conformal_prediction.py`) помечены как **IN-SAMPLE**. Честная **out-of-sample** валидация РЕАЛИЗОВАНА в DS-доработке: `conformal_split.py` (split-conformal на независимой панели FY2021–2025), тест `test_holdout_coverage_metallurgy` разблокирован. См. [DS-отчёт](../docs/DS_REPORT.md).
- **Нефтегаз больше НЕ «провален».** Pipeline v0.9 работает; `fetch_macro_state.py` тянет живые макрофиды, заглушки заменены рабочей логикой.

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

> Прежнее утверждение «нет зависимостей, только stdlib» больше неактуально — пакет использует научный стек и LLM-SDK.

---

## Roadmap инструментов

Статус относительно [ROADMAP v0.8](ROADMAP_v0.8.md) (см. `CHANGELOG` v0.9.0):

### ✅ Выполнено в v0.9

| Пункт | Что сделано |
|---|---|
| **A1** | `fetch_macro_state.py` — авто-обновление макро (4 живых фида) |
| **A3** | FRED fallback в `calc_rf_epu.py --source fred`; end_date-якорь + `epu_degraded` |
| **B1** | Маршрутизация шоков — все 27 подкатегорий (`data/shock_to_industries.json`) |
| **B2** | Multi-source spillover (`propagate_multi_source` + `propagate_credit_channel`) |
| **D1** | CI на GitHub Actions (`.github/workflows/test.yml`): pytest + ruff + black |
| **D2** | ruff/black в `pyproject.toml` |
| **D (частично)** | Расширение тестов — 254 зелёных (0 skipped) |
| **Рефакторинг** | `osl_common.py` — общие `RevenuePredict`/`FXRate`/`mae_pct` для 7 модулей; `batch_run.py` — пакетный прогон |

### ⏳ Осознанно не закрыто (нет данных)

| Пункт | Почему |
|---|---|
| **Out-of-sample conformal** | ✅ ЗАКРЫТО DS-доработкой — `conformal_split.py` + панель FY2021–2025 (был `skipped`, теперь зелёный) |
| **L3-калибровка на данных банка** | Нет доступа к внутренним данным портфеля |
| **Diebold-Yilmaz / DebtRank** | Нужно 3+ года истории revenue и balance sheets топ-заёмщиков |

---

## Структура папки

```
_tools/
├── README.md                  ← этот файл
├── pyproject.toml             ← зависимости + ruff/black/pytest
├── ROADMAP_v0.8.md            ← исторический план (большинство выполнено в v0.9)
├── CALIBRATION_GUIDE.md       ← регламент калибровки OSL
├── osl_common.py              ← общие структуры OSL (новый)
├── osl_metallurgy.py / osl_oilgas.py / osl_chemistry.py
├── osl_energy.py / osl_pharma.py / osl_retail.py / osl_oiv.py
├── conformal_prediction.py    ← интервалы доверия
├── osl_calibrator.py          ← авто-калибровка
├── calc_rf_cai.py / calc_rf_epu.py
├── fetch_macro_state.py       ← авто-обновление макро (новый)
├── spillover.py / segment_impact.py
├── run_pipeline.py / batch_run.py
├── agents/                    ← orchestrator.py + rag/
├── data/                      ← macro_state, shock_to_industries, brent_scenarios, ...
├── calibration/               ← <module>_calibrated.json (7 шт.)
└── tests/                     ← 254 теста (0 skipped)
```

---

*Документация v0.9 · 2026-06-15 · Обновляется с новыми скриптами*
