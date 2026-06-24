---
tags: [макро-радар, roadmap, v0.8]
дата: "2026-04-25"
статус: "draft"
---

# ROADMAP v0.8 — следующий цикл доработок Макро-радара

> [[../Макро-радар — Хаб|← Хаб]] · [[../CHANGELOG|← CHANGELOG v0.7]]

> [!done] СТАТУС на v0.9.2 (июнь 2026): большинство пунктов этого плана ВЫПОЛНЕНО — см. CHANGELOG v0.9.0. Документ сохранён как исторический план.
> Выполнено: **A1** (`fetch_macro_state.py`), **A3** (FRED fallback), **B1** (маршрутизация подкатегорий), **B2** (multi-source spillover), **D1** (CI), **D2** (ruff/black). pytest расширен до **127 зелёных (1 skipped)**.

## Контекст

v0.7 закрыл MVP — все 4 слоя дают цифры за один прогон. На прогоне свежей новости (ЦБ снизил КС до 14.5%, 24.04.2026) выявлены 4 системных лимита, требующих следующего цикла. v0.7.1 закрыл 3 из 4 (direction-aware L3, broad credit channel в категории 4, sentiment-aware EPU); v0.8 закрывает остальные.

---

## Группа A — Автономия данных

### A1. ✅ Auto-pull `current_state` из ЦБ API — ВЫПОЛНЕНО (v0.9)

**Проблема:** `data/macro_state.json` обновляется вручную. На свежей новости (КС=14.5%) обнаружили устаревшее значение КС=21% — потребовался ручной апдейт.

**Файл:** новый `_tools/fetch_macro_state.py`
**Что:**
- Парсер ЦБ РФ XML/JSON: КС, USD/RUB, M2, инфляция (через статистический бюллетень)
- Парсер для Brent: один из доступных public API (например, EIA или Bloomberg ticker)
- Парсер Росстат IPP: HTML-парсинг последнего релиза
- Скрипт перезаписывает `data/macro_state.json` (только `current_state` блок, baseline не трогается)
- Cron-задание раз в неделю; manual `python fetch_macro_state.py`

**Сложность:** middle (~12-16 ч). API ЦБ хорошо документирован; Brent и Росстат — HTML-парсинг.

### A2. RSS-парсер для EPU (расширение корпуса)

**Проблема:** EPU lite сейчас считается на корпусе `_Анализы/` (14 файлов). Для статистически значимого индекса нужен больший корпус.

**Файл:** новый `_tools/fetch_news_corpus.py` + расширение `calc_rf_epu.py`
**Что:**
- `feedparser` подключение к 5 RSS: ТАСС, Коммерсантъ, РБК, Ведомости, Forbes
- Корпус накапливается в SQLite (`_tools/data/news_corpus.db`); ежедневный pull
- EPU считается на rolling 30/90/365 дней
- Бэк-тест возможен: пик 03.2022 (санкции), пик 12.2024 (ставочный), нормализация 06.2023

**Сложность:** middle (~10-12 ч). RSS — простой, но надо быть аккуратным с rate limits и encoding.

### A3. ✅ Альтернативный fallback на US-EPU из FRED — ВЫПОЛНЕНО (v0.9)

**Если** A2 затягивается или RSS-источники недоступны: использовать готовый US-EPU index из FRED как proxy.

**Файл:** опция в `calc_rf_epu.py`
**Что:** `--source fred` подгружает https://fred.stlouisfed.org/series/USEPUINDXD (через `pandas_datareader` или прямой HTTP).
**Польза:** мгновенная история 1985-2026; корреляция с РФ-EPU невелика, но улавливает global uncertainty (геополитика, ставочные циклы Fed).

**Сложность:** low (~2-3 ч).

---

## Группа B — Углубление L0/L1.5

### B1. ✅ Подкатегории шока в L3 lookup-таблице — ВЫПОЛНЕНО (v0.9, маршрутизация 27 подкатегорий в `data/shock_to_industries.json`)

**Проблема:** `data/segment_impact_table.json` сейчас хранит коэффициенты только для 5 top-level категорий. Подкатегория 4.1 (повышение КС) и 4.X (снижение КС) — это разные шоки с разными размерами эффекта на разных горизонтах.

**Что:** расширить JSON до 18 подкатегорий × 10 сегментов × 3 КС-режима = 540 ячеек × 4 поля.

**Сложность:** high (~16-20 ч), не из-за технологий, а из-за экспертной работы по калибровке коэффициентов.

**Альтернативный путь (lite):** сделать только подкатегории, где импакт значимо отличается от top-level (например, 1.1 война-старт vs 1.5 тестирование оружия — две разные истории). ~6 ч.

### B2. ✅ Multi-source spillover (агрегация broad credit channel) — ВЫПОЛНЕНО (v0.9, `propagate_multi_source` + `propagate_credit_channel`)

**Проблема:** `propagate_shock(industry, magnitude)` пропагандирует от **одной** отрасли. Но шок ставки ЦБ влияет одновременно на 5+ отраслей через credit channel. Сейчас в pipeline берётся primary=retail и spillover показывает узкий веер.

**Файл:** расширение `spillover.py`
**Что:**
- Новая функция `propagate_multi_source(sources: dict[industry, magnitude])`:
  - Для каждой исходной отрасли запускает `propagate_shock`
  - Агрегирует по target-отрасли через MAX (худший сценарий) или SUM с весами
- Специальный режим `credit_channel` для шоков ЦБ: атакует все 5 broad-credit отраслей одновременно с разной величиной (retail 0.8, oiv 0.6, metallurgy 0.4, chemistry 0.4, oilgas 0.3)

**Сложность:** middle (~6-8 ч).

### B3. Multi-period actuals для остальных 5 отраслей

**Проблема:** v0.7 multi-period валидация работает только для металлургии и нефтегаза (ACTUAL_REVENUE_9M_2025 заполнено для 9 эмитентов). Остальные 5 отраслей — single-period.

**Файлы:** `osl_chemistry.py`, `osl_pharma.py`, `osl_retail.py`, `osl_energy.py`, `osl_oiv.py`
**Что:** добавить `ACTUAL_REVENUE_9M_2025` для всех публичных эмитентов (где есть IR-релизы 9M). Запустить `calibrate_*_multi_period()` для каждой.
**Польза:** убирает overfitting к одному факту 12М 2025 для всех 28 эмитентов.

**Сложность:** middle (~10-12 ч), большая часть — поиск actuals в IR.

---

## Группа C — Расширение горизонта прогноза

### C1. Forward projection — куда пойдёт CAI через 1Q/1H

**Проблема:** CAI сейчас даёт snapshot текущего состояния. Для tactical decisions нужен прогноз — куда CAI движется.

**Файл:** новый `_tools/calc_rf_cai_forecast.py`
**Что:**
- AR(1) модель на CAI каждой компоненты (КС, USD/RUB, ...) с lag=1Q
- Backtested baseline: какой был средний MoM-сдвиг при текущей фазе
- Output: `cai_forecast_1q`, `cai_forecast_2q`, `phase_at_horizon`

**Сложность:** middle (~8-10 ч). Нужны исторические ряды компонент за 2017-2025 — частично есть в `data/macro_state.json` historical_snapshots, нужно расширить.

### C2. Динамика OSL: тренд эмитента (не только snapshot)

**Проблема:** OSL даёт прогноз на 12М 2025. Для оценки кредитного риска нужен тренд: revenue 2024 → 2025 → 2026E.

**Файлы:** все `osl_*.py`
**Что:** добавить `ACTUAL_REVENUE_2024` для каждого эмитента; `predict_revenue_yoy_change(company)` возвращает {YoY 2024→2025, projected YoY 2025→2026 на текущих ценах}.

**Сложность:** middle (~12 ч).

### C3. L4 — Tactical recommendations layer

**Проблема:** В v0.7 анализ заканчивается на L3 (segment impact). Для bank-decision нужно явное «что делать» — `tactical_note` поле для каждого сегмента + общий plan для риск-комитета.

**Файл:** новый `_tools/tactical.py`
**Что:**
- На входе full state pipeline (L0-L3)
- Output: список actionable рекомендаций (`grow_portfolio` / `reduce_exposure` / `monitor_only` / `repricing` / `hedge_fx`)
- Простая rule-based логика: ΔPD < -0.5 → grow; +0.5 < ΔPD < +1.0 → repricing; ΔPD > +1.0 → reduce

**Сложность:** middle (~6-8 ч).

---

## Группа D — Качество и инфраструктура

### D1. ✅ CI/CD через GitHub Actions — ВЫПОЛНЕНО (v0.9, `.github/workflows/test.yml`)

**Файл:** `.github/workflows/test.yml`
**Что:** на каждый push прогон `pytest` + lint (ruff/black). Локальный pytest есть, но нет автомата.
**Сложность:** low (~2-3 ч). Зависит от того, есть ли у проекта git remote.

### D2. ✅ ruff/black форматирование — ВЫПОЛНЕНО (v0.9, секции `[tool.ruff]`/`[tool.black]` в pyproject.toml)

**Что:** добавить `[tool.ruff]` секцию в pyproject.toml; pre-commit hook.
**Сложность:** low (~1-2 ч).

### D3. Type hints + mypy

**Проблема:** Часть кода не типизирована (особенно `osl_*.py` с динамическими `dict[str, Any]`).
**Что:** прогон mypy в strict режиме; постепенное добавление type hints.
**Сложность:** middle (~10 ч).

### D4. Docstring примеры (executable doctests)

**Что:** в каждой публичной функции добавить doctest пример. Запуск через `pytest --doctest-modules`.
**Польза:** живая документация + тестирование примеров.
**Сложность:** low (~4-5 ч).

---

## Группа E — Эксперименты и ML (фаза 1.5+ → v0.9+)

Это уже за пределами v0.8 — оставлены как roadmap для будущих циклов. Включают:

- Foundation TS model (TimesFM/Chronos) для CAI nowcast
- LPCMCI Causal Discovery для L2 (вместо ручной матрицы)
- KAN внутри Mixture-of-Experts для L3 (когда будут реальные данные банка)
- Спутниковые данные (Sentinel-2, VIIRS) для leading-OSL
- Diebold-Yilmaz VAR на отраслевых ОЧР (когда соберём 3+ года истории revenue)
- DebtRank каскад на топ-50 заёмщиков (когда будут balance sheets)

---

## Приоритизация v0.8

**Critical path** (если время ограничено):
1. **A1** auto-pull macro_state (без него pipeline постоянно врёт устаревшими данными — критично)
2. **B2** multi-source spillover (без него L2 неверно для категории 4)
3. **C3** L4 tactical layer (без него аналитик не получает actionable вывод)
4. **D1** CI (защита регрессий)

**Suggested order:**
1. Неделя 1: A1 + A3 fallback (FRED) — закрытие data ingestion
2. Неделя 2: A2 RSS + B3 multi-period для 5 отраслей
3. Неделя 3: B1 подкатегории + B2 multi-source spillover
4. Неделя 4: C1 forecast + C3 tactical + D1 CI

**Итого ≈ 80-100 часов** на v0.8 (аналогично v0.7).

---

## Open вопросы

1. **B1 подкатегии — экспертные коэффициенты.** Кто их валидирует? Сейчас все confidence='low'. Нужен ли отдельный workshop с риск-аналитиком?
2. **A2 RSS — license/политика?** Парсинг RSS ТАСС/Коммерсантъ — формально это open feed, но scale-up может потребовать соглашений.
3. **C2 ACTUAL_REVENUE_2024 — где брать?** МСФО 12М 2024 публиковались год назад; не везде доступны для нашего набора эмитентов. Возможно потребуется частичное покрытие.
4. **D3 mypy — strict сразу или постепенно?** Strict даст ~200+ ошибок; постепенно — больше работы, но без шоковой терапии.

---

## Acceptance criteria для v0.8

- [x] `python fetch_macro_state.py` работает и обновляет JSON автоматически ✅ (v0.9)
- [x] EPU считается с FRED fallback (`--source fred`) + end_date-якорь + `epu_degraded` ✅ (v0.9)
- [x] L3 lookup / маршрутизация поддерживает все подкатегории — 27 подкатегорий в `data/shock_to_industries.json` ✅ (v0.9)
- [x] Spillover для категории 4 пропагандирует на 5 отраслей через `propagate_multi_source` ✅ (v0.9)
- [x] CI на GitHub Actions зелёный ✅ (v0.9)
- [x] pytest зелёный — **127 тестов (1 skipped)** ✅ (v0.9)
- [x] Demo: новость → анализ за <15 минут БЕЗ ручной коррекции (включая обновлённое макро-состояние) ✅ (v0.9)

---

*Roadmap v0.8 · 2026-04-25 · Будет обновляться по мере прогресса*
