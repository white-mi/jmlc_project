# Источники панели данных металлургии

> Граница изоляции: **только публичные данные** — МСФО/IR-отчёты эмитентов + биржевые
> цены (LME/LBMA/World Bank) + FX. Никаких клиентских/портфельных данных банка.
> Каждая цифра сопровождается источником + `confidence`. Спорные/`low`-точки требуют
> ручной сверки пользователем (fact-check pass).

Панель: **5 эмитентов × FY2021–2025** (Полюс — 2021–2024). Гранулярность — годовая (FY).

---

## 1. Выручка (panel_revenue.csv) — таргет

### Полюс (USD, золото) — ✅ high (первичные релизы Polyus)
| Период | $млрд | Источник |
|---|---|---|
| 2021FY | 4.966 | [Polyus FY2022 release](https://polyus.com/en/media/press-releases/financial-results-for-the-second-half-of-2022-and-full-year-2022/) (аудир. сравнение) |
| 2022FY | 4.257 | [Polyus FY2022 release](https://polyus.com/en/media/press-releases/financial-results-for-the-second-half-of-2022-and-full-year-2022/): «Revenue $4,257m, a 14% decrease YoY» |
| 2023FY | 5.237 | [Polyus FY2024 release](https://polyus.com/en/media/press-releases/financial-results-for-the-second-half-of-2024-and-full-year-2024/) (аудир. сравнение) |
| 2024FY | 7.343 | [Polyus FY2024 release](https://polyus.com/en/media/press-releases/financial-results-for-the-second-half-of-2024-and-full-year-2024/): «Revenue $7,343m, +40% vs 2023» |

### Норникель / Северсталь / ММК / НЛМК (RUB, МСФО) — ⚠️ med (агрегатор smart-lab)
Источник: smart-lab.ru, страницы `/q/<TICKER>/f/y/MSFO/` (агрегатор официальной МСФО-отчётности).
Тикеры: GMKN, CHMF, MAGN, NLMK. Значения (млрд ₽):

| Год | Норникель | Северсталь | ММК | НЛМК |
|---|---|---|---|---|
| 2021 | 1 317 | 835.5 | 873.2 | 1 029 |
| 2022 | 1 185 | 682.2 | 699.8 | 900.8 |
| 2023 | 1 172 | 728.3 | 763.4 | 933.4 |
| 2024 | 1 166 | 829.8 | 768.5 | 979.6 |
| 2025 | 1 147 | 712.9 | 609.9 | 831.4 |

**Кросс-валидация (поднимает доверие к источнику):** FY2025 по smart-lab для Северстали (712.9),
ММК (609.9), НЛМК (831.4) **совпадает** с захардкоженными `ACTUAL_REVENUE_12M_2025` в
`osl_metallurgy.py` (712.9 / 609.87 / 831.35). Норникель 2025 = 1 147 ₽млрд ⇒ при USD/RUB ≈ 83.3
это ≈ $13.77 млрд, что сходится с module ACTUAL 13.763 USD (расхождение <0.1%; при панельном
provisional-FX 83.7555 даёт $13.70 млрд — отсюда видно, что точный годовой FX 2025 ещё уточняется).
→ строки 2025 помечены `high`, ранние годы `med`.

> Норникель и Полюс — USD-репортёры; в панели Норникель хранится в **RUB** (smart-lab отдаёт RUB),
> Полюс — в **USD** (первичные релизы + цена золота в USD). Метрики (MAPE) валюто-инвариантны;
> линейные модели используют issuer fixed-effects, поглощающие сдвиг масштаба/валюты.

---

## 2. Цены (panel_prices.csv) — признаки

### Медь / Никель / Платина / Золото / Железная руда — ✅ high (World Bank Pink Sheet)
Источник: [World Bank Commodity Markets — CMO Historical Data Annual](https://thedocs.worldbank.org/en/doc/18675f1d1639c7a34d463f59263ba0a2-0050012025/related/CMO-Historical-Data-Annual.xlsx) (обновл. 06.01.2026). **2025 — ПОЛНЫЙ календарный год** (агентом подтверждено по месячному файлу: copper Jan–Dec 2025, mean=9947.305 ≡ годовое значение).

| Год | Медь $/t | Никель $/t | Платина $/oz | Золото $/oz | Жел.руда $/dmtu |
|---|---|---|---|---|---|
| 2021 | 9 317.05 | 18 464.97 | 1 091.13 | 1 799.63 | 161.71 |
| 2022 | 8 822.37 | 25 833.73 | 961.72 | 1 800.60 | 121.30 |
| 2023 | 8 490.29 | 21 521.12 | 966.36 | 1 942.67 | 120.57 |
| 2024 | 9 142.14 | 16 813.96 | 955.17 | 2 387.70 | 109.40 |
| 2025 | 9 947.31 | 15 162.16 | 1 278.29 | 3 441.51 | 100.18 |

> **Железная руда** (`steel_proxy_iron_ore`) — прокси цены сталелитейного сектора (годовая HRC FOB
> Чёрное море недоступна). Единица — **$/dmtu** (за 1% Fe, бенчмарк 62% Fe cfr China), НЕ $/dmt.
> Признак только для learned-моделей; в StructuralOSL не используется.
> Золото обновлено до первичного World Bank (раньше — metalcharts, согласуется: 1799/1800/1941/2386).

### USD/RUB среднегодовой — ✅ high (IRS) / 2025 ⚠️ low
[IRS Yearly average currency exchange rates](https://www.irs.gov/individuals/international-taxpayers/yearly-average-currency-exchange-rates): 2021=73.686, 2022=69.896, 2023=85.509, 2024=92.837. 2025=83.7555 — [exchangerates.org.uk](https://www.exchangerates.org.uk/USD-RUB-spot-exchange-rates-history-2025.html), provisional (`low`).

---

## 3. Производство (объёмы в panel_revenue.csv) — драйверы

### Норникель Cu/Ni/Pd/Pt — ✅ high (официальные production-релизы)
[Consolidated production results 2022 (EQS)](https://www.eqs-news.com/news/uk-regulatory/nornickel-announces-consolidated-production-results-for-2022/931cc8d8-304c-4724-af9e-5a56a308ef02_en) (2021+2022),
[production results 2024 (PDF)](https://nornik-upload.storage.yandexcloud.net/iblock/cf7/5gbm89s3ovimx76xl2cjsg6vk59ppmo3/nornickel_production_results_2024_eng_full.pdf) (2023+2024),
[production results 2025 (PDF)](https://nornik-upload.storage.yandexcloud.net/iblock/799/7996928fae17be5cfa09346d1dd25b36/nornickel_production_results_2025_eng_full.pdf) (2025). Pd/Pt в koz → переведены в oz (×1000).

### Полюс золото (продажи, oz) — ✅ high
2021=2 736k, 2022=2 423k (Polyus FY2022), 2023=2 908k ([Polyus FY2023](https://polyus.com/en/media/press-releases/financial-results-for-the-second-half-of-2023-and-full-year-2023/)), 2024=3 107k (Polyus FY2024).

### Сталь Северсталь/ММК/НЛМК (crude steel, Mt) — ⚠️ med (worldsteel)
worldsteel «Top steel-producing companies» (вторичная компиляция; росс. компании прекратили детальное
раскрытие после 2022): [2021](https://worldsteel.org/wp-content/uploads/2022_2021-Top-Steel-Producers.pdf),
[2022/2023](https://worldsteel.org/wp-content/uploads/2023_2022-Top-steel-producers-.pdf),
[2024](https://worldsteel.org/wp-content/uploads/World-Steel-in-Figures-2025-1.pdf). НЛМК 2024 = est.

---

## 4. Пробелы (документированы; НЕ фабрикуются)

Формально объявлены в `panel_schema.json → series_registry.documented_gaps`; их отсутствие
проверяется тестами (`test_required_series_have_rows_or_documented_gap`,
`test_issuer_relevant_price_present`) — пробел зафиксирован явно, а не «тихий ноль».

| Что | Статус | Влияние / план |
|---|---|---|
| **Палладий, среднегодовое 2021–2025** | **подтверждённо не найдено** (2 прохода): WB не ведёт Pd; Johnson Matthey PGM Report даёт только график/диапазоны (нет числовой годовой таблицы); LBMA/LPPM — только интерактивная выгрузка; macrotrends/kitco/BullionByPost — 402/403 | Pd ≈30% выручки Норникеля. StructuralOSL фолбэчит на встроенную `PRICES_12M_2025['palladium']`; learned-модели не получают Pd-признак. Снять при доступе к LBMA/LPPM annual file или JM Excel. → раздел в DS_REPORT |
| **Сталь HRC FOB Чёрное море, годовое** | не найдено (Argus/Fastmarkets/Kallanish — paywall; SteelBenchmarker — только spot) | **Добавлен прокси `steel_proxy_iron_ore`** (WB iron ore, high) как признак learned-моделей. StructuralOSL для сталеваров держит цену стали константой (module default); вариация выручки — через объём+FX. Это зона, где learned-модели могут улучшить бейзлайн |
| **USD/RUB 2025** | provisional (`low`) | IRS ещё не опубликовал 2025; сверить, когда выйдет |
| **Сталь 2025 (Северсталь/ММК/НЛМК)** | не подтверждено | worldsteel-список 2025 не вышел; объём 2025 для сталеваров пуст (NaN — легитимен) |

> Методический штамп: все цены — `simple_annual_mean`, окно усреднения = календарный год,
> лежит внутри отчётного периода эмитента (инвариант проверяется `test_panel.py`).
