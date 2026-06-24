---
tags: [макро-радар, инструменты, регламент]
дата: "2026-04-25"
версия: "1.0"
актуально_для: "v0.9 (июнь 2026)"
---

# Calibration Guide — Бесшовная калибровка OSL-моделей

> [[../Макро-радар — Хаб|← Хаб]]

Регламент использования `osl_calibrator.py` для регулярной калибровки моделей. Цель — **сделать так, чтобы калибровка не была разовым ручным hack'ом, а штатной регулярной процедурой**.

---

## Архитектура

> Актуально для **v0.9** (контекст ниже изначально написан для v0.6 от 2026-04-25; логика калибровки не менялась).

```
_tools/
├── osl_common.py                # общие структуры OSL: RevenuePredict / FXRate / mae_pct
│                                #   (новый в v0.9 — устранено дублирование по 7 модулям;
│                                #    калибратор и conformal опираются на эти структуры)
├── osl_calibrator.py            # auto-tune + apply + drift_check
├── conformal_prediction.py      # автоматически применяет калибровки при импорте (v0.6+)
└── calibration/
    ├── osl_metallurgy_calibrated.json
    ├── osl_oilgas_calibrated.json
    ├── osl_chemistry_calibrated.json
    ├── osl_pharma_calibrated.json
    ├── osl_retail_calibrated.json
    ├── osl_energy_calibrated.json
    └── osl_oiv_calibrated.json
```

---

## 3 базовые операции

### 1. **Калибровка** (после получения новых actuals)

```bash
cd _tools
python osl_calibrator.py --module all
```

Это:
- Подбирает оптимальный параметр для каждого эмитента (grid search + binary refine)
- Минимизирует `|predicted - actual| / actual`
- Сохраняет в `calibration/<module>_calibrated.json`

**Когда запускать:**
- ✅ После каждой публикации квартальной/годовой МСФО (раз в 3 мес)
- ✅ После корректировки реальных параметров (новые цены LME, новые НДПИ-ставки)
- ✅ После добавления нового эмитента в OSL-модуль
- ❌ Не нужно — между событиями (каждый день)

### 2. **Применение** (бесшовно, при каждом импорте Conformal)

```python
from osl_calibrator import apply_all_calibrations
apply_all_calibrations()  # применяет все 7 модулей
```

Это происходит **автоматически** при `import conformal_prediction`. Никаких ручных действий не нужно.

### 3. **Drift check** (мониторинг качества)

```bash
python osl_calibrator.py --module drift
```

Сравнивает текущий MAE с baseline (из последней калибровки). Если drift >5 п.п. — флаг **NEEDS_RECALIBRATION**.

**Когда запускать:**
- Еженедельно (ручной cron)
- Перед каждым multi-agent прогоном (защита от деградации)
- При появлении новости с большим макрошоком (USD/RUB ↑↑, Brent ↓↓)

---

## Процесс при появлении новых данных

### Сценарий 1 — Опубликован новый квартальный отчёт

```bash
# 1. Обновить ACTUAL_REVENUE_xxx_2025 в osl_<отрасль>.py
#    Просто заменить старую цифру на новую

# 2. Запустить калибратор
python osl_calibrator.py --module <отрасль>

# 3. Проверить, что покрытие Conformal сохранилось
python conformal_prediction.py --industry <отрасль>

# 4. Если drift > 5 п.п. — копнуть глубже:
#    - возможно структура revenue изменилась (новый сегмент)
#    - в этом случае требуется ручное обновление model.predict_revenue()
```

### Сценарий 2 — Новые рыночные параметры (LME/Brent ушли)

```bash
# 1. Обновить PRICES_12M_2025 в osl_<отрасль>.py
#    (ил cherrypick через osl_calibrator при подключении API)

# 2. Запустить полный re-calibration
python osl_calibrator.py --module all

# 3. Запустить drift check (для проверки что не сломали что-то ещё)
python osl_calibrator.py --module drift
```

### Сценарий 3 — Добавлен новый эмитент

```bash
# 1. Добавить в osl_<отрасль>.py:
#    - в PROFILES — новый CompanyProfile
#    - в production list — operational data
#    - в ACTUAL_REVENUE_xxx_2025 — фактическая выручка

# 2. Запустить калибратор для этого эмитента (auto)
python osl_calibrator.py --module <отрасль>

# 3. Если хочется добавить wrapper Conformal — отредактировать conformal_prediction.py
#    (для большинства модулей — wrapper уже общий)

# 4. Проверить покрытие
python conformal_prediction.py --industry <отрасль>
```

---

## Что калибрует auto-tune (per module)

| Module | Калибруемый параметр | Диапазон поиска |
|---|---|---|
| `osl_metallurgy` | `profile.other_income_pct` | [0.0; 0.30] |
| `osl_oilgas` | `profile.other_share` | [0.0; 0.50] |
| `osl_chemistry` | `profile.other_income_pct` | [0.0; 0.40] |
| `osl_pharma` | `profile.market_share_retail` | [0.10; 0.30] |
| `osl_retail` | `profile.take_rate` | [0.05; 0.50] |
| `osl_energy` | `profile.revenue_share_other` | [0.05; 0.80] |
| `osl_oiv` | `profile.oil_production_mt` | [0; orig × 3] |

**Один параметр на эмитента** — это упрощение. Для multi-parameter tuning (когда одного недостаточно) — добавить scipy.optimize.differential_evolution с N-параметрами.

---

## Известные ограничения

### 1. Overfitting к одному факту

Auto-tune подбирает **один параметр под один факт** (12М 2025). Это может означать что прогноз хорош для 2025, но плох для 2026 (если структура revenue изменилась).

**Решение:** при появлении нескольких quarterly actuals — minimize MAE через **multiple periods** (не реализовано в v0.6).

### 2. Структурные проблемы — за пределами single-param tune

Если базовая модель **структурно неверна** (как Энергетика с её сложной структурой generation/sales/heat) — auto-tune может скрыть проблему через подгонку one parameter, но при изменении входных условий error вернётся.

**Сигналы:**
- Best param на границе диапазона (например `revenue_share_other = 0.775` — близко к max 0.80)
- Перебор не сходится (большие колебания на разных эмитентах одной отрасли)

В таких случаях — **рефакторить модель**, не маскировать через калибровку.

### 3. Энергетика — формула calculator проблемная

Текущая формула `target_total = subtotal / (1 - other_share)` — при `other_share` близком к 1 даёт **нестабильность**. Видно по результатам v0.6: base прогнозы прыгают, conformal интервалы не согласованы.

**Рекомендация v0.7:** переписать predict_generation в osl_energy.py как **сумму абсолютных сегментов** (gen_revenue + capacity_revenue + sales_revenue + heat_revenue), без divide-by-(1-other) формулы.

---

## Чек-лист регулярной калибровки

Раз в **квартал** (после публикации МСФО эмитентов):

- [ ] Обновить ACTUAL_REVENUE_xxx в каждом OSL-модуле
- [ ] Обновить PRICES_xxx (текущие LME / Brent / FX)
- [ ] Запустить `python osl_calibrator.py --module all`
- [ ] Запустить `python conformal_prediction.py --industry all`
- [ ] Сравнить покрытие с предыдущим (>5 п.п. снижение → флаг)
- [ ] Сохранить отчёт в `_Анализы/<YYYY-MM-DD> — Calibration Quarterly.md`

---

## История калибровок

| Дата | Версия | Покрытие средн. | Изменения |
|---|---|---|---|
| 2026-04-25 | v0.6 | **82%** (23/28) | Auto-calibration framework |
| 2026-04-25 | v0.5 | 46% | Полное покрытие 7 отраслей через wrappers |
| 2026-04-25 | v0.4 | 80% (только metallurgy) | Введена Conformal Prediction |
| 2026-04-25 | v0.3 final | 75% MAE ≤10% | Калибровка metallurgy/NG/химии |

---

*Calibration Guide v1.0 · 2026-04-25 (актуально для v0.9, июнь 2026) · Авто-калибровка делает процесс бесшовным*
