---
marp: true
theme: gaia
paginate: true
math: katex
backgroundColor: #ffffff
color: #1a1a2e
style: |
  section { font-size: 24px; padding: 44px 58px; justify-content: flex-start; font-family: -apple-system, "Segoe UI", Roboto, sans-serif; }
  h1 { color: #16213e; }
  h2 { color: #0f3460; font-size: 33px; border-bottom: 3px solid #b71540; padding-bottom: 5px; margin: 0 0 14px; }
  h3 { color: #16213e; }
  strong { color: #b71540; }
  a { color: #0f3460; }
  table { font-size: 20px; border-collapse: collapse; width: 100%; }
  th { background: #16213e; color: #fff; }
  td, th { padding: 5px 10px; border: 1px solid #d9e0ee; }
  .katex { font-size: 1.02em; }
  .katex-display { background: linear-gradient(180deg,#f7f9fd,#eaf0fa); border: 1px solid #d3ddf0; border-radius: 12px; padding: 20px 8px; margin: 10px 0; box-shadow: 0 1px 6px rgba(20,32,58,.06); font-size: 1.14em; }
  pre { background: #14203a !important; border-radius: 10px; padding: 15px 18px !important; font-size: 18px; line-height: 1.4; box-shadow: 0 2px 10px rgba(20,32,58,.16); margin: 8px 0; }
  pre code { color: #e7edf7; }
  :not(pre) > code { background: #e6ebf5; color: #0f3460; padding: 1px 7px; border-radius: 4px; font-weight: 600; }
  .cols { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; align-items: center; }
  .c64 { display: grid; grid-template-columns: 1.18fr .82fr; gap: 24px; align-items: center; }
  .c46 { display: grid; grid-template-columns: .82fr 1.18fr; gap: 24px; align-items: center; }
  .big { font-size: 82px; font-weight: 800; color: #b71540; line-height: .98; }
  .mid { font-size: 33px; font-weight: 700; color: #0f3460; line-height: 1.05; }
  .sub { color: #5a6478; font-size: 19px; }
  .cap { color: #5a6478; font-size: 18px; text-align: center; margin-top: -2px; }
  .lead-in { font-size: 26px; color: #16213e; font-weight: 600; line-height: 1.25; }
  .chip { display: inline-block; background: #eef2fa; color: #0f3460; border: 1px solid #d3ddf0; border-radius: 20px; padding: 3px 13px; font-size: 17px; font-weight: 600; margin: 3px 4px 3px 0; }
  .stats { display: grid; grid-template-columns: repeat(4,1fr); gap: 12px; margin-top: 26px; }
  .stat { background: #16213e; border-radius: 9px; padding: 13px 8px; text-align: center; }
  .stat b { display: block; font-size: 26px; color: #ff6b81; line-height: 1.1; }
  .stat span { font-size: 14px; color: #cdd7ec; }
  .tag { display: inline-block; background: #b71540; color: #fff; font-size: 13px; font-weight: 700; padding: 2px 11px; border-radius: 20px; letter-spacing: .05em; }
  ul { margin: 4px 0; }
  li { margin: 3px 0; }
  section.lead { background: #16213e; color: #eaeef7; }
  section.lead h1 { color: #fff; font-size: 58px; }
  section.lead h3 { color: #cdd7ec; font-weight: 500; }
  section.lead strong { color: #ff6b81; }
  section.lead .sub { color: #9fb0d0; }
  section.lead .stat { background: rgba(255,255,255,.07); border: 1px solid rgba(154,208,255,.30); }
  section.backup { background: #f7f8fb; }
  section.backup h2 { color: #16213e; border-bottom-color: #16213e; }
  section.backup h3 { color: #0f3460; }
---

<!--
ТАЙМИНГ: слайды 1–9 = доклад (~5 мин). Слайды 11–16 = резерв для вопросов
(формулы/код по темам) — открывать на нужном. Фокус: значимость выводов, а не общие слова.
-->

<!-- _class: lead -->
<!-- _backgroundColor: #16213e -->
<!-- _color: #eaeef7 -->

# Макро-радар

### Новость → макро РФ → отрасль → **кредитный риск сегмента банка**
### …на **2–3 месяца раньше**, чем эффект дойдёт до отчётности

<div class="stats">
<div class="stat"><b>5 слоёв</b><span>L0 → L3, один прогон</span></div>
<div class="stat"><b>34 модуля · ≈11k строк</b><span>Python</span></div>
<div class="stat"><b>269 тестов · CI</b><span>0 skipped</span></div>
<div class="stat"><b>4 отрасли</b><span>walk-forward + conformal</span></div>
</div>

<!--
(20 сек): Макро-радар превращает новость в оценку изменения вероятности дефолта по
сегменту портфеля — и делает это на 2–3 месяца раньше отчётности. За 5 минут: значимая
проблема, формула каждого слоя, главный результат, и как мы знаем, что выводам можно верить.
-->

---

## Значимая слепая зона: риск считают в изоляции

<div class="c64">
<div>

Классический стресс-тест скорит заёмщика **в изоляции** и упускает то, что в кризис
рушится не один, а **коррелированная группа** через общие цепочки поставок.

$$\underbrace{\textstyle\sum_i \mathrm{PD}_i\,\mathrm{EAD}_i}_{\text{изолированно}}\;\le\;\underbrace{\mathrm{Risk}_{\mathrm{grp}}}_{+28\text{–}70\%}$$

<div class="cap">сумма индивидуальных лимитов — лишь нижняя граница группового риска</div>

</div>
<div style="text-align:center">

<span class="big">+28…70%</span>
<div class="mid">недооценка<br>системного риска</div>
<span class="sub">Fialkowski 2025 · arXiv:2502.17044</span>

</div>
</div>

**Кейс «Мечел» (12.2025):** отсрочка **132 млрд ₽**. Общий драйвер (КС 21%) ударил
металлургов, застройщиков и розницу разом — изолированный тест занизил бы риск на **30–45%**.

<!--
(35 сек): Банки считают риск поштучно. Но математически сумма индивидуальных лимитов — лишь
нижняя граница: реальный групповой риск на 28–70% выше. Живой пример — Мечел: 132 миллиарда,
и это была вся группа под одним драйвером ставки. Изолированный тест занизил бы почти вдвое.
-->

---

## Решение: один прослеживаемый путь = 5 слоёв кода

<div class="c64">
<div>

```python
state = classify(news)             # L0 → 27 подкатегорий шоков
state = enrich_macro(state)        # L1 → РФ-CAI, EPU, режим КС
state["osl"] = osl_predict(state)  # L1.5 → выручка раньше МСФО
state = spillover(state)           # L2 → каскад 7×7 (Fialkowski)
state = segment_impact(state)      # L3 → ΔPD по 5 каналам
```

</div>
<div>

Каждый слой — реальный модуль, каждое число — **привязано к каналу передачи и
публичному источнику**.

<span class="chip">L0 · 5 LLM-агентов + RAG</span>
<span class="chip">L1 · z-score индексы</span>
<span class="chip">L1.5 · OSL прогноз выручки</span>
<span class="chip">L2 · 7×7 spillover</span>
<span class="chip">L3 · сегменты банка</span>

</div>
</div>

State — растущий JSON-dict, `run_pipeline()` гоняет L0→L3 за один прогон.
**Значимость:** от новости до ΔPD сегмента — один прогон, и любой шаг открывается и проверяется руками, а не прячется в весах модели.

<!--
(35 сек): Ядро — одна прослеживаемая цепочка из пяти слоёв, и это реальный код: state течёт
как растущий JSON от классификации новости до удара по сегменту. Каждый слой — модуль, под
каждым числом — канал передачи и публичный источник. Аналитик защищает вывод, а не верит ящику.
-->

---

## L1.5 — операционный сигнал: выручка раньше отчётности

<div class="c64">
<div>

$$R_{i,t}=k_i\sum_{p} V_{i,p,t}\,P_{p,t}\,\mathrm{FX}_t$$

<div class="cap">объёмы V и цены P публикуются на 2–3 мес. раньше МСФО<br>k — скаляр эмитента из train (fixed, без leakage)</div>

</div>
<div>

```python
def predict(self, issuer, t):
    qp = sum(V[i,p,t]*P[p,t]
             for p in prods) * FX[t]
    return self.k[issuer] * qp
```

**Значимость:** прогноз падения выручки эмитента **раньше**, чем выйдет отчётность.

</div>
</div>

Точность на реальных МСФО (металлургия): **НЛМК 0.7%**, **ММК 3.1%** MAE.

Работает для **4 валидированных** отраслей — металлургия, нефтегаз, химия, энергетика (глубина по доступности данных).

<!--
(35 сек): Слой L1.5 — операционный сигнал. Выручка = сумма объём×цена×FX, со скаляром эмитента.
Ключевое: объёмы и цены видны на 2–3 месяца раньше МСФО — значит прогноз падения выручки
появляется раньше отчётности. На реальных данных лучшие сырьевики: MAE 0.7 и 3.1 процента.
-->

---

## Главный результат: сигнал видит тренд, которого не видит история

<div class="c46">
<div style="text-align:center">

<span class="big">100%</span>
<div class="mid" style="color:#16213e">структурная</div>
<span class="sub">12/12 · физсигнал года t</span>
<br><br>
<span class="big" style="color:#9aa3b2">17%</span>
<div class="mid" style="color:#9aa3b2">stale ≤2022</div>
<span class="sub">2/12 · авторегрессия</span>

</div>
<div>

$$I_t=\hat y_t \pm q_{1-\alpha}\!\big(\{|y_j-\hat y_j|\}_{\mathrm{cal}}\big)$$
<div class="cap">split-conformal · энергетика · прогноз 2024–25 по данным ≤2022</div>

Авторегрессия **застревает** на 2022. Структурная **читает контемпоральные физданные**
(генерацию, биржевые цены) → покрывает истину, когда история — нет.

<span class="sub">Это операционная ценность L1.5. (N мал, интервал широкий — направление, а не «доказанные 90%».)</span>

</div>
</div>

<!--
(35 сек): Вот главный результат. Конформный прогноз на 24–25 по данным до 22-го. Авторегрессия
застряла и покрывает истину 17% времени; структурная читает физические данные текущего года и
покрывает 100%. Это и есть операционная ценность: свежий физический сигнал видит тренд, которого
устаревшая история не видит.
-->

---

## L3 — один шок, разные знаки: канальная декомпозиция

$$\Delta \mathrm{PD}_s = A(\kappa)\sum_{c}\; \sigma_{s,c}\;\iota_{c}\;d_{c}\;\mathrm{PD}^{0}_{c}$$
<div class="cap">c — 5 каналов (consumer / oil_rev / fiscal / fx / supply) · A(κ) — усилитель режима КС · знак канала d ∈ {−1, +1}</div>

<div class="cols">
<div>

```python
dPD = amp(ks) * sum(
  sens[s][c] * intensity[shock][c]
  * direction[shock][c]   # знак канала
  * pd0[c] for c in CHANNELS)
```

</div>
<div>

**Значимый вывод — бифуркация:** один шок **одновременно** улучшает одни сегменты и ухудшает другие.

Деэскалация → Brent ↓ → потребсегменты **лучше**, но нефтегаз-корпораты и регионы-доноры **хуже**.

Не глобальный ±1 на всех — знак берётся по каналу; тесты `test_bifurcation_*` это закрепляют.

</div>
</div>

<!--
(35 сек): Слой L3 считает ΔPD сегмента как сумму по пяти каналам передачи, с усилителем режима
ставки и знаком канала. Отсюда неочевидный, но важный вывод — бифуркация: один и тот же шок
одновременно улучшает одни сегменты и ухудшает другие. Деэскалация роняет Brent — потребителям
лучше, нефтегазу и регионам-донорам хуже. Это не «плюс-минус единица на всех».
-->

---

## Как мы знаем, что выводам можно верить

<div class="c64">
<div>

$$\mathrm{DM}=\frac{\bar d}{\sqrt{2\pi\,\hat f_d(0)/T}},\quad d_t=L(e^{1}_t)-L(e^{2}_t)$$
<div class="cap">walk-forward (expanding, сплит по времени) + Diebold–Mariano vs наивной базы</div>

| Модель (металлургия, N=24) | MAPE | DM p |
|---|--:|--:|
| **structural_osl** (Q×P×FX) | **13.7%** | база |
| hist_gbm | 12.1% | 0.66 |
| elasticnet / ridge | 40.8 / 41.8% | 0.02 |

</div>
<div>

**Значимый методологический вывод:** на малых панелях доменный приор **не хуже** ML —
gbm лучше лишь на 1.6 п.п., но DM $p=0.66$ (внутри шума); линейные переобучаются.

Вывод: **выбор модели должен зависеть от режима данных**, а не от моды на глубокие сети.

</div>
</div>

<!--
(40 сек): Как мы знаем, что выводам можно верить — не на глаз, а walk-forward по времени плюс
тест Диболда-Мариано против наивной базы. На металлургии: бустинг номинально лучше на 1.6 пункта,
но p=0.66 — это шум; линейные переобучаются. Значимый вывод: на малых панелях доменный приор не
хуже ML, и выбор модели должен зависеть от режима данных, а не от хайпа.
-->

---

## Инженерия и применение ИИ

<div class="cols">
<div>

**Инженерия**
- **34 модуля · ≈11k строк** · **269 тестов**, 0 skipped
- CI 4 джоба: matrix 3.11/3.12 · e2e smoke L0→L3 · Docker clean-clone · security
- `ruff` + `black` + coverage — жёсткие гейты
- `requirements.lock` → CI и Docker детерминированы

</div>
<div>

**Применение ИИ**
- L0 — **измеренный** слой: gold-set, живой прогон
- Haiku 4.5 → **93% / 100%**; Sonnet 4.6 → **100% / 100%**
- eval вскрыл прод-баг → **regression-тест-страж**
- разработка агентная: сбор данных, верификация, ревью-гейты

</div>
</div>

**Дальше:** **ОИВ large-N** (регионы × годы → 150–425 строк) — пробить статзначимость · калибровка на шоках РФ 2014 / 2020 / 2022 · L2 — графовые методы LPCMCI + DebtRank

<!--
(35 сек): Инженерно — 34 модуля, 11 тысяч строк, 269 тестов, CI из четырёх джобов с Docker
clean-clone, всё запинено и детерминировано. Применение ИИ — L0 не просто промпт, а измеренный
слой: живой прогон, Haiku и Sonnet под 100%, и eval нашёл реальный баг, который я закрыл тестом.
Дальше — региональные бюджеты дают большой N и статзначимость.
-->

---

<!-- _class: lead -->
<!-- _backgroundColor: #16213e -->
<!-- _color: #eaeef7 -->

# Значимость в одном экране

**Опережение** отчётности на 2–3 мес. · **сигнал видит тренд** (100% vs 17%, где история слепа) ·
**бифуркация** сегментов · **каскад** +28–70%, а не сумма лимитов — и всё **прослеживаемо** до канала и источника.

<span class="sub">Цель дальше — довести L2/L3 до калибровки на реальных данных. Готов к вопросам.</span>

<!--
(20 сек): Итог одним экраном: опережаем отчётность, видим тренд там где история слепа, ловим
бифуркацию сегментов и групповой каскад — и всё прослеживаемо. Спасибо, готов к вопросам.
ПЕРСОНАЛИЗИРУЙ строку про цель/мотивацию под свою историю перед защитой.
-->

---

<!-- _class: backup -->

<span class="tag">РЕЗЕРВ · Q&A</span>

## L0 — 5 агентов + измеренная классификация

<div class="cols">
<div>

Конвейер (промпты — markdown, парсятся в рантайме; state — растущий JSON):

`Classifier → Context-RAG → Backtest-Analog → Impact → Summary`

RAG: SQLite + `sqlite-vec` по корпусу `_Анализы/` (58 разборов); эмбеддер — флаг TF-IDF / e5.

| Модель | main | subcat (27) | $ |
|---|--:|--:|--:|
| Haiku 4.5 | 93% | **100%** | 0.055 |
| Sonnet 4.6 | **100%** | **100%** | 0.193 |

</div>
<div>

Живой eval вскрыл прод-баг: плейсхолдеры промпта не совпадали с ключами подстановки →
новость не инжектилась (dry-run это скрывал). Фикс + офлайн-страж:

```python
# tests/test_l0_prompt_contract.py
for key in orchestrator_keys(agent):
    assert f"<{key}>" in prompt   # 15 кейсов
```

</div>
</div>

---

<!-- _class: backup -->

<span class="tag">РЕЗЕРВ · Q&A</span>

## DS — панель, EDA → выбор моделей, провенанс

<div class="cols">
<div>

**Панель:** 24 = 5 эмитентов × FY21–25. Публичные МСФО + World Bank Pink Sheet + USD/RUB +
объёмы (worldsteel).

**EDA диктует модели:**
- смешение валют → `log(R)` + within-issuer FE, currency-invariant MAPE
- мультиколлинеарность (VIF 27) → регуляризация обязательна
- 35% NaN объёмов → gradient boosting с нативным NaN

</div>
<div>

```python
for t in range(2022, 2026):     # expanding
    train = panel[panel.year <  t]  # анти-leakage
    test  = panel[panel.year == t]
    yhat  = model.fit(train).predict(test)
```

**Провенанс:** каждая ячейка — публичный источник; пробелы документированы, не занулены
(палладий ≈30% выручки; HRC spot — прокси iron-ore). `ACTUAL_*` — независимая кросс-проверка.

</div>
</div>

---

<!-- _class: backup -->

<span class="tag">РЕЗЕРВ · Q&A</span>

## Тиринг покрытия — глубина = доступность данных

<div class="cols">
<div>

Слои L0/L1/L2/L3 — **все 7 отраслей**. Различается только **L1.5 (OSL)**:

| Тир | Отрасли |
|---|---|
| **Валидированы** (4) | металлургия, нефтегаз, химия, энергетика |
| **Иллюстративны** (3) | фарма, розница, ОИВ |

Принцип: глубина L1.5 = наличие публичной Q×P-структуры. Манифест `industry_coverage.json`
→ `COVERAGE_TIERS.md` · тест-страж против дрейфа.

</div>
<div>

**Девелопмент — проверен и отклонён:**

IFRS-15 over-time (POC) размазывает выручку по backlog → нет единого физдрайвера.

$$\text{structural } 26.8\% \;>\; \text{persistence } 19.1\%$$

Гипотеза проверена на данных и отклонена (задокументировано) — не добавлена.

</div>
</div>

---

<!-- _class: backup -->

<span class="tag">РЕЗЕРВ · Q&A</span>

## Инженерия — карта кода и CI

<div class="cols">
<div>

**Модули** (≈11k строк): `run_pipeline` 651 · `orchestrator` 525 · `osl_models` 486 ·
`segment_impact` 359 · `osl_walkforward` 270 · `spillover` 213.

**Тесты** 269 (0 skipped): anti-leakage guards, bifurcation, conformal-split, JSON-контракт
stdout, prompt-контракт L0.

</div>
<div>

**CI** (`.github/workflows/test.yml`, 4 джоба):
- `tests` — matrix 3.11/3.12: `ruff` + `black --check` + `pytest --cov`≥60
- `smoke` — e2e L0→L3, ассерт непустых слоёв
- `docker` — clean-clone: тесты внутри сборки
- `security` — gitleaks + pip-audit

`requirements.lock` (numpy 2.4 / scipy 1.17 / sklearn 1.8) + пин `black`/`ruff` → гейты детерминированы.

</div>
</div>

---

<!-- _class: backup -->

<span class="tag">РЕЗЕРВ · Q&A</span>

## L3 — сегменты, каналы, режим КС

<div class="cols">
<div>

**10 базовых сегментов** × размерности (регион / возраст / профессия / продукт / валюта)
→ **18 активных микросегментов**.

**5 каналов** передачи: `consumer` · `oil_revenue` · `fiscal` · `fx` · `supply_chain`.

$$A(\kappa)=\text{усилитель режима КС},\quad d_c\in\{-1,+1\}$$

</div>
<div>

**Бифуркация закреплена тестами:** `test_bifurcation_*` требуют, чтобы один шок давал
**разные знаки** ΔPD по сегментам — «чинить» их к одному знаку нельзя.

**Границы:** L3 не калиброван на портфеле банка (`confidence='low'`, экспертные приоры) —
это интерфейс под будущую ML-модель на реальных PD.

</div>
</div>

---

<!-- _class: backup -->

<span class="tag">РЕЗЕРВ · Q&A</span>

## L2 spillover + дорожная карта

<div class="cols">
<div>

**L2 — матрица 7×7 Fialkowski** (arXiv:2502.17044):

$$\Delta_{\mathrm{out}} = 1.30\cdot M\,\Delta_{\mathrm{in}}$$
<div class="cap">M — матрица 7×7 зависимостей · ×1.30 — нижняя граница диапазона (эвристика, не РФ-калибр.)</div>

Отдельный кредитный канал для шоков ЦБ. Маршрутизация 27 подкатегорий — данными
(`shock_to_industries.json`), не кодом.

</div>
<div>

**Дорожная карта:**
- **ОИВ large-N**: регионы × годы = 150–425 строк → достижима DM-значимость (парадигма фискальная, не Q×P)
- калибровка на шоках РФ 2014 / 2020 / 2022
- L2 — графовые методы: **LPCMCI** + **DebtRank**
- миграция во внутренний контур (реальные PD → калибровка L3)

</div>
</div>
