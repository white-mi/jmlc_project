# Macro-Radar — Project Description

*ITMO Junior ML Contest submission. Public data only — no client or portfolio data.*

Macro-Radar is a layered analytics pipeline that traces a single news event through the
Russian macro state, cross-industry spillover, and into bank client-segment credit risk —
turning *"what just happened"* into ΔPD / Δdemand / Δchurn estimates **2–3 months before the
effect reaches financial statements**. It is built around one discipline: every number is
attributable to a transmission channel and a public source, and every claim is either
validated out-of-sample or marked as illustrative. This document is the five-page tour; the
companion write-ups are [`DS_REPORT.md`](DS_REPORT.md) (data science),
[`PRODUCT_REPORT.md`](PRODUCT_REPORT.md) (product), and
[`4_Tech_Architecture.md`](4_Tech_Architecture.md) (engineering).

---

## 1. Problem & motivation

Conventional credit stress-testing scores each borrower **in isolation**. That misses the
dominant failure mode of a downturn: one macro driver — a key-rate shock, a demand collapse —
hitting a *correlated group* of borrowers at once through shared supply chains. The sum of
per-borrower limits is then only a lower bound on the real group exposure.

This is a measurable gap, not a rhetorical one:

| Facet of the problem | Quantified | Source |
|---|---|---|
| Systemic-risk understatement when supply chains are ignored | **+28 % to +70 %** (on 100₽ of measured risk, 128–170₽ of real risk) | Fialkowski, Diem, Borsos, Thurner 2025, [arXiv:2502.17044](https://arxiv.org/abs/2502.17044) |
| Understatement on the observed Russian "Mechel credit tsunami" case | **30–45 %** | Vedomosti 2025-12-26, Forbes, ACRA |
| Lag with which an analyst learns of a revenue drop from IFRS reporting | **2–3 months**, after the fact | OSL backtest |

The 2026 context amplifies the gap: the Russian key rate sits in an acute-stress regime
(>18 %), where the credit channel and demand compression amplify correlated defaults — exactly
the scenario in which an isolated stress test errs most. Macro-Radar adds the missing layer:
for a given event it produces one *traceable* chain,

```
news → Russian macro state → cross-industry cascade → bank client segment
```

so an analyst can defend the conclusion instead of trusting a black box.

---

## 2. Solution & architecture

Macro-Radar is five layers. State is a growing JSON dict that each layer enriches;
`run_pipeline.run_full_pipeline()` orchestrates L0→L3 in a single pass and renders the result
to a markdown note.

| Layer | Role | Output |
|---|---|---|
| **L0** News filter | 5-agent LLM pipeline + RAG; classifies the event into one of 27 shock sub-categories | shock type + severity |
| **L1** Macro state | Russian composite activity index (z-score of 6 indicators), EPU (Baker-Bloom-Davis), key-rate regime | macro vector |
| **L1.5** Operational signal (OSL) | Forecasts issuer revenue from physical volumes × prices × FX, ahead of IFRS by 2–3 months | revenue + conformal interval |
| **L2** Industry spillover | 7×7 dependency matrix (Fialkowski); credit-channel propagation for central-bank shocks | ΔPD by industry |
| **L3** Client segments | Channel decomposition across client segments (5 channels × key-rate amplifier) | ΔPD / Δdemand / Δchurn |

Seven industries are covered: oil & gas, metallurgy, chemicals, retail, power, regional
governments, and pharma.

Two design invariants distinguish the architecture from a naive scorecard:

- **L3 is a channel decomposition, not a global ±1 direction.** A single shock can worsen some
  segments and improve others simultaneously (bifurcation): a Brent de-escalation improves
  consumer segments but **worsens** oil & gas corporates and oil-revenue regions. The ΔPD per
  segment is `Σ_k [ sens(segment,k) × intensity(shock,k) × direction(shock,k) × baseline_pd(k) ]`
  over five channels (consumer / oil-revenue / fiscal / FX / supply-chain), multiplied by a
  key-rate amplifier. Dedicated `test_bifurcation_*` tests lock this in.
- **Shock routing is data, not code.** All 27 sub-categories map to industries in
  `data/shock_to_industries.json`, never an inline dictionary — so the taxonomy is auditable
  and extensible.

L0 is itself a five-agent LLM pipeline (Classifier → Context-RAG → Backtest-Analog → Impact →
Summary). Agent prompts live as markdown and are parsed at runtime; the Backtest-Analog agent
retrieves historical analogues from a SQLite + sqlite-vec store over the saved analysis corpus.

---

## 3. Data Science

The DS layer (L1.5) is where the project is validated **out-of-sample, by demonstration rather
than assertion** — and where its most honest result lives. Full write-up:
[`DS_REPORT.md`](DS_REPORT.md).

**Panel.** 24 rows = 5 metallurgy issuers × FY2021–2025 (Polyus 2021–2024). Public data only:
IFRS revenue (issuer IR releases and a public MSFO aggregator, cross-validated against the
project's independent `ACTUAL_REVENUE` constants), commodity prices (World Bank Pink Sheet),
average annual USD/RUB, and production volumes (issuer releases / worldsteel). Two gaps are
documented openly rather than silently zeroed: palladium price (≈30 % of one issuer's revenue;
not published in open annual sources) and HRC steel spot (paid-only, replaced by an iron-ore
sector proxy).

**EDA → modelling choices.** Eight figures, each tied to a modelling implication: currency
mixing two orders of magnitude apart (USD vs RUB issuers) → model `log(revenue)` with
**within-issuer fixed effects** and a currency-invariant MAPE metric; price multicollinearity
(VIF 27 for copper, max |corr| 0.89, 5 periods < 6 prices) → regularisation is mandatory;
35 % NaN in volumes → gradient boosting with native NaN handling, not imputation.

**Models** (one `fit/predict` interface):

| Model | Idea | Rationale |
|---|---|---|
| **StructuralOSL** | Q × P × FX domain formula + one per-issuer scalar correction from train | strong interpretable prior; 1–3 params, cannot overfit |
| **ElasticNet / Ridge** | log-prices + within-issuer FE, target log(revenue) | regularised against price collinearity |
| **HistGradientBoosting** | adds volumes (native NaN), constrained (depth 2, leaf 4, L2 1) | flexible comparator; expected to overfit at N=24 |

**Walk-forward validation** is expanding-window, split strictly by time (train = years < t,
test = year t, folds 2022→2025), with metrics computed on the common set where every model
produces a forecast:

| Model | MAPE | Skill vs structural | Diebold–Mariano p |
|---|---|---|---|
| **structural_osl** | **13.7 %** | 0 (baseline) | — |
| hist_gbm | 12.1 % | +0.12 | **0.66 (not significant)** |
| elasticnet | 40.8 % | −1.97 | 0.024 |
| ridge | 41.8 % | −2.05 | 0.020 |

**The honest headline: at N = 24, no learned model significantly beats the domain prior.** The
gradient booster is nominally 1.6 points better, but Diebold–Mariano p = 0.66 places that
inside the noise; the regularised linear models overfit the 2025 gold-price extrapolation. The
mature DS conclusion — that at this sample size the domain prior is the right choice and ML
flexibility is not warranted — is *shown* by walk-forward plus a DM test, not claimed.

**Conformal intervals.** A split/inductive conformal procedure (proper-train ≤ 2022 →
calibration 2023 → temporal hold-out 2024–2025, using relative residuals that stay exchangeable
under currency mixing) reaches **6/6 = 100 % out-of-sample coverage** against a 90 % target
(conservative, n_calib = 5). This unblocks the project's single previously skipped test
(`test_holdout_coverage_metallurgy`), now genuinely out-of-sample because the panel is
independent of the in-code actuals.

---

## 4. Product thinking

**Target user.** The product's user is a credit / risk analyst (corporate business) at the
bank; the object of analysis is the bank's client segments and seven industries — that is the
model's input/output, not its audience. The analyst's job-to-be-done is to decide *"noise vs.
critical"* on a news event in minutes, and to act on a correlated group before a rating action,
not after IFRS.

**Competitors, honestly.** Full analysis in [`product/COMPETITORS.md`](product/COMPETITORS.md).
The differentiator is **not** "we have supply chains" — Bloomberg's SPLC already maps supplier
graphs. The honest position is an *overlay, not a replacement*:

| Competitor class | Examples | Where Macro-Radar differs |
|---|---|---|
| Global credit platforms | Bloomberg (DRSK/SPLC), Moody's CreditEdge, S&P RiskGauge (400M entities) | Russian-portfolio fit + segment bifurcation + traceability to channel/number; zero licence cost |
| Enterprise stress-test / ECL | SAS Risk Stratum, in-bank CCAR/EBA | cascade across a correlated group vs. per-borrower in isolation |
| Russian incumbents | СПАРК-Интерфакс, ACRA / Expert RA / NKR, central-bank stress test | macro→industry cascade vs. single-company scoring; goal is to *lead* the rating action |

Where the radar **loses** (stated plainly): coverage (~28 issuers vs. millions), maturity (an
experiment, no SLA), real client data (L3 is `confidence='low'`), and a spillover amplifier
that is still a heuristic. The position is: *"a layer on top of the standard stress test — it
does not replace it, it extends it."*

**MVP & impact.** All four/five layers produce numbers in a single run (MVP closed at v0.7,
April 2026; current v0.9.x adds robustness and methodology). Business value is illustrated
retrospectively on **Mechel 2025** — explicitly a *"what the radar would have shown, and when"*
walkthrough, not a live catch. The public facts: revenue −26 %, a 10.4 bn₽ loss, a **132 bn₽**
debt deferral, and an ACRA outlook cut to negative — all of which a per-borrower view treated as
three independent names while one shared driver (key rate 21 % + demand compression) hit
metallurgists, developers and large non-food retail at once. The measurable levers are the OSL
lead-time and the segment bifurcation; the group amplifier (×1.30–1.50) and any monetary
figure are marked **illustrative / not calibrated on Russian data**.

---

## 5. Engineering

The whole pipeline is a Python package in `_tools/` with a thin, deterministic core
(numpy / scipy / scikit-learn / pyyaml) and heavy ML dependencies kept behind optional extras.

- **203 tests, 0 skipped** (`pytest tests/ -q`), including leakage guards (train < test;
  scaler / fixed-effects / calibration fit on train only), metric/DM/conformal-coverage tests,
  and a contract test that `run_pipeline.py --json` keeps stdout pure JSON (import-time prints
  go to stderr).
- **CI** (GitHub Actions) runs four jobs: a Python 3.11/3.12 **matrix** (tests + coverage,
  ruff as a hard gate), an **e2e smoke** job asserting non-empty L1.5/L2/L3 in the pipeline's
  JSON output, a **docker build clean-clone gate** (the test suite runs *inside* the image
  build, so the image only builds when green), and a **security** job (gitleaks secret scan +
  pip-audit on the lockfile).
- **Reproducibility:** a pinned `requirements.lock` (numpy 2.4.2 / scipy 1.17.1 /
  scikit-learn 1.8.0) drives both Docker and CI so numeric results are deterministic; a
  `Makefile` exposes `test / lint / smoke / docker-build`.
- **Built with Claude Code**, and built *on* AI: L0 is itself a five-agent LLM pipeline with
  retrieval. The contributor guide for the AI assistant is `CLAUDE.md`.

A first end-to-end result is one command away:

```bash
cd _tools
pip install -r ../requirements.txt pytest
python -m pytest tests/ -q                                   # 203 passed, 0 skipped
python run_pipeline.py --smoke-shock 4.2 --smoke-industry oilgas   # numbers at every layer, no LLM
```

---

## 6. Honest limitations

These are design facts, stated up front, not discovered bugs:

- **Legacy conformal is in-sample.** `conformal_prediction.py` calibrates on the same hardcoded
  actuals; genuine out-of-sample validation lives only in the DS layer (`conformal_split.py`).
- **L3 is not calibrated on bank data** (`confidence='low'`, expert priors); it is the
  interface for a future ML model, not a fitted one.
- **The ×1.30 spillover amplifier is a Fialkowski heuristic**, the lower bound of the European
  range, **not yet calibrated on Russian shocks** (the plan: COVID-2020 / sanctions-2022 /
  rate-2024).
- **The DS layer is deep on one industry** (metallurgy, N=24); the other six rely on in-sample
  actuals. The infrastructure is industry-agnostic — extending the panel is a matter of adding
  CSV rows — but the out-of-sample claim is, for now, scoped to metallurgy.

The project's strongest signal is precisely this discipline: the forecast core (H1–H3) is
proven out-of-sample, the overlays are implemented and tested but openly marked uncalibrated,
and the one number an analyst might most want — the monetary saving — is labelled illustrative
rather than asserted.

---

*Macro-Radar · 2026-06-24 · public / illustrative data only · ITMO Junior ML Contest*
