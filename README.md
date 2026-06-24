# Macro-Radar

A layered analytics pipeline that traces a single news event through the Russian macro
state, cross-industry spillover, and into bank client-segment credit risk — turning "what
just happened" into ΔPD / Δdemand / Δchurn estimates **2–3 months before the effect reaches
financial statements**.

Built as a submission for the ITMO Junior ML Contest. Public data only — no client or
portfolio data.

[![tests](https://github.com/white-mi/jmlc_project/actions/workflows/test.yml/badge.svg)](https://github.com/white-mi/jmlc_project/actions/workflows/test.yml)
![python](https://img.shields.io/badge/python-3.10%2B-blue)

---

## Why it exists

Conventional credit stress-testing scores each borrower in isolation. That misses the
dominant failure mode of a downturn: one macro driver — a key-rate shock, a demand collapse —
hitting a *correlated group* of borrowers at once through shared supply chains. Fialkowski et
al. (2025, [arXiv:2502.17044](https://arxiv.org/abs/2502.17044)) quantify the resulting blind
spot at **+28 % to +70 %** of systemic risk.

Macro-Radar adds the missing layer. For a given event it produces one *traceable* chain:

```
news → Russian macro state → cross-industry cascade → bank client segment
```

Every number is attributable to a transmission channel and a public source, so an analyst can
defend the conclusion instead of trusting a black box.

## The pipeline (five layers)

| Layer | Role | Output |
|---|---|---|
| **L0** News filter | 5-agent LLM pipeline + RAG; classifies the event into one of 27 shock sub-categories | shock type + severity |
| **L1** Macro state | Russian composite activity index, EPU, key-rate regime | macro vector |
| **L1.5** Operational signal (OSL) | Forecasts issuer revenue from physical volumes × prices × FX, ahead of IFRS by 2–3 months | revenue + conformal interval |
| **L2** Industry spillover | 7×7 dependency matrix (Fialkowski), credit-channel propagation | ΔPD by industry |
| **L3** Client segments | Channel decomposition across 18 segments (5 channels) | ΔPD / Δdemand / Δchurn |

Seven industries: oil & gas, metallurgy, chemicals, retail, power, regional governments, pharma.

## Quickstart

```bash
git clone https://github.com/white-mi/jmlc_project.git
cd jmlc_project/_tools

pip install -r ../requirements.txt pytest   # lightweight core (TF-IDF, offline)
python -m pytest tests/ -q                    # 202 passed, 0 skipped

# End-to-end smoke run — numbers at every layer, no LLM call:
python run_pipeline.py --smoke-shock 4.2 --smoke-industry oilgas
```

Docker (the test suite runs inside the build, so the image only builds when green):

```bash
docker build -t macro-radar .
docker run --rm macro-radar
```

## Validation

The data-science layer is validated **out-of-sample**, not by assertion. On a public
metallurgy panel (5 issuers × FY2021–2025, IFRS revenue + exchange prices), expanding-window
walk-forward gives:

| Model | MAPE | vs. structural prior |
|---|---|---|
| Structural OSL (domain formula) | 13.7 % | baseline |
| Gradient boosting | 12.1 % | Diebold–Mariano p = 0.66 (not significant) |
| Regularised linear | ~41 % | overfits price extrapolation |

The honest result: **at N = 24, no learned model significantly beats the domain prior** — and
that is *shown* by walk-forward plus a Diebold–Mariano test, not claimed. Split-conformal
intervals reach their target coverage on a temporal hold-out. Full write-up:
[`docs/DS_REPORT.md`](docs/DS_REPORT.md).

## Repository layout

```
_tools/                 Python package — every layer, plus tests/, data/, calibration/, agents/
  run_pipeline.py         end-to-end L0→L3 in one pass
  osl_*.py                OSL per industry + DS layer (panel / models / walk-forward / conformal)
  backtest_analyses.py    reproducible summary of the saved analysis corpus
docs/                   DS_REPORT, PRODUCT_REPORT, architecture & analyst guides; Fialkowski (2025) paper
_Анализы/               saved news analyses (the radar's track record)
_Справочники/           shock taxonomy, client-segment reference
Dockerfile · Makefile · .github/workflows/test.yml
```

> Documentation under `docs/` is partly in Russian (the project's working language). This
> README and `docs/DS_REPORT.md` are the English entry points.

## Honest limitations

Design facts, not bugs:

- The legacy conformal layer (`conformal_prediction.py`) is **in-sample**; genuine
  out-of-sample validation lives only in the DS layer (`conformal_split.py`).
- **L3 is not calibrated on bank data** (`confidence='low'`, expert priors).
- The ×1.30 spillover amplifier is a Fialkowski heuristic, **not yet calibrated on Russian shocks**.
- The DS layer is deep on one industry (metallurgy, N = 24); the other six rely on in-sample actuals.

## Built with AI

Macro-Radar is both built *with* and built *on* AI: it was developed using Claude Code, and the
L0 layer is itself a five-agent LLM pipeline with retrieval. The contributor guide for the AI
assistant is [`CLAUDE.md`](CLAUDE.md).
