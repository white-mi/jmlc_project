# Contributing

Thanks for your interest in Macro-Radar. This is a research / contest project; contributions
and feedback are welcome.

## Setup

```bash
git clone https://github.com/white-mi/jmlc_project.git
cd jmlc_project/_tools
pip install -r ../requirements.lock pytest ruff black   # locked numeric stack
```

Python **3.11+** required (the pinned `numpy`/`scipy` stack drops 3.10).
Optional extras (RAG embeddings, LLM agents, EDA) are in `_tools/pyproject.toml`:
`pip install -e ".[agent]"` / `".[eda]"`.

## Before opening a PR

Run the same gates CI runs (from `_tools/`):

```bash
ruff check .                  # lint — a hard gate in CI
pytest tests/ -q              # 254 passed, 0 skipped (deterministic on a clean clone)
python run_pipeline.py --smoke-shock 4.2 --smoke-industry oilgas --json | python -m json.tool
```

`black .` is run as a (currently non-blocking) format check — please keep new code formatted.

## Project layout & conventions

- All code lives in `_tools/` (run scripts from there — they `sys.path.insert` and load `data/*.json`
  by relative paths). Layers: `osl_*` (L1.5), `spillover` (L2), `segment_impact` (L3),
  `calc_rf_*`/`fetch_macro_state` (L1), `agents/` (L0 multi-agent + RAG), `run_pipeline` (glue).
- **Determinism matters** (numbers are the result): keep seeds fixed; install from `requirements.lock`.
- **Public data only.** Do not commit client/portfolio data, secrets, or internal documents.
  Copy `.env.example` → `.env` (gitignored) for any API key.
- New tests must pass on a **clean clone** (no reliance on local-only files); use
  `tests/fixtures/` for corpus-like inputs.
- Honest reporting: distinguish in-sample vs out-of-sample; label illustrative numbers as such.

## Commit messages

Conventional-ish prefixes (`feat:`, `fix:`, `ci:`, `docs:`, `chore:`) + a short imperative subject.
