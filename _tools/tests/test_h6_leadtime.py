"""
Тесты measure_h6_leadtime.py — прокси-измерение H6 (лид радара над рейтинговыми действиями).

Два уровня:
  1) compute() на синтетике — детерминированная проверка арифметики лага/прокси/медиан;
  2) валидация реального публичного реестра `data/rating_actions.json` (схема, ISO-даты,
     source_url, action ∈ множества, action_date ≥ конец периода) — страж честности данных.
"""

import json
import os
from datetime import date

import pytest

import measure_h6_leadtime as mh

_ALLOWED_ACTIONS = {"affirm", "upgrade", "downgrade", "outlook", "assign"}
_ALLOWED_AGENCIES = {"Эксперт РА", "АКРА", "НКР"}


# ---- 1. арифметика на синтетике (детерминированно) ----

_SYNTH = [
    {
        "issuer": "X",
        "agency": "Эксперт РА",
        "date": "2024-11-02",
        "action": "affirm",
        "financials_period_end": "2024-06-30",
        "source_url": "https://example/x",
    },
    {
        "issuer": "Y",
        "agency": "АКРА",
        "date": "2025-07-21",
        "action": "downgrade",
        "financials_period_end": "2024-12-31",
        "source_url": "https://example/y",
    },
    {
        "issuer": "Z",
        "agency": "Эксперт РА",
        "date": "2023-11-17",
        "action": "affirm",
        "financials_period_end": "2023-06-30",
        "source_url": "https://example/z",
    },
]


def test_compute_arithmetic():
    res = mh.compute(_SYNTH)
    rows = {r["issuer"]: r for r in res["rows"]}
    assert rows["X"]["agency_lag_days"] == 125
    assert rows["X"]["radar_lead_proxy_days"] == 125 - mh.OSL_SIGNAL_LAG_DAYS
    assert rows["Z"]["agency_lag_days"] == 140
    # медиана лагов [125, 140, 202] = 140; прокси = 140 − 30
    assert res["median_agency_lag_days"] == 140
    assert res["median_radar_lead_proxy_days"] == 140 - mh.OSL_SIGNAL_LAG_DAYS
    assert res["n_actions"] == 3 and res["n_with_lag"] == 3
    assert res["share_lead_positive"] == 1.0


def test_compute_handles_missing_period_end():
    actions = _SYNTH + [
        {
            "issuer": "W",
            "agency": "НКР",
            "date": "2024-03-31",
            "action": "affirm",
            "financials_period_end": None,
            "source_url": "https://example/w",
        },
    ]
    res = mh.compute(actions)
    assert res["n_actions"] == 4 and res["n_with_lag"] == 3  # W без периода не в расчёте лага
    w = next(r for r in res["rows"] if r["issuer"] == "W")
    assert w["agency_lag_days"] is None and w["radar_lead_proxy_days"] is None


def test_json_contract_stdout_clean(capsys):
    # --json: stdout должен парситься как чистый JSON
    mh.compute(_SYNTH)  # прогрев
    # эмулируем вызов main c подменённой загрузкой
    orig = mh._load
    mh._load = lambda path=None: _SYNTH
    try:
        mh.main(["--json"])
    finally:
        mh._load = orig
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert parsed["median_agency_lag_days"] == 140
    assert "rows" not in parsed  # summary без сырых строк


# ---- 2. валидация реального публичного реестра ----


def _load_registry():
    if not os.path.exists(mh.DATA_PATH):
        pytest.skip("data/rating_actions.json ещё не собран")
    with open(mh.DATA_PATH, encoding="utf-8") as f:
        return json.load(f)


def test_registry_schema_and_sanity():
    reg = _load_registry()
    assert isinstance(reg, list) and len(reg) >= 8, "нужен набор ≥8 действий для честного 🟡"
    issuers = set()
    for a in reg:
        assert a.get("issuer") and a.get("agency") in _ALLOWED_AGENCIES
        assert a.get("action") in _ALLOWED_ACTIONS
        assert a.get("source_url", "").startswith("http"), f"нет source_url: {a}"
        d = date.fromisoformat(a["date"])  # валидная ISO-дата
        pe = a.get("financials_period_end")
        if pe:
            pe_d = date.fromisoformat(pe)
            assert d >= pe_d, f"действие раньше отчётного периода: {a}"  # без look-ahead
        issuers.add(a["issuer"])
    assert len(issuers) >= 6, "нужно ≥6 разных эмитентов"


def test_registry_computes_positive_lead():
    reg = _load_registry()
    res = mh.compute(reg)
    assert res["n_with_lag"] >= 8
    # честный 🟡: медианный прокси-лид положителен (радар опережает), но это не ✅
    assert res["median_radar_lead_proxy_days"] is not None
    assert res["median_radar_lead_proxy_days"] > 0
