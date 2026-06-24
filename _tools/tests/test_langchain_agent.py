"""Тесты LangChain-интеграции: тулы исполняются, симуляция покрывает все сценарии,
проводка реального агента конструируется (без ключа/сети)."""

import sys
from pathlib import Path

import pytest

pytest.importorskip("langchain_core")  # пропустить, если LangChain не установлен

TOOLS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLS))

import langchain_agent as la  # noqa: E402


def test_all_tools_registered():
    names = {getattr(t, "name", "") for t in la.TOOLS}
    assert {"get_macro_state", "classify_news_shock", "osl_forecast",
            "industry_spillover", "segment_impact", "run_credit_pipeline",
            "update_macro_state", "find_news_analogs"} <= names


def test_tool_get_macro_state():
    out = la.get_macro_state.invoke({})
    assert "cai" in out and "kc_regime" in out and "epu_degraded" in out


def test_tool_classify_and_pipeline():
    c = la.classify_news_shock.invoke(
        {"news_text": "ЕС готовит новый пакет санкций против металлургии"})
    assert c["subcategory"] == "1.4"
    out = la.run_credit_pipeline.invoke(
        {"news_text": "ЕС готовит новый пакет санкций против металлургии"})
    assert out["subcategory"] == "1.4" and "worst_segment" in out


def test_tool_segment_impact_region_and_bad_args():
    ok = la.segment_impact.invoke(
        {"subcategory": "1.2", "kc_regime": "moderate_stress", "region": "oil_region"})
    assert isinstance(ok, dict) and "fl_massovy" in ok
    bad = la.segment_impact.invoke({"subcategory": "1.2", "kc_regime": "WRONG"})
    assert "error" in bad


def test_tool_spillover_severity_scaling():
    low = la.industry_spillover.invoke({"industry": "oilgas", "severity_score": 20})
    high = la.industry_spillover.invoke({"industry": "oilgas", "severity_score": 90})
    assert high["magnitude_pp"] > low["magnitude_pp"]


def test_simulation_full_coverage():
    rep = la.simulate_coverage()
    assert rep["errors"] == [], rep["errors"]
    assert rep["all_tools_covered"] is True
    assert rep["subcategories_covered"] == 27
    assert rep["scenarios"] >= 100


def test_agent_loop_offline_and_wiring():
    res = la.simulate_agent_loop()
    assert res.get("ok") is True
    assert res.get("tool_called") == "run_credit_pipeline"
    # реальная проводка create_agent(ChatAnthropic) конструируется с плейсхолдер-ключом
    assert res.get("real_agent_wiring_constructs") is True, res.get("wiring_error")


def test_build_agent_returns_none_without_key():
    # API_KEY пуст по умолчанию → агент не строится, но и не падает
    assert la.build_agent(api_key="") is None
