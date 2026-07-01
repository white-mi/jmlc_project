"""Тесты L1 (CAI/EPU), L2 (Spillover), L3 (Segment Impact) и pipeline smoke."""

import pytest

# ============================================================
# L1 — CAI
# ============================================================


def test_cai_loads_current_state():
    from calc_rf_cai import get_current_cai

    r = get_current_cai()
    assert isinstance(r.cai, float)
    assert r.phase in ("expansion", "late-cycle", "recovery", "contraction")
    assert -3 <= r.cai <= 3


def test_cai_backtest_4_of_4_correct():
    """Бэк-тест: все 4 ретроспективных снапшота корректно классифицируются."""
    from calc_rf_cai import compute_cai, load_state

    state = load_state()
    indicators = state["indicators"]
    correct = 0
    total = 0
    for snap in state.get("historical_snapshots", []):
        expected = snap["expected_phase"]
        result = compute_cai(snap, indicators)
        total += 1
        if result.phase == expected:
            correct += 1
    assert correct == total, f"Backtest: {correct}/{total} (expected 4/4)"


# ============================================================
# L1 — EPU
# ============================================================


def test_epu_returns_value_in_range():
    # Считаем EPU на bundled-фикстуре (реальный корпус _Анализы/ внутренний и не входит в репо),
    # чтобы тест был детерминирован и проходил на чистом клоне/в CI.
    from pathlib import Path

    from calc_rf_epu import compute_epu

    fixtures = Path(__file__).parent / "fixtures" / "analyses"
    r = compute_epu(analyses_dir=fixtures)
    assert 0 <= r.epu_value <= 100
    assert r.n_total_texts > 0, "EPU должен прочитать фикстуру-корпус"


# ============================================================
# L2 — Spillover
# ============================================================


def test_spillover_invariants():
    from spillover import matrix_invariants_check

    result = matrix_invariants_check()
    assert result["ok"] is True, f'Issues: {result["issues"]}'
    assert result["industries_count"] == 7


@pytest.mark.parametrize(
    "industry", ["metallurgy", "oilgas", "chemistry", "retail", "energy", "pharma", "oiv"]
)
def test_spillover_diagonal_one(industry):
    """Диагональ матрицы = 1.0 — прямой удар на ту же отрасль."""
    from spillover import propagate_shock

    r = propagate_shock(industry, magnitude_pp=0.5)
    assert abs(r.impacts[industry] - 0.5) < 1e-6


def test_spillover_oilgas_to_oiv_significant():
    """Шок в нефтегазе → значительное воздействие на ОИВ (нефтегазовые регионы)."""
    from spillover import propagate_shock

    r = propagate_shock("oilgas", magnitude_pp=1.0)
    assert r.impacts["oiv"] >= 0.5, f'oilgas → oiv too weak: {r.impacts["oiv"]}'


# ============================================================
# L3 — Segment Impact (v0.8 channel-decomposition)
# ============================================================


def test_segment_table_full_coverage():
    """Проверяем покрытие channels × subcategories × segments (включая подсегменты)."""
    from segment_impact import coverage_check

    result = coverage_check()
    assert result["ok"] is True, f'Issues: {result["issues"]}'
    assert result["channels"] == 5
    assert result["segments"] >= 10, (
        f"Expected ≥10 сегментов (10 base + опциональные подсегменты), " f'got {result["segments"]}'
    )
    assert (
        result["subcategories"] >= 27
    ), f'Expected ≥27 подкатегорий шоков, got {result["subcategories"]}'


@pytest.mark.parametrize("cat", ["1", "2", "3", "4", "5"])
def test_segment_impact_default_subcategory(cat):
    """Top-level category должна разрешаться в default подкатегорию.
    Возвращает результат для всех сегментов (включая подсегменты).
    Для consumer-доминированных сегментов ΔPD ≥ 0 на классических ухудшающих шоках —
    нефтегазовые подсегменты могут иметь бенефит от войны (Brent ↑) — это допустимо.
    """
    from segment_impact import predict_segment_impact

    impacts = predict_segment_impact(cat, kc_regime="normal")
    assert len(impacts) >= 10
    # Consumer-доминированные сегменты должны страдать на ухудшающем шоке
    consumer_dominated = ("fl_massovy", "fl_sredniy", "sme_micro", "sme_small")
    for sgmt in consumer_dominated:
        assert impacts[sgmt].delta_pd >= -1e-6, (
            f"cat={cat} sgmt={sgmt}: consumer-доминированный сегмент должен "
            f"страдать (ΔPD≥0), got {impacts[sgmt].delta_pd}"
        )


def test_subsegment_stratification_diverges_from_base():
    """Стратификация по экспозиции: подсегменты ml_large_corp_oilgas и
    ml_public_oilgas должны страдать ОТ деэскалации сильнее, чем base.
    Подсегменты «не-нефтегазовые» (retail, diversified) — наоборот легче.
    """
    from segment_impact import predict_segment_impact

    impacts = predict_segment_impact("1.2", kc_regime="moderate_stress")

    # Нефтегазовые подсегменты страдают сильнее base
    assert "ml_large_corp_oilgas" in impacts
    assert impacts["ml_large_corp_oilgas"].delta_pd > impacts["ml_large_corp"].delta_pd, (
        f'ml_large_corp_oilgas ({impacts["ml_large_corp_oilgas"].delta_pd}) должен '
        f'страдать сильнее base ml_large_corp ({impacts["ml_large_corp"].delta_pd}) '
        f"при деэскалации (Brent ↓)"
    )

    assert "ml_public_oilgas" in impacts
    assert impacts["ml_public_oilgas"].delta_pd > impacts["ml_public"].delta_pd, (
        f'ml_public_oilgas ({impacts["ml_public_oilgas"].delta_pd}) должен '
        f'страдать сильнее base ml_public ({impacts["ml_public"].delta_pd})'
    )

    # Не-нефтегазовые подсегменты страдают меньше base (или вообще выигрывают)
    assert "ml_large_corp_retail" in impacts
    assert impacts["ml_large_corp_retail"].delta_pd < impacts["ml_large_corp"].delta_pd, (
        "ml_large_corp_retail должен реагировать иначе чем base (ритейл выигрывает "
        "от снижения инфляционного давления)"
    )

    assert "ml_public_diversified" in impacts
    assert (
        impacts["ml_public_diversified"].delta_pd < impacts["ml_public"].delta_pd
    ), "ml_public_diversified (Москва, СПб) должен страдать меньше нефтегазового"


def test_segment_impact_acute_amplifies():
    """Сравнение normal vs acute — acute должен быть в ~2.2 раза сильнее."""
    from segment_impact import predict_segment_impact

    normal = predict_segment_impact("2.1", kc_regime="normal")
    acute = predict_segment_impact("2.1", kc_regime="acute_stress")
    for sgmt in normal:
        if abs(normal[sgmt].delta_pd) > 0.01:  # значимая база
            ratio = acute[sgmt].delta_pd / normal[sgmt].delta_pd
            assert 2.0 < ratio < 2.5, f"{sgmt}: ratio acute/normal = {ratio:.2f}, expected ~2.2"


def test_bifurcation_deescalation_oilgas_vs_consumer():
    """v0.8 ключевой тест: при деэскалации (1.2) ФЛ выигрывают (ΔPD<0),
    нефтегазовые корпы (ml_large_corp) и нефтегазовые регионы (ml_public)
    проигрывают (ΔPD>0). Это и есть бифуркация шока."""
    from segment_impact import predict_segment_impact

    impacts = predict_segment_impact("1.2", kc_regime="moderate_stress")

    # ФЛ — выигрывают (consumer-канал доминирует)
    assert impacts["fl_massovy"].delta_pd < -0.1, (
        f"fl_massovy при деэскалации должен иметь ΔPD<-0.1, "
        f'got {impacts["fl_massovy"].delta_pd}'
    )
    assert impacts["fl_premium"].delta_pd < 0
    assert impacts["sme_micro"].delta_pd < 0

    # Нефтегаз корпы и регионы — страдают (oil_revenue + fiscal)
    assert impacts["ml_large_corp"].delta_pd > 0, (
        f"ml_large_corp (нефтегаз) при Brent ↓ должен иметь ΔPD>0, "
        f'got {impacts["ml_large_corp"].delta_pd}'
    )
    assert impacts["ml_public"].delta_pd > 0.1, (
        f"ml_public (нефтегазовые регионы) должен иметь ΔPD>0.1, "
        f'got {impacts["ml_public"].delta_pd}'
    )


def test_kc_rate_increase_consumer_segments_worse():
    """Шок 4.1 (резкое повышение КС) — потребительские сегменты страдают.
    fl_private/ml_public могут быть около-нейтральны (FX-укрепление компенсирует
    консумер-удар, что физически верно для премиум-сегмента с валютной долей)."""
    from segment_impact import predict_segment_impact

    impacts = predict_segment_impact("4.1", kc_regime="moderate_stress")
    consumer_dominated = (
        "fl_massovy",
        "fl_sredniy",
        "fl_premium",
        "sme_micro",
        "sme_small",
        "sme_mid",
        "ml_mid_corp",
    )
    for sgmt in consumer_dominated:
        assert (
            impacts[sgmt].delta_pd > 0
        ), f"{sgmt}: ΔPD должен быть > 0 при шоке 4.1, got {impacts[sgmt].delta_pd}"


def test_kc_rate_decrease_all_segments_better():
    """Шок 4.2 (резкое снижение КС) — почти все сегменты выигрывают.
    Исключение возможно для FX-чувствительных (ослабление рубля)."""
    from segment_impact import predict_segment_impact

    impacts = predict_segment_impact("4.2", kc_regime="moderate_stress")
    consumer_dominated = ("fl_massovy", "fl_sredniy", "sme_micro", "sme_small")
    for sgmt in consumer_dominated:
        assert impacts[sgmt].delta_pd < 0, (
            f"{sgmt} (consumer-доминированный) при снижении КС должен иметь ΔPD<0, "
            f"got {impacts[sgmt].delta_pd}"
        )


def test_channel_breakdown_returned_when_requested():
    """Breakdown должен раскладывать ΔPD по 5 каналам."""
    from segment_impact import predict_segment_impact

    impacts = predict_segment_impact("1.2", kc_regime="moderate_stress", include_breakdown=True)
    # ml_public имеет ненулевую sensitivity ко всем 5 каналам
    br = impacts["ml_public"].channel_breakdown
    assert br is not None
    assert "oil_revenue" in br and "fiscal" in br
    # Сумма вкладов ≈ итоговая ΔPD (с поправкой на amplifier уже включённый в contrib)
    summed = sum(b["delta_pd_contrib"] for b in br.values())
    assert abs(summed - impacts["ml_public"].delta_pd) < 1e-3


def test_churn_always_positive():
    """Δchurn — отток на любое изменение среды, знак всегда положительный."""
    from segment_impact import predict_segment_impact

    for sub in ("1.1", "1.2", "4.1", "4.2"):
        impacts = predict_segment_impact(sub, kc_regime="moderate_stress")
        for sgmt, imp in impacts.items():
            assert imp.delta_churn >= 0, (
                f"shock={sub} sgmt={sgmt}: Δchurn должен быть ≥0, " f"got {imp.delta_churn}"
            )


def test_legacy_direction_inverts_signs():
    """Legacy: direction=-1 при top-level category (без подкатегории)
    должен инвертировать знаки relative to direction=+1."""
    from segment_impact import predict_segment_impact

    classical = predict_segment_impact("4", kc_regime="normal", direction=1)
    inverted = predict_segment_impact("4", kc_regime="normal", direction=-1)
    for sgmt in classical:
        if abs(classical[sgmt].delta_pd) > 0.01:
            assert (classical[sgmt].delta_pd > 0) != (
                inverted[sgmt].delta_pd > 0
            ), f"{sgmt}: legacy direction=-1 должен инвертировать знак"


# ============================================================
# Pipeline smoke
# ============================================================


def test_pipeline_smoke_runs_to_completion():
    """End-to-end pipeline в smoke режиме (без LLM) проходит без ошибок."""
    from run_pipeline import run_full_pipeline

    state = run_full_pipeline(
        smoke_shock="1.1",
        smoke_industry="oilgas",
        date="2026-04-25",
        skip_l0=False,
    )
    assert "L0_classification" in state
    assert "L1_macro" in state
    assert "L1_5_osl" in state
    assert "L2_spillover" in state
    assert "L3_segments" in state
    assert state["L1_macro"]["phase"] in ("expansion", "late-cycle", "recovery", "contraction")
    # OSL должен возвращать данные хоть для одной отрасли
    osl = state["L1_5_osl"]
    assert any("emitents" in data for data in osl.values()), "No OSL data returned for any industry"
