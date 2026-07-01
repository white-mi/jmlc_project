"""
Тесты панели данных (data/panel/) — структура, целостность, анти-leakage.

Главный инвариант (schema.meta.leakage_rule): окно усреднения цены ДОЛЖНО лежать
внутри отчётного периода эмитента. Цена «из будущего» сфабриковала бы skill модели.

Тесты мягко SKIP-аются, если панель ещё не заполнена, — чтобы контрибьюторы без
CSV оставались зелёными (данные собираются итеративно через web-research).
"""

import csv
import json
from pathlib import Path

import pytest

import osl_panel
import osl_metallurgy

SCHEMA = json.loads((osl_panel.PANEL_DIR / "panel_schema.json").read_text(encoding="utf-8"))
# Панель — мультиотраслевой CSV (металлургия + нефтегаз + химия + энергетика). Доп-схемы добавляют
# объёмные колонки и ценовые серии; тесты структуры/серий валидируют ОБЪЕДИНЕНИЕ.
SCHEMA_OG = json.loads(
    (osl_panel.PANEL_DIR / "panel_schema_oilgas.json").read_text(encoding="utf-8")
)
SCHEMA_CH = json.loads(
    (osl_panel.PANEL_DIR / "panel_schema_chemistry.json").read_text(encoding="utf-8")
)
SCHEMA_EN = json.loads(
    (osl_panel.PANEL_DIR / "panel_schema_energy.json").read_text(encoding="utf-8")
)
# не-металлургические объёмы = всё из VOL_COLUMNS, чего нет в металлургической схеме
# (source of truth — код; охватывает oilgas + chemistry + будущие отрасли)
NON_MET_VOLS = [c for c in osl_panel.VOL_COLUMNS if c not in SCHEMA["revenue_columns"]]


def _extra_series(schema):
    reg = schema.get("series_registry", {})
    return set(reg.get("required_series", [])) | set(reg.get("optional_series", {}))


# серии доп-отраслей (для whitelist в test_all_used_series_are_known)
EXTRA_SERIES = _extra_series(SCHEMA_OG) | _extra_series(SCHEMA_CH) | _extra_series(SCHEMA_EN)


def _csv_header(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return next(csv.reader(fh))


def _rows():
    return osl_panel.load_panel(industry="metallurgy")


def _need(rows):
    if not rows:
        pytest.skip("panel_revenue.csv пуст — данные ещё собираются")


# ---------- структура / парсинг ----------


def test_panel_loads():
    _need(_rows())
    rows = _rows()
    assert all(r.issuer for r in rows)
    assert all(r.industry == "metallurgy" for r in rows)


def test_issuers_match_profiles():
    """Эмитент панели должен существовать в PROFILES osl_metallurgy (иначе модель не соберётся)."""
    rows = _rows()
    _need(rows)
    known = set(osl_metallurgy.PROFILES)
    unknown = {r.issuer for r in rows} - known
    assert not unknown, f"Эмитенты вне PROFILES: {unknown}"


def test_high_confidence_rows_have_target():
    """Строки confidence=high обязаны нести таргет (выручку)."""
    rows = _rows()
    _need(rows)
    bad = [(r.issuer, r.period) for r in rows if r.confidence == "high" and not r.has_target]
    assert not bad, f"high-строки без таргета: {bad}"


def test_fy_period_months_is_12():
    rows = _rows()
    _need(rows)
    for r in rows:
        if r.period_kind == "FY":
            assert (
                r.period_months == 12
            ), f"{r.issuer} {r.period}: FY, но period_months={r.period_months}"
            assert r.is_cumulative is True


def test_target_currency_consistent():
    """revenue_currency=USD ⇒ заполнен usd_bn; RUB ⇒ rub_bn."""
    rows = _rows()
    _need(rows)
    for r in rows:
        if not r.has_target:
            continue
        if r.revenue_currency == "USD":
            assert r.revenue_usd_bn is not None, f"{r.issuer} {r.period}: USD без usd_bn"
        elif r.revenue_currency == "RUB":
            assert r.revenue_rub_bn is not None, f"{r.issuer} {r.period}: RUB без rub_bn"


# ---------- АНТИ-LEAKAGE (главный инвариант) ----------


def test_price_window_within_reporting_period():
    """Для каждой цены, джойнящейся к периоду эмитента, окно усреднения ⊆ отчётного периода.
    Любое нарушение = look-ahead (цена включает месяцы вне отчётного окна)."""
    rows = _rows()
    _need(rows)
    prices = osl_panel.load_prices()
    if not prices:
        pytest.skip("panel_prices.csv пуст")

    # период → (period_start, period_end) из revenue-строк
    period_bounds = {}
    for r in rows:
        if r.period_start and r.period_end:
            period_bounds[r.period] = (r.period_start, r.period_end)

    violations = []
    for p in prices:
        bounds = period_bounds.get(p.period)
        if not bounds or not (p.window_start and p.window_end):
            continue
        ps, pe = bounds
        if p.window_start < ps or p.window_end > pe:
            violations.append(
                (p.period, p.series, f"{p.window_start}..{p.window_end} вне {ps}..{pe}")
            )
    assert not violations, f"LEAKAGE — окно цены вне отчётного периода: {violations}"


def test_price_window_ordered():
    """window_start <= window_end для каждой цены."""
    prices = osl_panel.load_prices()
    if not prices:
        pytest.skip("panel_prices.csv пуст")
    for p in prices:
        if p.window_start and p.window_end:
            assert p.window_start <= p.window_end, f"{p.period}/{p.series}: окно перевёрнуто"


# ---------- хелперы для моделей ----------


def test_period_order_is_chronological():
    rows = _rows()
    _need(rows)
    order = osl_panel.period_order(rows)
    ends = [next(r.period_end for r in rows if r.period == p and r.period_end) for p in order]
    assert ends == sorted(ends), "period_order не хронологичен"


def test_to_matrix_alignment():
    rows = _rows()
    _need(rows)
    X, y, meta = osl_panel.to_matrix(rows, ["vol_gold_oz", "price:gold", "period_months"])
    assert len(X) == len(y) == len(meta) == len(rows)
    assert all(len(x) == 3 for x in X)


def test_prices_joined_to_rows():
    """Полюс-строки с золотой ценой за свой период должны получить prices['gold']."""
    rows = _rows()
    _need(rows)
    polyus = [
        r
        for r in rows
        if r.issuer == "Полюс"
        and r.period in {p.period for p in osl_panel.load_prices() if p.series == "lbma_gold"}
    ]
    if not polyus:
        pytest.skip("нет пересечения Полюс×золото")
    assert any("gold" in r.prices for r in polyus), "цена золота не приджойнилась к Полюсу"


# ---------- качество данных ----------


def test_no_duplicate_issuer_period():
    rows = _rows()
    _need(rows)
    keys = [(r.issuer, r.period) for r in rows]
    dups = {k for k in keys if keys.count(k) > 1}
    assert not dups, f"дубли (эмитент, период): {dups}"


def test_revenue_positive():
    rows = _rows()
    _need(rows)
    for r in rows:
        if r.has_target:
            assert r.target_bn > 0, f"{r.issuer} {r.period}: неположительная выручка {r.target_bn}"


def test_volumes_positive_when_present():
    rows = _rows()
    _need(rows)
    for r in rows:
        for k, v in r.volumes.items():
            if v is not None:
                assert v > 0, f"{r.issuer} {r.period}: {k}={v} <= 0"


def test_confidence_in_allowed_set():
    rows = _rows()
    _need(rows)
    allowed = {"high", "med", "low", ""}
    bad = {(r.issuer, r.period, r.confidence) for r in rows if r.confidence not in allowed}
    assert not bad, f"недопустимый confidence: {bad}"


def test_min_periods_per_issuer_for_walkforward():
    """Walk-forward требует >=3 независимых периода на эмитента."""
    rows = _rows()
    _need(rows)
    by_issuer = {}
    for r in rows:
        if r.has_target:
            by_issuer.setdefault(r.issuer, set()).add(r.period)
    thin = {iss: len(ps) for iss, ps in by_issuer.items() if len(ps) < 3}
    assert not thin, f"эмитенты с <3 периодов (мало для walk-forward): {thin}"


def test_prices_present_for_panel_periods():
    """Для каждого периода с эмитентами должна быть хотя бы одна цена (иначе фичи пусты)."""
    rows = _rows()
    _need(rows)
    priced_periods = {p.period for p in osl_panel.load_prices()}
    panel_periods = {r.period for r in rows}
    missing = panel_periods - priced_periods
    assert not missing, f"периоды без цен: {missing}"


# ---------- соответствие схеме / провенанс ----------


def test_csv_headers_match_schema():
    """Заголовки CSV обязаны точно совпадать с ключами схемы (детект дрейфа)."""
    rev_header = _csv_header(osl_panel.REVENUE_CSV)
    # ожидаемый заголовок = металлургическая схема + oilgas-объёмы после vol_steel_t
    expected = list(SCHEMA["revenue_columns"])
    i = expected.index("vol_steel_t") + 1
    expected = expected[:i] + NON_MET_VOLS + expected[i:]
    assert rev_header == expected, "panel_revenue.csv ≠ schema.revenue_columns (+ доп-объёмы)"
    if osl_panel.PRICES_CSV.exists():
        price_header = _csv_header(osl_panel.PRICES_CSV)
        assert price_header == list(
            SCHEMA["prices_columns"]
        ), "panel_prices.csv ≠ schema.prices_columns"


def test_all_used_series_are_known():
    """Каждая series в CSV должна быть в series_to_profile_metal ИЛИ usd_rub ИЛИ в реестре прокси."""
    prices = osl_panel.load_prices()
    if not prices:
        pytest.skip("panel_prices.csv пуст")
    known = set(SCHEMA["series_to_profile_metal"]) | {"usd_rub"}
    known |= set(SCHEMA.get("series_registry", {}).get("proxy_series", {}))
    known |= EXTRA_SERIES  # oilgas + chemistry + energy (electricity_rsv/capacity_kom) серии
    used = {p.series for p in prices}
    unknown = used - known
    assert not unknown, f"неизвестные series (нет в схеме): {unknown}"


def test_required_series_have_rows_or_documented_gap():
    """Серия из required_series обязана иметь строки; отсутствие допустимо ТОЛЬКО если
    она явно в documented_gaps (честная фиксация пробела вместо тихого нуля)."""
    reg = SCHEMA.get("series_registry")
    if not reg:
        pytest.skip("series_registry ещё не объявлен")
    prices = osl_panel.load_prices()
    present = {p.series for p in prices}
    gaps = set(reg.get("documented_gaps", {}))
    missing_required = [s for s in reg.get("required_series", []) if s not in present]
    assert not missing_required, f"required-серии без данных и без gap-пометки: {missing_required}"
    # gap-серии действительно НЕ должны иметь строк (иначе это не gap)
    contradictions = [s for s in gaps if s in present]
    assert not contradictions, f"серии помечены gap, но имеют строки: {contradictions}"


def test_issuer_relevant_price_present():
    """Каждый эмитент должен видеть цену СВОЕГО ключевого драйвера — либо она есть,
    либо драйвер-series в documented_gaps (тогда честно знаем про слепое пятно)."""
    rows = _rows()
    _need(rows)
    reg = SCHEMA.get("series_registry", {})
    gap_metals = set()
    for s in reg.get("documented_gaps", {}):
        gap_metals.add(SCHEMA["series_to_profile_metal"].get(s, s))
    # ключевой металл-драйвер по эмитенту (revenue_model/share)
    driver = {
        "Полюс": "gold",
        "Норникель": "palladium",
        "Северсталь": "steel_fob_chm",
        "ММК": "steel_fob_chm",
        "НЛМК": "steel_fob_chm",
    }
    blind = []
    for r in rows:
        d = driver.get(r.issuer)
        if d and d not in r.prices and d not in gap_metals:
            blind.append((r.issuer, r.period, d))
    assert not blind, f"эмитент без цены ключевого драйвера и без gap-пометки: {blind}"


def test_report_date_after_period_end_when_present():
    """report_date (если заполнен) обязан быть позже конца периода — иначе look-ahead."""
    rows = _rows()
    _need(rows)
    bad = [
        (r.issuer, r.period)
        for r in rows
        if r.report_date and r.period_end and r.report_date <= r.period_end
    ]
    assert not bad, f"report_date <= period_end (look-ahead): {bad}"
