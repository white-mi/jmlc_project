"""
Загрузчик панели данных металлургии (эмитент × период) для DS-слоя Макро-радара.

Панель — фундамент честного supervised-обучения и walk-forward валидации OSL:
вместо одной точки (12М 2025) на эмитента собираем исторический ряд FY 2021-2025
из публичных МСФО/IR + биржевых цен. Только stdlib (csv) — без pandas, чтобы не
тянуть тяжёлые зависимости в core (pandas только в опц. extra [eda]).

Файлы:
  data/panel/panel_revenue.csv  — таргет (выручка) + объёмы + источники
  data/panel/panel_prices.csv   — средние цены сырья и FX за период
  data/panel/panel_schema.json  — спецификация колонок

API:
  load_panel(industry=None) -> list[PanelRow]      # строки с приджойненными ценами
  load_prices()             -> list[PricePoint]
  period_order(rows)        -> list[str]            # периоды по возрастанию даты
  to_matrix(rows, feature_cols) -> (X, y, meta)    # numpy-free матрица для моделей

Запуск как скрипт печатает сводку панели:
  python osl_panel.py
  python osl_panel.py --industry metallurgy
"""

import argparse
import csv
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PANEL_DIR = Path(__file__).parent / "data" / "panel"
REVENUE_CSV = PANEL_DIR / "panel_revenue.csv"
PRICES_CSV = PANEL_DIR / "panel_prices.csv"

# series → metal-ключ в PROFILES/PRICES_12M_2025 (osl_metallurgy.py)
SERIES_TO_METAL = {
    "lme_copper": "copper",
    "lme_nickel": "nickel",
    "lbma_gold": "gold",
    "lbma_palladium": "palladium",
    "lbma_platinum": "platinum",
    "steel_fob_bsea": "steel_fob_chm",
}

# Объёмные колонки — ОБЪЕДИНЕНИЕ по всем отраслям (имена различимы; строки отрасли,
# где колонка не применима, оставляют её пустой → None). Расширяется при добавлении отрасли.
VOL_COLUMNS = (
    # металлургия
    "vol_copper_t",
    "vol_nickel_t",
    "vol_pd_oz",
    "vol_pt_oz",
    "vol_gold_oz",
    "vol_steel_t",
    # нефтегаз
    "vol_oil_t",
    "vol_gas_mmcm",
    "vol_refined_t",
    "vol_lng_t",
    "vol_condensate_t",
    # химия
    "vol_fertilizer_kt",
    "vol_polymer_kt",
    # энергетика
    "vol_generation_twh",
    "vol_capacity_gw",
)


# ============================================================
# ПАРСЕРЫ ЯЧЕЕК (пустая ячейка = легитимный NaN/None, не ошибка)
# ============================================================


def _f(val: str) -> Optional[float]:
    """Float или None для пустой ячейки."""
    val = (val or "").strip()
    if val == "" or val.lower() in ("na", "nan", "none", "null"):
        return None
    return float(val)


def _i(val: str) -> Optional[int]:
    f = _f(val)
    return None if f is None else int(f)


def _b(val: str) -> bool:
    return (val or "").strip().lower() in ("1", "true", "yes", "да", "t")


def _d(val: str) -> Optional[date]:
    """ISO-дата YYYY-MM-DD или None."""
    val = (val or "").strip()
    if not val:
        return None
    y, m, d = (int(x) for x in val.split("-"))
    return date(y, m, d)


# ============================================================
# СТРУКТУРЫ
# ============================================================


@dataclass
class PricePoint:
    period: str
    period_end: Optional[date]
    series: str
    avg_value: float
    unit: str
    averaging: str
    window_start: Optional[date]
    window_end: Optional[date]
    source_url: str = ""
    source: str = ""
    confidence: str = ""


@dataclass
class PanelRow:
    issuer: str
    industry: str
    period: str
    period_end: Optional[date]
    period_start: Optional[date]
    period_kind: str
    is_cumulative: bool
    period_months: Optional[int]
    revenue_rub_bn: Optional[float]
    revenue_usd_bn: Optional[float]
    revenue_currency: str
    report_date: Optional[date]
    volumes: Dict[str, Optional[float]] = field(default_factory=dict)
    source_url: str = ""
    source_quote: str = ""
    confidence: str = ""
    # приджойненные цены: metal-ключ → avg_value (USD); 'usd_rub' → FX
    prices: Dict[str, float] = field(default_factory=dict)

    @property
    def target_bn(self) -> Optional[float]:
        """Единый таргет: rub_bn если есть, иначе usd_bn (валюта — в revenue_currency)."""
        return self.revenue_rub_bn if self.revenue_rub_bn is not None else self.revenue_usd_bn

    @property
    def has_target(self) -> bool:
        return self.target_bn is not None


# ============================================================
# ЗАГРУЗКА
# ============================================================


def load_prices(path: Path = PRICES_CSV) -> List[PricePoint]:
    if not path.exists():
        return []
    out: List[PricePoint] = []
    with path.open(encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            if not (r.get("series") or "").strip():
                continue
            out.append(
                PricePoint(
                    period=r["period"].strip(),
                    period_end=_d(r.get("period_end", "")),
                    series=r["series"].strip(),
                    avg_value=float(r["avg_value"]),
                    unit=(r.get("unit") or "").strip(),
                    averaging=(r.get("averaging") or "").strip(),
                    window_start=_d(r.get("window_start", "")),
                    window_end=_d(r.get("window_end", "")),
                    source_url=(r.get("source_url") or "").strip(),
                    source=(r.get("source") or "").strip(),
                    confidence=(r.get("confidence") or "").strip(),
                )
            )
    return out


def _prices_by_period(prices: List[PricePoint]) -> Dict[str, Dict[str, float]]:
    """{period: {metal_key|'usd_rub': avg_value}} — мэтчинг series→metal через SERIES_TO_METAL.

    ВНИМАНИЕ: джойн НЕ фильтрует по релевантности металла эмитенту — каждая строка
    эмитента получает ВСЕ цены своего периода (напр. сталевар ММК получит и gold, и
    copper). Это осознанно: отбор релевантных признаков делегирован слою моделей
    (osl_models.py выбирает feature_cols под бизнес-модель эмитента)."""
    by: Dict[str, Dict[str, float]] = {}
    for p in prices:
        key = SERIES_TO_METAL.get(p.series, p.series)  # usd_rub остаётся usd_rub
        by.setdefault(p.period, {})[key] = p.avg_value
    return by


def load_panel(
    industry: Optional[str] = None, revenue_path: Path = REVENUE_CSV, prices_path: Path = PRICES_CSV
) -> List[PanelRow]:
    """Читает panel_revenue.csv, приджойнивает цены по совпадению `period`."""
    if not revenue_path.exists():
        return []
    price_map = _prices_by_period(load_prices(prices_path))
    rows: List[PanelRow] = []
    with revenue_path.open(encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            if not (r.get("issuer") or "").strip():
                continue
            if industry and (r.get("industry") or "").strip() != industry:
                continue
            row = PanelRow(
                issuer=r["issuer"].strip(),
                industry=(r.get("industry") or "").strip(),
                period=(r.get("period") or "").strip(),
                period_end=_d(r.get("period_end", "")),
                period_start=_d(r.get("period_start", "")),
                period_kind=(r.get("period_kind") or "").strip(),
                is_cumulative=_b(r.get("is_cumulative", "")),
                period_months=_i(r.get("period_months", "")),
                revenue_rub_bn=_f(r.get("revenue_rub_bn", "")),
                revenue_usd_bn=_f(r.get("revenue_usd_bn", "")),
                revenue_currency=(r.get("revenue_currency") or "").strip(),
                report_date=_d(r.get("report_date", "")),
                volumes={c: _f(r.get(c, "")) for c in VOL_COLUMNS},
                source_url=(r.get("source_url") or "").strip(),
                source_quote=(r.get("source_quote") or "").strip(),
                confidence=(r.get("confidence") or "").strip(),
                prices=dict(price_map.get((r.get("period") or "").strip(), {})),
            )
            rows.append(row)
    return rows


# ============================================================
# ХЕЛПЕРЫ ДЛЯ МОДЕЛЕЙ / ВАЛИДАЦИИ
# ============================================================


def period_order(rows: List[PanelRow]) -> List[str]:
    """Уникальные периоды, отсортированные по period_end (хронологически)."""
    seen: Dict[str, Optional[date]] = {}
    for r in rows:
        seen.setdefault(r.period, r.period_end)
    return sorted(seen, key=lambda p: (seen[p] is None, seen[p] or date.min, p))


def to_matrix(
    rows: List[PanelRow], feature_cols: List[str]
) -> Tuple[List[List[Optional[float]]], List[Optional[float]], List[dict]]:
    """numpy-free матрица. feature_cols понимает:
    - 'vol_*' (объёмы), 'price:<metal>' (приджойненная цена), 'usd_rub',
      'period_months'. Возвращает (X, y=target_bn, meta-строки)."""
    X, y, meta = [], [], []
    for r in rows:
        feats: List[Optional[float]] = []
        for c in feature_cols:
            if c.startswith("price:"):
                feats.append(r.prices.get(c.split(":", 1)[1]))
            elif c == "usd_rub":
                feats.append(r.prices.get("usd_rub"))
            elif c == "period_months":
                feats.append(float(r.period_months) if r.period_months else None)
            elif c in VOL_COLUMNS:
                feats.append(r.volumes.get(c))
            else:
                feats.append(None)
        X.append(feats)
        y.append(r.target_bn)
        meta.append(
            {
                "issuer": r.issuer,
                "period": r.period,
                "currency": r.revenue_currency,
                "confidence": r.confidence,
            }
        )
    return X, y, meta


def summary(rows: List[PanelRow]) -> str:
    if not rows:
        return "  (панель пуста — заполни data/panel/panel_revenue.csv)"
    issuers = sorted({r.issuer for r in rows})
    periods = period_order(rows)
    lines = [
        f"  Строк: {len(rows)} | эмитентов: {len(issuers)} | периодов: {len(periods)}",
        f'  Периоды: {", ".join(periods)}',
        "",
    ]
    for iss in issuers:
        ir = [r for r in rows if r.issuer == iss]
        with_t = [r for r in ir if r.has_target]
        with_p = [r for r in ir if r.prices]
        lines.append(
            f"  {iss:12s} — {len(ir)} строк, {len(with_t)} с таргетом, " f"{len(with_p)} с ценами"
        )
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Сводка панели данных металлургии")
    ap.add_argument("--industry", default=None)
    args = ap.parse_args()
    rows = load_panel(industry=args.industry)
    print("=" * 64)
    print("  ПАНЕЛЬ ДАННЫХ — Макро-радар DS-слой")
    print("=" * 64)
    print(summary(rows))


if __name__ == "__main__":
    main()
