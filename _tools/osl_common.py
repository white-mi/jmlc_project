"""
osl_common — общие структуры для всех OSL-модулей (S3.1).

Раньше `RevenuePredict` и `FXRate` дублировались в каждом из 7 osl_*.py
(правка схемы требовала 7 синхронных изменений). Здесь — единый суперсет:
поля, специфичные для отрасли (usd/rub-выручка, breakdown), опциональны, поэтому
все модули конструируют объект своими kwargs без изменения логики.
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class FXRate:
    """Средний курс USD/RUB за период (для конверсии выручки)."""

    avg_usd_rub: float
    period: str = ""


@dataclass
class RevenuePredict:
    """Унифицированный результат прогноза выручки OSL (суперсет полей 7 отраслей).

    Сырьевые модули (металлургия, нефтегаз, химия) используют usd/breakdown_usd;
    рублёвые (розница, энергетика, фарма, ОИВ) — rub/breakdown_rub. Незаполненные
    поля остаются None.
    """

    company: str
    period: str = ""
    predicted_usd_bn: Optional[float] = None
    predicted_rub_bn: Optional[float] = None
    breakdown_usd_bn: Optional[Dict[str, float]] = None
    breakdown_rub_bn: Optional[Dict[str, float]] = None
    actual_usd_bn: Optional[float] = None
    actual_rub_bn: Optional[float] = None
    mae_pct: Optional[float] = None


def mae_pct(predicted: float, actual: Optional[float]) -> Optional[float]:
    """Относительная ошибка |pred-actual|/actual×100. None при actual в {None, 0}."""
    if not actual:
        return None
    return abs(predicted - actual) / actual * 100
