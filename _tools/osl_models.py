"""
Три модели прогноза годовой выручки эмитентов металлургии — единый интерфейс
fit(rows)/predict(rows) для честной walk-forward валидации (Stage D).

Модели (обоснование выбора — DS_REPORT / EDA):
  1. StructuralOSL  — структурная формула Q×P×FX (reuse predict_global_commodity/
     predict_hybrid из osl_metallurgy) + 1 обучаемый скаляр-коррекция на эмитента
     (median(actual/pred) по train). Сильный интерпретируемый БЕЙЗЛАЙН.
  2. LinearPanel    — ElasticNetCV/RidgeCV на log(цены)+issuer fixed-effects,
     таргет log(выручка). Регуляризация против мультиколлинеарности цен (EDA #7).
  3. GBMPanel       — HistGradientBoostingRegressor (нативный NaN-handling под
     разреженные объёмы, EDA #5), зажат под малый N. Гибкий компаратор.

Только core-зависимости (numpy, scikit-learn) — без pandas, чтобы модели работали
в основном окружении. Таргет каждой строки — в родной валюте эмитента (USD Полюс /
RUB остальные); метрики (MAPE) валюто-инвариантны, поэтому модели предсказывают
target_bn напрямую (в log-пространстве, экспонента назад).

CLI:
  python osl_models.py                  # in-sample + leave-last-period-out превью
"""

import argparse
import sys
from typing import Optional

import numpy as np
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
import osl_panel
import osl_metallurgy as M

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Опорное значение железной руды для масштабирования цены стали в StructuralOSL.
# Фиксированная КОНСТАНТА (не из train/test → без leakage): iron_ore 2025 = 100.18,
# соответствует module-базе steel_fob_chm=510 (откалибрована под 2025). Цена стали
# периода = 510 × iron_ore_период / 100.18 — даёт per-period вариацию через прокси.
IRON_ORE_REF_2025 = 100.18

# Признаки learned-моделей ПО ОТРАСЛИ (ключи цен совпадают с приджойненными в row.prices;
# объёмные колонки — из osl_panel.VOL_COLUMNS). Отрасль выводится из rows[0].industry.
INDUSTRY_PRICE_FEATURES = {
    'metallurgy': ['gold', 'copper', 'nickel', 'platinum', 'steel_proxy_iron_ore', 'usd_rub'],
    'oilgas': ['urals', 'gas_eu', 'lng_jkm', 'usd_rub'],
}
INDUSTRY_VOL_FEATURES = {
    'metallurgy': ['vol_copper_t', 'vol_nickel_t', 'vol_pd_oz', 'vol_pt_oz',
                   'vol_gold_oz', 'vol_steel_t'],
    'oilgas': ['vol_oil_t', 'vol_gas_mmcm', 'vol_refined_t', 'vol_lng_t', 'vol_condensate_t'],
}
# обратная совместимость (металлургия по умолчанию)
PRICE_FEATURES = INDUSTRY_PRICE_FEATURES['metallurgy']
VOL_FEATURES = INDUSTRY_VOL_FEATURES['metallurgy']
# объёмная колонка → металл-ключ структурной модели (металлургия)
VOL_TO_METAL = {'vol_copper_t': 'copper', 'vol_nickel_t': 'nickel', 'vol_pd_oz': 'palladium',
                'vol_pt_oz': 'platinum', 'vol_gold_oz': 'gold'}


def _industry_of(rows) -> str:
    return rows[0].industry if rows else 'metallurgy'


def _price_features(rows):
    return INDUSTRY_PRICE_FEATURES.get(_industry_of(rows), PRICE_FEATURES)


def _vol_features(rows):
    return INDUSTRY_VOL_FEATURES.get(_industry_of(rows), VOL_FEATURES)


def _targets(rows) -> np.ndarray:
    return np.array([r.target_bn for r in rows], dtype=float)


# ============================================================
# 1. StructuralOSL — структурный бейзлайн + скаляр-коррекция
# ============================================================

class StructuralOSL:
    name = 'structural_osl'

    def __init__(self):
        self.k_ = {}        # issuer → масштабный коэффициент (из train)
        self.k_global_ = 1.0

    def _raw(self, row) -> Optional[float]:
        """Сырой структурный прогноз для строки (в валюте таргета).

        Per-period из панели: цены gold/copper/nickel/platinum, FX (usd_rub), все объёмы,
        и цена СТАЛИ — через прокси (steel_fob_chm = 510 × iron_ore_период / 100.18,
        даёт per-period вариацию для сталеваров).

        ЧТО ОСТАЁТСЯ ЗАМОРОЖЕННЫМ (module-константа, НЕ per-period) — честное ограничение:
          • Цена ПАЛЛАДИЯ (≈30% выручки Норникеля) — в панели её нет (documented gap),
            всегда module-default 1050. Норникелевский Pd-компонент не реагирует на год.
          • Байпродукты Норникеля (золото/кобальт/родий ≈7%) — нет объёмов в панели,
            в прогноз не входят (систематический недоучёт, поглощается скаляром k).
        Возвращает None (не падает), если нет ключевого объёма (напр. сталь 2025 — gap)."""
        if row.issuer not in M.PROFILES:
            return None
        prof = M.PROFILES[row.issuer]
        # цены: module-default, переопределяем панельными где есть
        prices = dict(M.PRICES_12M_2025)
        for metal in ('gold', 'copper', 'nickel', 'platinum'):
            pv = row.prices.get(metal)
            if pv is not None:
                base = prices[metal]
                prices[metal] = M.CommodityPrice(metal, base.unit, pv, row.period, 'panel')
        # цена стали — через iron-ore прокси (per-period вместо замороженной 510)
        io = row.prices.get('steel_proxy_iron_ore')
        if io:
            sb = prices['steel_fob_chm']
            prices['steel_fob_chm'] = M.CommodityPrice(
                'steel_fob_chm', sb.unit, sb.avg_price_usd * io / IRON_ORE_REF_2025,
                row.period, 'iron_ore_proxy')
        fx = M.FXRate(avg_usd_rub=row.prices.get('usd_rub') or M.FX_12M_2025.avg_usd_rub,
                      period=row.period)
        if prof.revenue_model == 'global_commodity':
            production = []
            for vcol, metal in VOL_TO_METAL.items():
                v = row.volumes.get(vcol)
                if v:
                    unit = 'oz' if vcol.endswith('_oz') else 't'
                    production.append(M.ProductionData(row.issuer, metal, v, unit, row.period))
            if not production:
                return None
            pred = M.predict_global_commodity(row.issuer, production, prices, prof, fx)
        else:  # hybrid (сталевары)
            v = row.volumes.get('vol_steel_t')
            if not v:
                return None
            production = [M.ProductionData(row.issuer, 'steel', v, 't', row.period)]
            pred = M.predict_hybrid(row.issuer, production, prices, prof, fx)
        return pred.predicted_rub_bn if row.revenue_currency == 'RUB' else pred.predicted_usd_bn

    def fit(self, rows):
        ratios_all = []
        by_issuer = {}
        for r in rows:
            raw = self._raw(r)
            if raw and r.target_bn:
                ratio = r.target_bn / raw
                by_issuer.setdefault(r.issuer, []).append(ratio)
                ratios_all.append(ratio)
        self.k_ = {iss: float(np.median(rs)) for iss, rs in by_issuer.items()}
        self.k_global_ = float(np.median(ratios_all)) if ratios_all else 1.0
        return self

    def predict(self, rows) -> np.ndarray:
        out = []
        for r in rows:
            raw = self._raw(r)
            if raw is None:
                out.append(np.nan)
            else:
                out.append(raw * self.k_.get(r.issuer, self.k_global_))
        return np.array(out, dtype=float)


# ============================================================
# Общий конструктор признаков для learned-моделей
# ============================================================

class _IssuerFE:
    """Panel fixed-effects «внутри-преобразованием»: вычитаем per-issuer среднее log-таргета
    (по TRAIN) → модель учит ЦЕНОВЫЕ отклонения, а не уровень. Это снимает межвалютный
    масштаб (USD Полюс vs RUB сталевары) БЕЗ штрафа на FE (в отличие от one-hot+регуляризация,
    где уровень эмитента ошибочно сжимается к общему среднему). Anti-leakage: средние — только train."""

    def fit(self, rows):
        by = {}
        for r in rows:
            by.setdefault(r.issuer, []).append(np.log(r.target_bn))
        self.issuer_mean_ = {k: float(np.mean(v)) for k, v in by.items()}
        self.global_mean_ = float(np.mean([np.log(r.target_bn) for r in rows]))
        return self

    def level(self, rows) -> np.ndarray:
        return np.array([self.issuer_mean_.get(r.issuer, self.global_mean_) for r in rows])


class _Design:
    """Матрица признаков (log-цены [+ log-объёмы для GBM]); скейлер фитится на train.
    БЕЗ issuer one-hot — уровень эмитента выносится через _IssuerFE (within-FE)."""

    def __init__(self, use_volumes: bool):
        self.use_volumes = use_volumes
        self.scaler_: Optional[StandardScaler] = None

    def fit(self, rows):
        self.scaler_ = StandardScaler().fit(self._log_prices(rows))
        return self

    def _log_prices(self, rows) -> np.ndarray:
        feats = _price_features(rows)
        X = []
        for r in rows:
            X.append([np.log(r.prices[m]) if r.prices.get(m) else np.nan
                      for m in feats])
        return np.array(X, dtype=float)

    def transform(self, rows) -> np.ndarray:
        blocks = [self.scaler_.transform(self._log_prices(rows))]
        if self.use_volumes:
            vfeats = _vol_features(rows)
            vol = np.array([[r.volumes.get(c) if r.volumes.get(c) is not None else np.nan
                             for c in vfeats] for r in rows], dtype=float)
            with np.errstate(invalid='ignore'):
                vol = np.where(vol > 0, np.log(vol), np.nan)  # NaN остаётся (GBM нативно)
            blocks.append(vol)
        return np.hstack(blocks)


# ============================================================
# 2. LinearPanel — ElasticNet/Ridge + issuer FE
# ============================================================

class LinearPanel:
    def __init__(self, kind: str = 'elasticnet'):
        self.kind = kind
        self.name = kind
        self.design = _Design(use_volumes=False)  # цены NaN-free → линейная без объёмов
        self.fe = _IssuerFE()
        self.model = None

    def fit(self, rows):
        self.design.fit(rows)
        self.fe.fit(rows)
        X = self.design.transform(rows)
        y = np.log(_targets(rows)) - self.fe.level(rows)   # within-FE: учим отклонения
        n = len(rows)
        cv = max(2, min(4, n // 4))
        if self.kind == 'ridge':
            self.model = RidgeCV(alphas=np.logspace(-3, 3, 25))
        else:
            self.model = ElasticNetCV(l1_ratio=[0.2, 0.5, 0.8, 0.95],
                                      alphas=np.logspace(-3, 1, 30), cv=cv, max_iter=20000)
        self.model.fit(X, y)
        return self

    def predict(self, rows) -> np.ndarray:
        return np.exp(self.model.predict(self.design.transform(rows)) + self.fe.level(rows))


# ============================================================
# 3. GBMPanel — HistGradientBoosting (нативный NaN), зажат под малый N
# ============================================================

class GBMPanel:
    name = 'hist_gbm'

    def __init__(self):
        self.design = _Design(use_volumes=True)
        self.fe = _IssuerFE()
        self.model = None

    def fit(self, rows):
        self.design.fit(rows)
        self.fe.fit(rows)
        X = self.design.transform(rows)
        y = np.log(_targets(rows)) - self.fe.level(rows)   # within-FE
        self.model = HistGradientBoostingRegressor(
            max_depth=2, max_iter=120, learning_rate=0.05,
            min_samples_leaf=4, l2_regularization=1.0, random_state=0)
        self.model.fit(X, y)
        return self

    def predict(self, rows) -> np.ndarray:
        return np.exp(self.model.predict(self.design.transform(rows)) + self.fe.level(rows))


class PersistenceBaseline:
    """Наивный floor: выручка года t = ПОСЛЕДНЯЯ наблюдённая выручка эмитента в train
    (в родной валюте). Эмитент не в train → медиана train. Всегда конечный прогноз."""
    name = 'persistence'

    def fit(self, rows):
        self.last_ = {}
        for r in sorted(rows, key=lambda x: x.period_end):
            if r.has_target:
                self.last_[r.issuer] = r.target_bn
        tg = [r.target_bn for r in rows if r.has_target]
        self.global_ = float(np.median(tg)) if tg else float('nan')
        return self

    def predict(self, rows) -> np.ndarray:
        return np.array([self.last_.get(r.issuer, self.global_) for r in rows], dtype=float)


class IssuerMeanBaseline:
    """Наивный floor: выручка = среднее выручек эмитента по train (в родной валюте)."""
    name = 'issuer_mean'

    def fit(self, rows):
        by = {}
        for r in rows:
            if r.has_target:
                by.setdefault(r.issuer, []).append(r.target_bn)
        self.mean_ = {k: float(np.mean(v)) for k, v in by.items()}
        tg = [r.target_bn for r in rows if r.has_target]
        self.global_ = float(np.median(tg)) if tg else float('nan')
        return self

    def predict(self, rows) -> np.ndarray:
        return np.array([self.mean_.get(r.issuer, self.global_) for r in rows], dtype=float)


MODELS = {'structural_osl': StructuralOSL, 'elasticnet': lambda: LinearPanel('elasticnet'),
          'ridge': lambda: LinearPanel('ridge'), 'hist_gbm': GBMPanel,
          'persistence': PersistenceBaseline, 'issuer_mean': IssuerMeanBaseline}


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    m = (~np.isnan(y_pred)) & (y_true != 0)
    return float(np.mean(np.abs((y_pred[m] - y_true[m]) / y_true[m])) * 100) if m.any() else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--industry', default='metallurgy')
    args = ap.parse_args()
    rows = osl_panel.load_panel(industry=args.industry)
    if not rows:
        print('Панель пуста.'); return
    rows = [r for r in rows if r.has_target]
    y = _targets(rows)

    print('=' * 64)
    print('  IN-SAMPLE (sanity; НЕ метрика качества — настоящая оценка в Stage D)')
    print('=' * 64)
    for name, ctor in MODELS.items():
        m = ctor().fit(rows)
        print(f'  {name:16s} MAPE_in={mape(y, m.predict(rows)):6.2f}%')

    # leave-last-period-out превью (train ≤2024, test 2025) — намёк на Stage D.
    # ВАЖНЫЙ CAVEAT: встроенные параметры StructuralOSL (PROFILES, PRICES_12M_2025)
    # изначально откалиброваны под FY2025 → тест НА 2025 даёт структурной модели
    # преимущество (цена стали теперь per-period через iron-ore прокси, но цена Pd
    # заморожена — gap, и module-профили тюнились под 2025). Полный walk-forward
    # (Stage D) тестирует и на 2022-2024, где преимущества нет — только там сравнение
    # справедливо. NB: на gap-строках (сталь 2025) StructuralOSL даёт NaN → его
    # эффективная тест-выборка УЖЕ, чем у learned-моделей; Stage D учитывает это явно.
    train = [r for r in rows if r.period_end and r.period_end.year <= 2024]
    test = [r for r in rows if r.period_end and r.period_end.year == 2025]
    if train and test:
        yt = _targets(test)
        print('\n' + '=' * 64)
        print(f'  LEAVE-LAST-OUT превью: train≤2024 ({len(train)}) → test 2025 ({len(test)})')
        print('=' * 64)
        for name, ctor in MODELS.items():
            m = ctor().fit(train)
            print(f'  {name:16s} MAPE_oos={mape(yt, m.predict(test)):6.2f}%')


if __name__ == '__main__':
    main()
