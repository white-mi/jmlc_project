"""
measure_h6_leadtime.py — прокси-измерение гипотезы H6.

H6: «радар опережает рейтинговые действия АКРА / Эксперт РА». Честного *live*-бэктеста
сигналов радара из корпуса `_Анализы/` не построить (даты там ретроспективны). Поэтому
считаем **прокси** на публичных датированных событиях, без look-ahead:

    agency_lag_days   = дата рейтингового действия − конец периода финотчётности,
                        на которую агентство опиралось  (ИЗМЕРЯЕМЫЙ публичный факт:
                        насколько «протухли» данные агентства к моменту действия).

    osl_signal_lag    = задержка доступности OSL-сигнала после конца периода: физические/
                        производственные отчёты выходят ~месяц после периода и на 2–3 мес.
                        РАНЬШЕ МСФО, на которые опирается агентство (H1, validated OOS).
                        Консервативно OSL_SIGNAL_LAG_DAYS = 30.

    radar_lead_proxy  = agency_lag_days − OSL_SIGNAL_LAG_DAYS   (прокси-лид радара над
                        действием агентства; медиана по набору).

ЧЕСТНЫЙ ПОТОЛОК: это ПРОКСИ, не live-бэктест; малый N; допущение, что OSL срабатывает при
доступности физданных. Результат поддерживает H6 = 🟡 (направление положительное), НЕ ✅.

Данные: `data/rating_actions.json` — публичный реестр (у каждой записи есть `source_url`).
Stdlib-only (без numpy/pyyaml), чтобы не ломать core-CI.

Запуск:
    cd _tools
    python measure_h6_leadtime.py           # печатает сводку в stdout
    python measure_h6_leadtime.py --emit     # + пишет output/backtest/h6_leadtime.md
    python measure_h6_leadtime.py --json      # чистый JSON в stdout (контракт пайплайна)
"""

import argparse
import json
import os
import statistics
import sys
from datetime import date

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_HERE, "data", "rating_actions.json")

# Задержка доступности OSL-сигнала после конца отчётного периода (дни).
# Консервативная константа: производственные/операционные отчёты выходят ~2–4 недели
# после периода и на 2–3 мес. раньше МСФО (H1, validated OOS — docs/DS_REPORT.md §4).
OSL_SIGNAL_LAG_DAYS = 30


def _load(path=None):
    with open(path or DATA_PATH, encoding="utf-8") as f:
        return json.load(f)


def _median_int(values):
    return int(round(statistics.median(values))) if values else None


def compute(actions):
    """Считает per-case лаги и агрегаты. Без побочных эффектов."""
    rows = []
    for a in actions:
        pe = a.get("financials_period_end")
        d = a.get("date")
        lag = proxy = None
        if pe and d:
            lag = (date.fromisoformat(d) - date.fromisoformat(pe)).days
            proxy = lag - OSL_SIGNAL_LAG_DAYS
        rows.append({**a, "agency_lag_days": lag, "radar_lead_proxy_days": proxy})

    lags = [r["agency_lag_days"] for r in rows if r["agency_lag_days"] is not None]
    proxies = [r["radar_lead_proxy_days"] for r in rows if r["radar_lead_proxy_days"] is not None]
    dates = [a["date"] for a in actions if a.get("date")]

    by_agency = {}
    for r in rows:
        if r["agency_lag_days"] is None:
            continue
        by_agency.setdefault(r["agency"], []).append(r["agency_lag_days"])

    return {
        "n_actions": len(actions),
        "n_with_lag": len(lags),
        "n_issuers": len({a.get("issuer") for a in actions}),
        "date_min": min(dates, default=None),
        "date_max": max(dates, default=None),
        "osl_signal_lag_days": OSL_SIGNAL_LAG_DAYS,
        "median_agency_lag_days": _median_int(lags),
        "median_radar_lead_proxy_days": _median_int(proxies),
        "share_lead_positive": (
            round(sum(1 for p in proxies if p > 0) / len(proxies), 2) if proxies else None
        ),
        "median_agency_lag_by_agency": {k: _median_int(v) for k, v in sorted(by_agency.items())},
        "rows": rows,
    }


def render(res):
    """Markdown-рендер с честными кавеатами."""
    ml = res["median_agency_lag_days"]
    mp = res["median_radar_lead_proxy_days"]
    lines = [
        "# H6 — прокси-лид радара над рейтинговыми действиями",
        "",
        "> **Прокси, не live-бэктест.** Считаем на публичных датированных событиях "
        "(реестр `data/rating_actions.json`), без look-ahead из ретроспективного корпуса "
        "`_Анализы/`. Малый N. Результат поддерживает **H6 = 🟡** (направление положительное), "
        "**не ✅**.",
        "",
        f"- Рейтинговых действий: **{res['n_actions']}** по {res['n_issuers']} эмитентам "
        f"(период {res['date_min']} … {res['date_max']})",
        f"- С привязкой к концу отчётного периода (в расчёте лага): **{res['n_with_lag']}**",
        "",
        "**Измеряемый факт** — насколько «протухли» данные агентства к моменту действия:",
        f"- медиана `agency_lag_days` (действие − конец периода финотчётности) = "
        f"**{ml} дн.**"
        + (
            "  ("
            + ", ".join(f"{k}: {v}" for k, v in res["median_agency_lag_by_agency"].items())
            + ")"
            if res["median_agency_lag_by_agency"]
            else ""
        ),
        "",
        "**Прокси-лид радара** над действием агентства (H1: OSL-сигнал доступен из физданных "
        f"~{res['osl_signal_lag_days']} дн. после периода, на 2–3 мес. раньше МСФО):",
        f"- медиана `radar_lead_proxy_days` = `agency_lag_days − {res['osl_signal_lag_days']}` = "
        f"**{mp} дн.**  (доля положительных: {res['share_lead_positive']})",
        "",
        "| Эмитент | Агентство | Дата | Действие | Период фин. | lag, дн. | прокси-лид, дн. |",
        "|---|---|---|---|---|--:|--:|",
    ]
    for r in sorted(res["rows"], key=lambda x: (x.get("issuer", ""), x.get("date", ""))):
        lines.append(
            f"| {r.get('issuer','')} | {r.get('agency','')} | {r.get('date','')} "
            f"| {r.get('action','')} | {r.get('financials_period_end') or '—'} "
            f"| {r['agency_lag_days'] if r['agency_lag_days'] is not None else '—'} "
            f"| {r['radar_lead_proxy_days'] if r['radar_lead_proxy_days'] is not None else '—'} |"
        )
    lines += [
        "",
        "**Кавеаты (честно):** (1) прокси, не live-бэктест реальных сигналов радара; (2) малый N; "
        "(3) допущение, что OSL срабатывает при доступности физданных (~30 дн. после периода); "
        "(4) агентства часто опираются на промежуточную (H1) отчётность — берём период, указанный "
        "в релизе. Источник каждой даты — `source_url` в реестре.",
    ]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description="H6 proxy lead-time measurement")
    ap.add_argument("--emit", action="store_true", help="записать output/backtest/h6_leadtime.md")
    ap.add_argument("--json", action="store_true", help="чистый JSON-результат в stdout")
    args = ap.parse_args(argv)

    actions = _load()
    res = compute(actions)

    if args.json:
        # Контракт: stdout — чистый JSON; всё человекочитаемое — в stderr.
        summary = {k: v for k, v in res.items() if k != "rows"}
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    text = render(res)
    print(text)

    if args.emit:
        out_dir = os.path.join(_HERE, "output", "backtest")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "h6_leadtime.md")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"\n[emit] {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
