"""
Тиринг покрытия отраслей (Макро-радар) — загрузчик `data/industry_coverage.json` + рендер
`docs/COVERAGE_TIERS.md`.

ПРИНЦИП: глубина L1.5-обработки (OSL) осознанно масштабируется под доступность ПУБЛИЧНЫХ данных.
Слои L0/L1/L2/L3 покрывают все отрасли одинаково; тиры различаются только в L1.5. Числа валидации
читаются ВЖИВУЮ из `output/osl_metrics/*.json` (в манифесте не дублируются → нет дрейфа).

  python coverage_tiers.py      # печать сводки + (пере)генерация docs/COVERAGE_TIERS.md

Имя модуля — `coverage_tiers` (НЕ `coverage`): иначе шадовит библиотеку `coverage`, которую
тянет pytest-cov (`pytest --cov`) → падение сбора тестов в CI. Только core-зависимости (json).
"""

import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

TOOLS = Path(__file__).parent
sys.path.insert(0, str(TOOLS))
REPO = TOOLS.parent
MANIFEST = TOOLS / "data" / "industry_coverage.json"
DOC = REPO / "docs" / "COVERAGE_TIERS.md"

RU = {
    "metallurgy": "Металлургия",
    "oilgas": "Нефтегаз",
    "chemistry": "Химия",
    "energy": "Энергетика",
    "pharma": "Фарма",
    "retail": "Розница",
    "oiv": "ОИВ",
    "developers": "Девелопмент",
}
LAYER_COLS = ["L0", "L1", "L1.5", "L2", "L3"]
# отметка ячейки слоя в сетке
LAYER_MARK = {
    "full": "✅",
    "validated": "✅ структ",
    "structural_deferred": "◐ learned",
    "illustrative": "○ иллюстр",
    "illustrative_pending": "○ pending",
}
TIER_RU = {
    "validated": "валидирован",
    "validated_structural_deferred": "валидирован (структ. отложена)",
    "illustrative": "иллюстративный",
    "illustrative_pending": "иллюстративный (pending)",
}


def load(manifest=MANIFEST) -> dict:
    return json.loads(Path(manifest).read_text(encoding="utf-8"))


def _panel_n(industry: str):
    """Размер панели (строк с таргетом) — headline-N, как в DS_REPORT_*.md. None, если панели нет."""
    try:
        import osl_panel

        return sum(1 for r in osl_panel.load_panel(industry) if r.has_target)
    except Exception:
        return None


def _live_metrics(rel_path: str, industry: str):
    """panel-N / N_common / база / победитель / MAPE из metrics JSON (если есть), иначе None."""
    p = REPO / rel_path
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    s = d.get("summary", {})
    cand = [(m, v.get("mape_common")) for m, v in s.items() if v.get("mape_common") is not None]
    win = min(cand, key=lambda t: t[1]) if cand else (None, None)
    return {
        "panel_n": _panel_n(industry),
        "n_common": d.get("n_common"),
        "base": d.get("base"),
        "winner": win[0],
        "winner_mape": win[1],
    }


def rows(manifest=None) -> list:
    """Список сводок по отраслям (порядок манифеста: валидированные первыми)."""
    man = manifest or load()
    out = []
    for ind, meta in man["industries"].items():
        out.append(
            {
                "industry": ind,
                "tier": meta["tier"],
                "layers": meta["layers"],
                "rationale": meta["rationale"],
                "ds_report": meta.get("ds_report"),
                "module": meta.get("module"),
                "metrics": _live_metrics(meta["metrics"], ind) if meta.get("metrics") else None,
            }
        )
    return out


def _rejected(man: dict) -> dict:
    return {k: v for k, v in man.get("rejected", {}).items() if not k.startswith("_")}


def summary(manifest=None) -> str:
    """Однострочная сводка по тирам (для CLI/печати)."""
    man = manifest or load()
    rs = rows(man)
    by_tier = {}
    for r in rs:
        by_tier.setdefault(r["tier"], []).append(RU.get(r["industry"], r["industry"]))
    parts = [f'{TIER_RU.get(t, t)}: {", ".join(v)}' for t, v in by_tier.items()]
    rej = _rejected(man)
    tail = f' | отклонено: {", ".join(v["ru"].split(" ")[0] for v in rej.values())}' if rej else ""
    return f"{len(rs)} отраслей — " + " | ".join(parts) + tail


def render_markdown(manifest=None) -> str:
    man = manifest or load()
    rs = rows(man)
    n_val = sum(1 for r in rs if r["tier"].startswith("validated"))
    n_ill = len(rs) - n_val
    L = [
        "---",
        "tags: [макро-радар, архитектура, покрытие]",
        f'версия: "{man["_version"]}"',
        'note: "сгенерировано _tools/coverage_tiers.py — не редактировать вручную"',
        "---",
        "",
        "# Покрытие отраслей — тиринг по доступности данных",
        "",
        f"> Радар покрывает **{len(rs)} отраслей**, но обрабатывает их на **разной глубине** — "
        "осознанно. **Слои L0 (фильтр новостей) / L1 (макро) / L2 (спилловер) / L3 (сегменты) "
        f"работают для ВСЕХ {len(rs)}**; различается только **L1.5 (OSL — прогноз выручки)**, и "
        "различается по одному принципу:",
        "",
        "> **Глубина L1.5 = доступность ПУБЛИЧНЫХ данных под Q×P-методологию.**",
        "",
        f"> Из {len(rs)} отраслей **{n_val} валидированы** (реальная панель + out-of-sample "
        f"walk-forward + conformal + DS-отчёт), **{n_ill} иллюстративны** (нет публичной Q×P-структуры "
        "→ OSL-бектест без walk-forward). Это не недоделка, а **задокументированный принцип**: "
        "отрасль получает ровно ту глубину, которую выдерживают её публичные данные.",
        "",
        "Источник правды — [`_tools/data/industry_coverage.json`](../_tools/data/industry_coverage.json) "
        "(проверяется `tests/test_coverage.py`). Числа валидации — вживую из walk-forward JSON; "
        "полный разбор в [`DS_REPORT_SYNTHESIS.md`](DS_REPORT_SYNTHESIS.md).",
        "",
        "## Сетка покрытия (отрасль × слой)",
        "",
        "| Отрасль | " + " | ".join(LAYER_COLS) + " | Тир |",
        "|---|" + "|".join([":-:"] * len(LAYER_COLS)) + "|---|",
    ]
    for r in rs:
        cells = [LAYER_MARK.get(r["layers"].get(c, "full"), "✅") for c in LAYER_COLS]
        L.append(
            f'| **{RU.get(r["industry"], r["industry"])}** | '
            + " | ".join(cells)
            + f' | {TIER_RU.get(r["tier"], r["tier"])} |'
        )
    L += [
        "",
        "✅ полный · ◐ частичный (структурная отложена → learned) · ○ иллюстративный (бектест-only)",
        "",
        "## Валидированные отрасли — глубина L1.5",
        "",
        "_N — размер панели; в скобках n — общий набор, где все модели дали прогноз (на нём MAPE)._",
        "",
        "| Отрасль | N (n) | Победитель WF | MAPE | DS-отчёт |",
        "|---|---|---|---|---|",
    ]
    for r in rs:
        if not r["tier"].startswith("validated"):
            continue
        m = r["metrics"] or {}
        mape = f'{m["winner_mape"]:.1f}%' if m.get("winner_mape") is not None else "—"
        pn, nc = m.get("panel_n"), m.get("n_common")
        ncell = (
            f"{pn} ({nc})" if pn is not None and nc is not None else (nc if nc is not None else "—")
        )
        rep = (
            f'[{Path(r["ds_report"]).name}]({Path(r["ds_report"]).name})' if r["ds_report"] else "—"
        )
        L.append(
            f'| {RU.get(r["industry"], r["industry"])} | {ncell} | {m.get("winner", "—")} | {mape} | {rep} |'
        )
    L += ["", "## Почему каждая отрасль на своей глубине (мотивация)", ""]
    for r in rs:
        tag = "🟢" if r["tier"].startswith("validated") else "⚪"
        L.append(
            f'- {tag} **{RU.get(r["industry"], r["industry"])}** ({TIER_RU.get(r["tier"], r["tier"])}) — '
            + r["rationale"]
        )
    rejected = _rejected(man)
    if rejected:
        L += [
            "",
            "## Оценённые и отклонённые кандидаты",
            "",
            "Гипотезы, которые мы выдвинули, проверили на данных и **отклонили**. Отклонение по "
            "данным — полноценный результат и часть того же принципа (глубина = применимость данных).",
            "",
        ]
        for v in rejected.values():
            doc = Path(v["doc"]).name
            L.append(
                f'- ⛔ **{v["ru"]}** — гипотеза: {v["hypothesis"]} **Вердикт ({v["evaluated"]}):** '
                f'{v["verdict"]} Полный разбор: [{doc}]({doc}).'
            )
    L += [
        "",
        "---",
        "",
        "_Сгенерировано `python _tools/coverage_tiers.py` из `industry_coverage.json`. "
        "Не редактировать вручную — править манифест._",
        "",
    ]
    return "\n".join(L)


def main():
    man = load()
    print("=" * 70)
    print("  " + summary(man))
    print("=" * 70)
    DOC.write_text(render_markdown(man), encoding="utf-8")
    print(f"  → {DOC.relative_to(REPO)}")


if __name__ == "__main__":
    main()
