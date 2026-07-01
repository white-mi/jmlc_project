"""
Drift check скрипт (для регулярного запуска через cron / Task Scheduler).

Прогоняет drift_check для всех 7 OSL-модулей, логирует результат
в `calibration/drift_log.md`. Если drift >5 п.п. для любого эмитента —
exit code 1 (можно использовать для алёрта).

Использование (Windows Task Scheduler):
    cd C:\\path\\to\\_tools
    python scripts\\drift_check.py

Использование (Unix cron):
    0 9 * * 1 cd /path/to/_tools && python scripts/drift_check.py
"""

import sys
from datetime import datetime
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLS_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

LOG_PATH = TOOLS_DIR / "calibration" / "drift_log.md"


def main():
    from osl_calibrator import drift_check, apply_all_calibrations

    # Применяем сохранённые калибровки перед сравнением
    apply_all_calibrations()

    modules = [
        "osl_metallurgy",
        "osl_oilgas",
        "osl_chemistry",
        "osl_pharma",
        "osl_retail",
        "osl_energy",
        "osl_oiv",
    ]

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    flags = []
    output_lines = [f"\n## {timestamp}\n"]

    for module in modules:
        result = drift_check(module)
        if "error" in result:
            output_lines.append(f'- ⚠️ {module}: {result["error"]}')
            continue
        for company, info in result.items():
            flag = info.get("flag", "OK")
            drift_pct = info.get("drift_pct", 0)
            current = info.get("current_mae", 0)
            mark = "✅" if flag == "OK" else "🔴"
            output_lines.append(
                f"- {mark} {module}.{company}: "
                f"current_mae={current:.1f}%, drift={drift_pct:+.1f}% [{flag}]"
            )
            if flag != "OK":
                flags.append(f"{module}.{company}")

    text = "\n".join(output_lines) + "\n"

    # Append to log
    if LOG_PATH.exists():
        existing = LOG_PATH.read_text(encoding="utf-8")
    else:
        existing = "# OSL Drift Log\n\nЕженедельный мониторинг качества калибровки. Запускается через cron / Task Scheduler.\n"
    LOG_PATH.write_text(existing + text, encoding="utf-8")

    # Console output
    print(text)
    if flags:
        print(f'\n🔴 NEEDS_RECALIBRATION: {", ".join(flags)}')
        sys.exit(1)
    print("\n✅ All drift checks passed")


if __name__ == "__main__":
    main()
