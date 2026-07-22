# Russian Propagation — воспроизводимое окружение для тестов и smoke-прогона пайплайна.
# Лёгкий образ: TF-IDF-режим RAG (без sentence-transformers/сети), как в CI.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    RADAR_RAG_USE_ST=0

WORKDIR /app

# Зависимости — из закреплённого lock-файла (детерминизм численных результатов). Тяжёлый ML-стек
# (sentence-transformers/LLM) не нужен для тестов и smoke — он в extras pyproject.
COPY requirements.lock /app/requirements.lock
# pytest + hypothesis (property-based тесты L2/L3) сверх пиннутого рантайм-стека — как в CI-job tests.
RUN pip install --no-cache-dir -r /app/requirements.lock pytest hypothesis

# Исходники проекта. Живой корпус разборов не нужен (тесты используют фикстуры
# tests/fixtures/), а `_Анализы/_история/` обязателен: на нём считается retrieval-eval
# реального gold-set (tests/test_rag_eval.py) — гейт, который прогоняется внутри сборки.
COPY _tools/ /app/_tools/
COPY _Анализы/_история/ /app/_Анализы/_история/
# docs/ нужен тестам: test_coverage проверяет, что манифест industry_coverage.json ссылается на
# реальные DS-отчёты / COVERAGE_TIERS.md / DEVELOPERS_EVALUATION.md (они лежат в docs/, не в _tools/).
COPY docs/ /app/docs/

WORKDIR /app/_tools

# Прогон тестов на этапе сборки = образ собирается только если всё зелёное.
RUN python -m pytest tests/ -q

# По умолчанию — smoke-прогон сквозного пайплайна (числа на всех слоях, без LLM).
CMD ["python", "run_pipeline.py", "--smoke-shock", "4.2", "--smoke-industry", "oilgas"]
