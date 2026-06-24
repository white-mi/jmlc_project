# Макро-радар — воспроизводимое окружение для тестов и smoke-прогона пайплайна.
# Лёгкий образ: TF-IDF-режим RAG (без sentence-transformers/сети), как в CI.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    RADAR_RAG_USE_ST=0

WORKDIR /app

# Зависимости — из закреплённого lock-файла (детерминизм численных результатов). Тяжёлый ML-стек
# (sentence-transformers/LLM) не нужен для тестов и smoke — он в extras pyproject.
COPY requirements.lock /app/requirements.lock
RUN pip install --no-cache-dir -r /app/requirements.lock pytest

# Исходники проекта (тесты гоняются на bundled-фикстуре tests/fixtures/, корпус _Анализы/ не нужен)
COPY _tools/ /app/_tools/

WORKDIR /app/_tools

# Прогон тестов на этапе сборки = образ собирается только если всё зелёное.
RUN python -m pytest tests/ -q

# По умолчанию — smoke-прогон сквозного пайплайна (числа на всех слоях, без LLM).
CMD ["python", "run_pipeline.py", "--smoke-shock", "4.2", "--smoke-industry", "oilgas"]
