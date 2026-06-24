# Макро-радар — воспроизводимое окружение для тестов и smoke-прогона пайплайна.
# Лёгкий образ: TF-IDF-режим RAG (без sentence-transformers/сети), как в CI.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    RADAR_RAG_USE_ST=0

WORKDIR /app

# Зависимости рантайма/тестов (совпадают с CI). Тяжёлый ML-стек (ST/LLM) не нужен
# для тестов и smoke — подключается отдельно через extras в pyproject.
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt pytest

# Исходники проекта
COPY _tools/ /app/_tools/
COPY _Анализы/ /app/_Анализы/

WORKDIR /app/_tools

# Прогон тестов на этапе сборки = образ собирается только если всё зелёное.
RUN python -m pytest tests/ -q

# По умолчанию — smoke-прогон сквозного пайплайна (числа на всех слоях, без LLM).
CMD ["python", "run_pipeline.py", "--smoke-shock", "4.2", "--smoke-industry", "oilgas"]
