# Макро-радар — оркестратор разработки. Запуск из корня репозитория.
.PHONY: help install install-dev test lint fmt smoke docker-build docker-test clean

help:
	@echo "install      — рантайм-зависимости (requirements.txt)"
	@echo "install-dev  — + pytest/ruff/black (editable из _tools)"
	@echo "test         — pytest (269 тестов, TF-IDF режим)"
	@echo "lint         — ruff check (гейт CI)"
	@echo "fmt          — black . (black --check — гейт CI)"
	@echo "smoke        — сквозной smoke-прогон пайплайна (без LLM)"
	@echo "docker-build — собрать образ (прогон тестов внутри сборки)"
	@echo "docker-test  — собрать и запустить тесты в контейнере"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt pytest ruff black
	cd _tools && pip install -e ".[dev]"

test:
	cd _tools && RADAR_RAG_USE_ST=0 python -m pytest tests/ -q

lint:
	cd _tools && ruff check .

fmt:
	cd _tools && black .

smoke:
	cd _tools && python run_pipeline.py --smoke-shock 4.2 --smoke-industry oilgas

docker-build:
	docker build -t macro-radar .

docker-test:
	docker build -t macro-radar . && docker run --rm macro-radar python -m pytest tests/ -q

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	rm -rf _tools/output output .pytest_cache _tools/.pytest_cache
