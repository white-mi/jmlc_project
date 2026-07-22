"""Shared pytest fixtures для тестов Russian Propagation."""

import os
import sys
from pathlib import Path

# Тесты идут в TF-IDF-пространстве: детерминированно, без сети и без загрузки torch/e5.
# Прод-дефолт — e5 (см. embeddings.RAG_USE_ST); реальный e5-прогон запускается отдельно
# (`pytest -m heavy` / `RADAR_RAG_USE_ST=1 python eval_rag.py --embedder e5`).
os.environ.setdefault("RADAR_RAG_USE_ST", "0")

TOOLS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLS_DIR))
sys.path.insert(0, str(TOOLS_DIR / "agents"))
