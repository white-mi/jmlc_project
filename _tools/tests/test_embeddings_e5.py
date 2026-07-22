"""Контракт нейро-эмбеддера (e5) и гард пространств эмбеддингов.

Тесты CI-safe: реальная модель НЕ скачивается — ST-обёртка подменяется фейком, который
записывает, какой текст ему дали. Проверяем ровно то, что ломается молча:
асимметричные префиксы `query:`/`passage:`, пороги на пространство, и запрет искать
в БД одним эмбеддером, если она проиндексирована другим.

Реальный e5-прогон качества — отдельно: `RADAR_RAG_USE_ST=1 python eval_rag.py --embedder e5`.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLS / "agents" / "rag"))

import embeddings as E  # noqa: E402


class _FakeSTModel:
    """Мини-заглушка sentence-transformers: помнит последний вход, возвращает L2-норм. вектор."""

    def __init__(self):
        self.seen = []

    def encode(self, text, normalize_embeddings=False):
        self.seen.append(text)
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        v = rng.normal(size=E.EMBEDDING_DIM).astype(np.float32)
        if normalize_embeddings:
            v = v / np.linalg.norm(v)
        return v


@pytest.fixture
def fake_st():
    emb = E.SentenceTransformerEmbedder.__new__(E.SentenceTransformerEmbedder)
    emb.model = _FakeSTModel()
    emb.available = True
    return emb


def test_e5_uses_asymmetric_prefixes(fake_st):
    """E5 обучен на паре query/passage. Одинаковый префикс с обеих сторон деградирует
    ранжирование, а внешне всё работает — поэтому это контракт, а не деталь реализации."""
    fake_st.encode("Иран закрыл пролив", is_query=True)
    fake_st.encode("Иран закрыл пролив", is_query=False)
    assert fake_st.model.seen == ["query: Иран закрыл пролив", "passage: Иран закрыл пролив"]


def test_e5_output_shape_and_norm(fake_st):
    v = fake_st.encode("текст", is_query=False)
    assert v.shape == (E.EMBEDDING_DIM,)
    assert v.dtype == np.float32
    assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-5


def test_e5_unavailable_returns_zero_vector():
    emb = E.SentenceTransformerEmbedder.__new__(E.SentenceTransformerEmbedder)
    emb.model, emb.available = None, False
    v = emb.encode("текст")
    assert v.shape == (E.EMBEDDING_DIM,) and not v.any()


def test_embedder_names_and_thresholds():
    assert E.embedder_name(E.TfidfEmbedder()) == E.EMBEDDER_TFIDF
    assert E.SentenceTransformerEmbedder.name == E.EMBEDDER_E5
    # Порог TF-IDF нельзя переносить на e5: у e5 даже неродственные доки дают cos ~0.88
    assert E.default_threshold(E.EMBEDDER_E5) > E.default_threshold(E.EMBEDDER_TFIDF)
    assert E.default_threshold("неизвестное") == E.default_threshold(E.EMBEDDER_TFIDF)


def test_embedder_name_falls_back_for_legacy_artifact():
    """Legacy-артефакты joblib без атрибута `name` не должны ронять поиск."""

    class Legacy:
        pass

    assert E.embedder_name(Legacy()) == E.EMBEDDER_TFIDF


def test_save_embedder_removes_stale_artifact(tmp_path, fake_st):
    """После реиндекса на e5 старый TF-IDF-артефакт обязан исчезнуть, иначе запрос
    закодируется протухшим базисом и уйдёт в чужое пространство."""
    path = tmp_path / "emb.joblib"
    tfidf = E.TfidfEmbedder()
    tfidf.fit(["ставка цб выросла", "наводнение затопило регион", "санкции против банков"])
    assert E.save_embedder(tfidf, path) is True and path.exists()

    assert E.save_embedder(fake_st, path) is False
    assert not path.exists(), "устаревший TF-IDF-артефакт остался после перехода на e5"


def test_query_in_wrong_embedding_space_raises(tmp_path, monkeypatch):
    """БД проиндексирована TF-IDF; запрос приходит в пространстве e5 → явная ошибка,
    а не тихая выдача мусора (косинусы из разных пространств несопоставимы)."""
    import find_analogs as fa
    import index_news as ix
    import init_db

    db = tmp_path / "rag.db"
    analyses = tmp_path / "_Анализы"
    analyses.mkdir()
    (analyses / "2026-06-01 — A.md").write_text(
        '---\nдата_новости: "2026-06-01"\nшок_категория: "4.1 (Повышение КС)"\n---\n'
        "# ЦБ повысил ставку\n\n## L0 — Классификация\n\n"
        "| Параметр | Значение |\n|---|---|\n| **Что произошло** | ЦБ поднял ключевую ставку |\n",
        encoding="utf-8",
    )
    init_db.init_db(db_path=db)
    assert ix.index_all(db_path=db, analyses_dir=analyses, use_st=False) == 1

    # Подменяем эмбеддер на «e5»: персистнутый TF-IDF-базис игнорируем, как будто его нет
    monkeypatch.setattr(E, "load_embedder", lambda _p: None)
    monkeypatch.setattr(E, "get_embedder", lambda prefer_st=True: _fake_e5())

    with pytest.raises(RuntimeError, match="реиндекс"):
        fa.find_analogs("ставка", db_path=db, use_st=True)


def _fake_e5():
    emb = E.SentenceTransformerEmbedder.__new__(E.SentenceTransformerEmbedder)
    emb.model, emb.available = _FakeSTModel(), True
    return emb
