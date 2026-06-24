"""
RAG v1.1 — embedding provider с двумя режимами.

Режимы:
1. TF-IDF (default, всегда работает): sklearn TfidfVectorizer + усечение/паддинг до 384
2. Sentence-Transformers (опционально, при наличии): multilingual-e5-small / large

Размерность 384 фиксирована (для совместимости с sqlite-vec virtual table).
"""

import os
import sys
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

EMBEDDING_DIM = 384

# S3.3: единый источник истины для выбора эмбеддера. Управляет И индексацией,
# И поиском — чтобы вектора в БД и вектор запроса были в ОДНОМ пространстве.
# Default off (TF-IDF, всегда доступен). Включить нейроэмбеддинги e5-small:
#   1) pip install sentence-transformers
#   2) set RADAR_RAG_USE_ST=1
#   3) переиндексировать БД: python index_news.py  (полный реиндекс)
RAG_USE_ST = os.environ.get('RADAR_RAG_USE_ST', '0') == '1'


class TfidfEmbedder:
    """TF-IDF embedder с PCA-усечением до EMBEDDING_DIM. Работает без тяжёлых моделей."""

    def __init__(self, max_features: int = 5000):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD

        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            stop_words=None,  # для русского нет встроенного списка; терпимо
            sublinear_tf=True,
        )
        self.svd = TruncatedSVD(n_components=EMBEDDING_DIM, random_state=42)
        self.fitted = False

    def fit(self, texts: list[str]):
        """Обучить TF-IDF на корпусе всех анализов."""
        if len(texts) == 0:
            return
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        # SVD требует n_components < n_features (max_features)
        n_components = min(EMBEDDING_DIM, tfidf_matrix.shape[1] - 1, len(texts) - 1)
        if n_components < 1:
            # Слишком мало данных — fallback на нулевые вектора
            self.fitted = False
            return
        self.svd.n_components = n_components
        self.svd.fit(tfidf_matrix)
        self.fitted = True

    def encode(self, text: str) -> np.ndarray:
        """Encoded vector. Если SVD < 384 → padding нулями до 384."""
        if not self.fitted:
            return np.zeros(EMBEDDING_DIM, dtype=np.float32)
        tfidf_v = self.vectorizer.transform([text])
        embedding = self.svd.transform(tfidf_v)[0]
        # Паддинг до 384
        if len(embedding) < EMBEDDING_DIM:
            padded = np.zeros(EMBEDDING_DIM, dtype=np.float32)
            padded[:len(embedding)] = embedding
            embedding = padded
        # Нормализация для cosine
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding.astype(np.float32)


class SentenceTransformerEmbedder:
    """Wrapper над sentence-transformers (если установлен)."""

    def __init__(self, model_name: str = "intfloat/multilingual-e5-small"):
        # multilingual-e5-small = 384 dim (соответствует EMBEDDING_DIM)
        # multilingual-e5-large = 1024 dim (требует другую vec0 dim)
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.available = True
            print(f"  ✅ SentenceTransformer loaded: {model_name}")
        except ImportError:
            print("  ⚠️ sentence-transformers не установлен.")
            print("     Для production: pip install sentence-transformers")
            self.model = None
            self.available = False

    def fit(self, texts: list[str]):
        """No-op — pretrained модель не требует обучения."""
        pass

    def encode(self, text: str) -> np.ndarray:
        if not self.available:
            return np.zeros(EMBEDDING_DIM, dtype=np.float32)
        # E5 models require "query: " or "passage: " prefix
        prefixed = f"passage: {text}"
        emb = self.model.encode(prefixed, normalize_embeddings=True)
        return emb.astype(np.float32)


def get_embedder(prefer_st: bool = RAG_USE_ST):
    """
    Получить embedder с автоматическим выбором.
    prefer_st=True: попытка ST, fallback TFIDF.
    prefer_st=False: всегда TFIDF.
    По умолчанию берётся из RAG_USE_ST (env RADAR_RAG_USE_ST).
    """
    if prefer_st:
        try:
            import sentence_transformers  # noqa: F401  — проба доступности ST
            return SentenceTransformerEmbedder()
        except ImportError:
            pass
    print("  → Using TF-IDF embedder (always available)")
    return TfidfEmbedder()


if __name__ == '__main__':
    print("=" * 60)
    print("  RAG v1.1 — Embedding provider test")
    print("=" * 60)

    embedder = get_embedder(prefer_st=False)  # force TF-IDF for test

    # Тестовый corpus
    texts = [
        "Иран закрыл Ормузский пролив для судов США и Израиля",
        "Наводнение в Дагестане затопило 2000 домов",
        "ЦБ снизил ключевую ставку до 14.5% годовых",
        "Минфин РФ возобновляет операции с валютой и золотом",
    ]
    embedder.fit(texts)

    for t in texts:
        emb = embedder.encode(t)
        print(f"  '{t[:50]}...' → embedding shape={emb.shape}, norm={np.linalg.norm(emb):.4f}")
