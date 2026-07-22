"""
RAG — embedding provider с двумя режимами.

Режимы:
1. TF-IDF (default, всегда работает): sklearn TfidfVectorizer + усечение/паддинг до 384
2. Sentence-Transformers (опционально, при наличии): multilingual-e5-small / large

Размерность 384 фиксирована (для совместимости с sqlite-vec virtual table).
"""

import os
import sys
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

EMBEDDING_DIM = 384

# Имена пространств эмбеддингов. Пишутся в `db_meta` при индексации и сверяются при запросе:
# косинусы из разных пространств несопоставимы, поэтому смешение — ошибка, а не «чуть хуже».
EMBEDDER_TFIDF = "tfidf"
EMBEDDER_E5 = "e5-small"

# Пороги similarity зависят от эмбеддера и НЕ переносятся между ними.
# Числа не «на глаз», а по замеру на реальном gold-set (38 запросов, `--gold real`):
#   TF-IDF: top1-косинус min 0.00 / медиана 0.73 / max 0.99, разрыв top1↔top2 медиана 0.33
#           → порог реально отсекает мусор, 0.15 консервативен.
#   e5    : top1-косинус min 0.83 / медиана 0.86 / max 0.89, разрыв top1↔top2 медиана 0.017
#           → абсолютный косинус почти не разделяет документы. Порог 0.80 работает только
#             как грубый отсекатель полного мусора; настоящая фильтрация у e5 — это top_k
#             и метадата-фильтры, а не значение косинуса. Порог 0.86 (медиана!) отрезал бы
#             половину правильных ответов — поэтому «перенести порог TF-IDF на e5» и
#             «взять красивое число» одинаково неверно.
DEFAULT_THRESHOLDS = {EMBEDDER_TFIDF: 0.15, EMBEDDER_E5: 0.80}

# Единый источник истины для выбора эмбеддера. Управляет И индексацией,
# И поиском — чтобы вектора в БД и вектор запроса были в ОДНОМ пространстве.
# Default ON (e5-small): на gold-set он даёт precision@1 100 % против 92 % у TF-IDF
# (docs/EVAL.md). TF-IDF остаётся детерминированным фолбэком без сети и torch —
# его включают через `RADAR_RAG_USE_ST=0` (так работает CI и так же выставляет conftest,
# чтобы тесты не тянули тяжёлую модель).
# Смена флага = смена пространства ⇒ обязателен полный реиндекс: python index_news.py
RAG_USE_ST = os.environ.get("RADAR_RAG_USE_ST", "1") == "1"


class TfidfEmbedder:
    """TF-IDF embedder с PCA-усечением до EMBEDDING_DIM. Работает без тяжёлых моделей."""

    name = EMBEDDER_TFIDF

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

    def encode(self, text: str, is_query: bool = False) -> np.ndarray:
        """Encoded vector. Если SVD < 384 → padding нулями до 384. `is_query` — для
        единого интерфейса с ST-эмбеддером; TF-IDF симметричен, параметр не влияет."""
        if not self.fitted:
            return np.zeros(EMBEDDING_DIM, dtype=np.float32)
        tfidf_v = self.vectorizer.transform([text])
        embedding = self.svd.transform(tfidf_v)[0]
        # Паддинг до 384
        if len(embedding) < EMBEDDING_DIM:
            padded = np.zeros(EMBEDDING_DIM, dtype=np.float32)
            padded[: len(embedding)] = embedding
            embedding = padded
        # Нормализация для cosine
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding.astype(np.float32)


# Кэш загруженных ST-моделей: find_analogs создаёт эмбеддер на КАЖДЫЙ запрос, и без кэша
# eval на 38 запросах перезагружает модель 38 раз (минуты вместо секунд).
_ST_MODEL_CACHE: dict[str, object] = {}


class SentenceTransformerEmbedder:
    """Wrapper над sentence-transformers (если установлен)."""

    name = EMBEDDER_E5

    def __init__(self, model_name: str = "intfloat/multilingual-e5-small"):
        # multilingual-e5-small = 384 dim (соответствует EMBEDDING_DIM)
        # multilingual-e5-large = 1024 dim (требует другую vec0 dim)
        try:
            if model_name not in _ST_MODEL_CACHE:
                from sentence_transformers import SentenceTransformer

                _ST_MODEL_CACHE[model_name] = SentenceTransformer(model_name)
                print(f"  ✅ SentenceTransformer loaded: {model_name}", file=sys.stderr)
            self.model = _ST_MODEL_CACHE[model_name]
            self.available = True
        except ImportError:
            print("  ⚠️ sentence-transformers не установлен.")
            print("     Для production: pip install sentence-transformers")
            self.model = None
            self.available = False

    def fit(self, texts: list[str]):
        """No-op — pretrained модель не требует обучения."""
        pass

    def encode(self, text: str, is_query: bool = False) -> np.ndarray:
        if not self.available:
            return np.zeros(EMBEDDING_DIM, dtype=np.float32)
        # E5 требует АСИММЕТРИЧНЫЙ префикс: "query: " для запроса, "passage: " для документа
        # (часть контракта retrieval; одинаковый префикс деградирует качество).
        prefix = "query: " if is_query else "passage: "
        emb = self.model.encode(prefix + text, normalize_embeddings=True)
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
    print("  → Using TF-IDF embedder (always available)", file=sys.stderr)
    return TfidfEmbedder()


def embedder_name(embedder) -> str:
    """Имя пространства эмбеддингов (`tfidf` / `e5-small`) — то, что пишется в `db_meta`.

    Отдельная функция, а не только атрибут: эмбеддер может прийти из joblib-артефакта,
    сохранённого до появления `name`.
    """
    return getattr(embedder, "name", EMBEDDER_TFIDF)


def default_threshold(name: str) -> float:
    """Порог similarity для конкретного пространства (см. DEFAULT_THRESHOLDS)."""
    return DEFAULT_THRESHOLDS.get(name, DEFAULT_THRESHOLDS[EMBEDDER_TFIDF])


# --- Персист фитнутого TF-IDF-эмбеддера (единый базис index↔query) -----------------
# Косинусы сопоставимы только в одном SVD-базисе, поэтому фитнутый TF-IDF персистится при
# индексации и загружается при запросе — index и query кодируются одним эмбеддером. ST-путь
# не персистим: модель предобучена, fit — no-op, базис уже фиксирован.


def save_embedder(embedder, path) -> bool:
    """Сохранить фитнутый TF-IDF-эмбеддер (joblib). Возвращает True если сохранён.

    Если эмбеддер не персистится (ST-модель предобучена), СТАРЫЙ артефакт удаляется:
    иначе после реиндекса на e5 запрос грузил бы протухший TF-IDF-базис и искал в чужом
    пространстве.
    """
    if isinstance(embedder, TfidfEmbedder) and embedder.fitted:
        import joblib

        joblib.dump(embedder, str(path))
        return True
    if os.path.exists(str(path)):
        os.remove(str(path))
    return False


def load_embedder(path):
    """Загрузить персистнутый эмбеддер; None если файла нет/ошибка чтения."""
    if not os.path.exists(str(path)):
        return None
    try:
        import joblib

        return joblib.load(str(path))
    except Exception as e:  # noqa: BLE001 — легаси/битый артефакт → тихий fallback на ре-фит
        print(f"  ⚠️ load_embedder failed ({e}); fallback to re-fit", file=sys.stderr)
        return None


if __name__ == "__main__":
    print("=" * 60)
    print("  RAG — Embedding provider test")
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
