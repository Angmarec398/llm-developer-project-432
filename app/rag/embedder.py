"""Модуль эмбеддингов: локальная модель на процессоре (sentence-transformers).

Модель multilingual-e5-small обучена с задачными префиксами: документ
кодируется с префиксом "passage: ", запрос — "query: ". Без префиксов
близость релевантных пар проседает так, что нормальный порог отсечения
выметает верные результаты — префиксы часть контракта модели, берутся
из её карточки, а не из головы. Модель загружается один раз на процесс
и кэшируется на диске после первой загрузки; векторы считаются
пачкой — иначе индексация набора документов растянется на часы.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from app.core.config import settings

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

DOCUMENT_PREFIX = "passage: "
QUERY_PREFIX = "query: "

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Загрузить модель один раз на процесс (лениво, при первом вызове).

    Импорт внутри функции: тяжёлая зависимость тянется только тогда,
    когда эмбеддинги реально считаются — тесты и --help её не трогают.
    """
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(settings.embedding_model)
    return _model


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Закодировать фрагменты документа пачкой (префикс passage).

    Тексты кодируются порциями по embedding_batch_size; порядок
    векторов совпадает с порядком текстов — репозиторий пишет
    их пачкой попарно.
    """
    return _embed([DOCUMENT_PREFIX + text for text in texts])


def embed_query(text: str) -> list[float]:
    """Закодировать поисковый запрос (префикс query), вернуть один вектор."""
    return _embed([QUERY_PREFIX + text])[0]


def _embed(prefixed_texts: list[str]) -> list[list[float]]:
    """Закодировать пачку подготовленных текстов, вернуть список векторов.

    Векторы нормализуются до единичной длины — косинусная близость
    становится устойчивой к масштабу. Размер каждого вектора сверяется
    с embedding_dimension: рассинхрон модели и схемы БД ломался бы
    молча, на стадии поиска; проверка стоит сразу у источника.
    """
    if not prefixed_texts:
        return []
    model = _get_model()
    vectors: list[list[float]] = []
    batch_size = settings.embedding_batch_size
    for start in range(0, len(prefixed_texts), batch_size):
        batch = prefixed_texts[start: start + batch_size]
        encoded = model.encode(batch, normalize_embeddings=True)
        vectors.extend(vector.tolist() for vector in encoded)
    for vector in vectors:
        if len(vector) != settings.embedding_dimension:
            raise ValueError(
                f"размерность вектора {len(vector)} не совпадает "
                f"с embedding_dimension={settings.embedding_dimension}"
            )
    return vectors
