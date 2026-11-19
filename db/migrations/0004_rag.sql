-- Шаг 6: поиск по смыслу и по словам по фрагментам документов.
--
-- embedding — вектор фрагмента; размерность фиксирована миграцией,
-- смена модели эмбеддингов = новая миграция с пересозданием колонки.
-- tsv — нормализованная форма текста для полнотекстового поиска,
-- поддерживается триггером, по ней — инвертированный GIN-индекс.

CREATE EXTENSION IF NOT EXISTS vector;

ALTER TABLE chunks
    ADD COLUMN IF NOT EXISTS embedding vector(384),
    ADD COLUMN IF NOT EXISTS tsv tsvector;

-- Фрагменты, вставленные до этой миграции: досчитать tsv сразу,
-- векторы им позже досчитает команда переиндексации (app/rag/reindex.py).
UPDATE chunks SET tsv = to_tsvector('russian', text) WHERE tsv IS NULL;

CREATE OR REPLACE FUNCTION chunks_set_tsv() RETURNS trigger AS $$
BEGIN
    NEW.tsv := to_tsvector('russian', NEW.text);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS chunks_tsv_trigger ON chunks;
CREATE TRIGGER chunks_tsv_trigger
    BEFORE INSERT OR UPDATE OF text ON chunks
    FOR EACH ROW EXECUTE FUNCTION chunks_set_tsv();

-- Приближённый индекс по векторам; оператор cosine задаёт метрику
-- один раз — индекс строится под неё.
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS chunks_tsv_gin_idx
    ON chunks USING gin (tsv);
