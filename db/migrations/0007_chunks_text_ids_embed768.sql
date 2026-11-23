-- Смена модели эмбеддингов на embeddinggemma-300m (768 вместо 384)
-- и унификация схемы фрагментов под контракт поиска и генерации.
--
-- documents: kind (тип документа) и chunks_count (счётчик фрагментов).
-- chunks: строковые id (как у документов), doc_id/ordinal вместо
-- document_id/seq, метаданные собраны в JSONB-колонку, флаг suspicious,
-- векторная колонка пересоздана под новую размерность.
-- Существующие векторы теряют смысл при смене размерности и сбрасываются:
-- их досчитает команда переиндексации (make reindex).

ALTER TABLE documents
    ADD COLUMN IF NOT EXISTS kind TEXT NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS chunks_count INTEGER NOT NULL DEFAULT 0;

ALTER TABLE chunks RENAME COLUMN document_id TO doc_id;
ALTER TABLE chunks RENAME COLUMN seq TO ordinal;

ALTER INDEX IF EXISTS chunks_document_id_idx RENAME TO idx_chunks_doc;

-- id: BIGSERIAL → TEXT; sequence и default больше не нужны.
ALTER TABLE chunks ALTER COLUMN id DROP DEFAULT;
ALTER TABLE chunks ALTER COLUMN id TYPE TEXT;
DROP SEQUENCE IF EXISTS chunks_id_seq;

ALTER TABLE chunks ADD COLUMN IF NOT EXISTS metadata JSONB NOT NULL DEFAULT '{}';
UPDATE chunks SET metadata = jsonb_strip_nulls(jsonb_build_object(
    'doc_id', doc_id,
    'page', page,
    'section', section,
    'articul', article,
    'brand', brand
));
ALTER TABLE chunks
    DROP COLUMN IF EXISTS page,
    DROP COLUMN IF EXISTS section,
    DROP COLUMN IF EXISTS article,
    DROP COLUMN IF EXISTS brand;

ALTER TABLE chunks
    ADD COLUMN IF NOT EXISTS suspicious BOOLEAN NOT NULL DEFAULT FALSE;

ALTER TABLE chunks
    DROP CONSTRAINT IF EXISTS chunks_document_id_seq_key;
ALTER TABLE chunks
    ADD CONSTRAINT chunks_doc_id_ordinal_key UNIQUE (doc_id, ordinal);

-- Смена размерности вектора: старые векторы несовместимы, колонка
-- пересоздаётся, индекс строится заново под ту же метрику (cosine).
DROP INDEX IF EXISTS chunks_embedding_hnsw_idx;
ALTER TABLE chunks DROP COLUMN IF EXISTS embedding;
ALTER TABLE chunks ADD COLUMN embedding vector(768);
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);
