-- Учёт вызовов модели (шаг 8): задача, модель, токены, стоимость, длительность.
CREATE TABLE IF NOT EXISTS llm_calls (
    id                BIGSERIAL PRIMARY KEY,
    job_id            TEXT REFERENCES jobs(id),
    request_id        TEXT,
    agent             TEXT NOT NULL,
    model             TEXT NOT NULL,
    prompt_tokens     INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    cost              NUMERIC(14, 6) NOT NULL DEFAULT 0,
    duration_ms       INTEGER NOT NULL DEFAULT 0,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_llm_calls_job ON llm_calls(job_id);
CREATE INDEX IF NOT EXISTS idx_llm_calls_model ON llm_calls(model);
