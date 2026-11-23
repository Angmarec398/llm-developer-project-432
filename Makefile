.PHONY: help install setup lint lint-fix test infra api worker migrate temporal temporal-show temporal-worker eval eval-full reindex

help:
	@echo "make install         - установить зависимости из uv.lock"
	@echo "make setup           - install + миграции"
	@echo "make lint            - ruff check app tests"
	@echo "make lint-fix        - ruff check --fix app tests"
	@echo "make test            - pytest"
	@echo "make infra           - поднять postgres + temporal (docker compose)"
	@echo "make temporal-show ID=card-abc123 - история workflow (temporal CLI из docker)"
	@echo "make api             - FastAPI на :8000"
	@echo "make worker          - temporal-воркер (workflow + activities)"
	@echo "make migrate         - применить миграции БД"
	@echo "make reindex         - досчитать embeddings для чанков без векторов"
	@echo "make eval / eval-full - прогон метрик качества"

install:
	uv sync --frozen

setup: install migrate

lint:
	uv run ruff check app tests

lint-fix:
	uv run ruff check app tests --fix

test:
	uv run pytest -q

infra:
	docker compose up -d

temporal:
	docker compose up -d temporal temporal-ui temporal-admin-tools

temporal-show:
	docker compose exec temporal-admin-tools temporal workflow show --workflow-id $(ID)

api:
	uv run uvicorn app.main:app --reload --port 8000

worker:
	uv run python -m app.temporal_worker

temporal-worker:
	uv run python -m app.temporal_worker

migrate:
	uv run python -m app.core.migrate

reindex:
	uv run python -m app.rag.reindex

eval:
	uv run python -m evals

eval-full:
	uv run python -m evals --full
