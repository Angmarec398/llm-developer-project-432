.PHONY: help infra api worker migrate lint temporal temporal-show temporal-worker eval eval-full

help:
	@echo "make infra           - поднять postgres (docker compose)"
	@echo "make temporal        - добавить temporal server (:7233) + UI (:8233) + CLI"
	@echo "make temporal-show ID=card-abc123 - история workflow (temporal CLI из docker)"
	@echo "make api             - FastAPI на :8000"
	@echo "make worker          - temporal-воркер (workflow + activities)"
	@echo "make migrate         - применить миграции БД"
	@echo "make lint            - ruff check app tests"
	@echo "make lint-fix        - корректировка ошибок линтера"


infra:
	docker compose up -d

temporal:
	docker compose --profile temporal up -d

temporal-show:
	docker compose exec temporal-admin-tools temporal workflow show --workflow-id $(ID)

api:
	uvicorn app.main:app --reload --port 8000

worker:
	python -m app.temporal_worker

temporal-worker:
	python -m app.temporal_worker

migrate:
	python -m app.core.migrate

lint:
	ruff check app tests

lint-fix:
	ruff check app tests --fix

reindex:
	python -m app.rag.reindex

eval:
	python -m evals

eval-full:
	python -m evals --full