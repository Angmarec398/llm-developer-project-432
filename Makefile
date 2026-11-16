.PHONY: help infra api worker temporal temporal-show temporal-worker

help:
	@echo "make infra           - поднять postgres + rabbitmq (docker compose)"
	@echo "make temporal        - добавить temporal server (:7233) + UI (:8233) + CLI"
	@echo "make temporal-show ID=card-abc123 - история workflow (temporal CLI из docker)"
	@echo "make api             - FastAPI на :8000"
	@echo "make worker          - celery-воркер (очереди card.generate + card.critique)"
	@echo "make temporal-worker - temporal-воркер (workflow + activities)"

infra:
	docker compose up -d

temporal:
	docker compose --profile temporal up -d

temporal-show:
	docker compose exec temporal-admin-tools temporal workflow show --workflow-id $(ID)

api:
	uvicorn app.main:app --reload --port 8000

worker:
	celery -A app.workers.celery_app worker -Q card.generate,card.critique --pool=solo -l info

temporal-worker:
	python -m app.temporal_worker

migrate:
	python -m app.core.migrate
