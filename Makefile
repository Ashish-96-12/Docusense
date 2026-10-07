.PHONY: install api ui test eval docker

install:
	pip install -r requirements-dev.txt

api:
	uvicorn docusense.api:app --reload --port 8000

ui:
	streamlit run ui/streamlit_app.py

test:
	pytest

eval:
	python -m eval.run_eval

docker:
	docker compose up --build
