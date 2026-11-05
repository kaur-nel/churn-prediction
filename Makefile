.PHONY: lint test data

lint:
	ruff check .

test:
	pytest -q

data:
	python -m src.data