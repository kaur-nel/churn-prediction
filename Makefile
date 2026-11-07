.PHONY: lint test data split features

lint:
	ruff check .

test:
	pytest -q

data:
	python -m src.data

split:
	python -m src.split

features:
	python -m src.features