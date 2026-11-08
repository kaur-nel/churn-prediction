.PHONY: lint test data split features imbalance compare tune

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

imbalance:
	python -m src.imbalance

compare:
	python -m src.compare

tune:
	python -m src.tune