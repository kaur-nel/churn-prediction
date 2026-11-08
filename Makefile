.PHONY: lint test data split features imbalance compare tune final business evaluate train

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

final:
	python -m src.final_model

business:
	python -m src.business

evaluate:
	python -m src.evaluate

train: data split final