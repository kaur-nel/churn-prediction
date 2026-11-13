.PHONY: lint test pin data split features imbalance compare tune final business evaluate explain drift predict report health train

lint:
	ruff check .

test:
	pytest -q

pin:
	python -m src.pin_requirements

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

explain:
	python -m src.explain

drift:
	python -m src.drift

predict:
	python -m src.predict --input $(INPUT)

report:
	python -m src.report

health:
	python -m src.health

train:
	python -m src.train