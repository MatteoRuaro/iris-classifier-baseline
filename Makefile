.PHONY: install format lint test train evaluate predict


install:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt && pre-commit install


format:
	black .


lint:
	flake8 src tests


test:
	PYTHONPATH=src pytest


train:
	PYTHONPATH=src python scripts/train.py --config config/config.yaml


evaluate:
	PYTHONPATH=src python scripts/evaluate.py --config config/config.yaml

predict:
	python scripts/predict.py --inputs 5.1 3.5 1.4 0.2