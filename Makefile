export SHELL := /bin/bash

install-test:
	pip install -e '.[test]'

install-dev:
	pip install -e '.[dev]'

install-docs:
	pip install -e '.[stats,docs]'

install-all:
	pip install -e '.[build]'

check-deps:
	python3 ci/check_extras_consistency.py

check-deps-wheel:
	python3 ci/check_extras_consistency.py --check-wheel

fix-deps:
	python3 ci/check_extras_consistency.py --fix

test:
	pytest -n auto --cov=seaborn --cov=tests --cov-config=setup.cfg tests

lint:
	flake8 seaborn/ tests/

typecheck:
	mypy --follow-imports=skip seaborn/_core seaborn/_marks seaborn/_stats
