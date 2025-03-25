.PHONY: setup clean test lint coverage docs build docker run docker-build docker-run help

PYTHON = python
PIP = pip
PYTEST = pytest
DOCKER = docker
DOCKER_COMPOSE = docker-compose

help:
	@echo "Available commands:"
	@echo "  setup        Install dependencies"
	@echo "  clean        Clean build files"
	@echo "  test         Run tests"
	@echo "  lint         Run code linting"
	@echo "  coverage     Run test coverage"
	@echo "  docs         Build documentation"
	@echo "  build        Build Python package"
	@echo "  docker       Build Docker image and run container"
	@echo "  docker-build Build Docker image"
	@echo "  docker-run   Run Docker container"

setup:
	$(PIP) install -r requirements.txt
	$(PIP) install -e .[dev]

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name "*.pyc" -delete

test:
	$(PYTEST) tests/

lint:
	black .
	isort .
	flake8 agentic_rag tests

coverage:
	$(PYTEST) --cov=agentic_rag --cov-report=term --cov-report=html tests/

docs:
	cd docs && make html

build:
	$(PYTHON) setup.py sdist bdist_wheel

docker-build:
	$(DOCKER) build -t agentic-rag .

docker-run:
	$(DOCKER) run -p 7860:7860 -v "$(PWD)/data:/app/data" -v "$(PWD)/.env:/app/.env" agentic-rag

docker: docker-build docker-run
