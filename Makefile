.PHONY: ingest run serve eval test lint format all clean

# ─── Pipeline ──────────────────────────────────────────────────────────────

ingest:
	python main.py ingest --pdf files/cie11.pdf

run:
	python main.py run --profile evals/profiles/anxiety_basic.json

serve:
	python main.py serve

eval:
	python main.py eval --suite evals/suites/standard.yaml

# ─── Quality gates ─────────────────────────────────────────────────────────

test:
	PYTHONPATH=. .venv/bin/python3 -m pytest tests/ -v --tb=short

lint:
	.venv/bin/python3 -m ruff check .
	.venv/bin/python3 -m mypy . --ignore-missing-imports

format:
	.venv/bin/python3 -m black .
	.venv/bin/python3 -m ruff check --fix .

all: lint format test

# ─── Utilities ─────────────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .mypy_cache .ruff_cache .pytest_cache htmlcov .coverage
