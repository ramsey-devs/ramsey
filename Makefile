.PHONY: tests, lints, docs, format

tests:
	uv run pytest

lints:
	uv run ruff check ramsey examples

format:
	uv run ruff check --select I --fix ramsey examples
	uv run ruff format ramsey examples

docs:
	cd docs && make html
