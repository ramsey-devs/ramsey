.PHONY: dist

dist:
	python -m build

upload:
    python3 -m twine upload  dist/