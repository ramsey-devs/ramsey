.PHONY: tag, tests, lints, docs, format, examples

PKG_VERSION=`hatch version`

tag:
	 git tag -a v${PKG_VERSION} -m v${PKG_VERSION}
	 git push --tag

tests:
	hatch run test:tests

lints:
	hatch run test:lints

format:
	hatch run test:format

docs:
	cd docs && make html

examples:
	hatch run test:examples
