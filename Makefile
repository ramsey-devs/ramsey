PKG_VERSION=`python setup.py --version`

.PHONY : tag

tag:
	git tag -a v${PKG_VERSION} -m "v${PKG_VERSION}"
	git push --tag
