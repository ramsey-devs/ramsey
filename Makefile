PKG_VERSION=`python setup.py --version`

tag:
	git tag -a "v$(PKG_VERSION)" -m "v$(PKG_VERSION)"
	git push --tag
