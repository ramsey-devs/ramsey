PKG_VERSION=`python setup.py --version`

tag:
	git tag -a $(PKG_VERSION) -m $(PKG_VERSION)
	git push --tag
