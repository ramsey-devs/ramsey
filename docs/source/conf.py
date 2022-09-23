import glob
import os
from datetime import date

project = "ramsey"
copyright = f"{date.today().year}, the ramsey developers"
author = "ramsey developers"
release = "0.0.2"

extensions = [
    "nbsphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "sphinx_gallery.gen_gallery",
    "sphinx_math_dollar",
    "IPython.sphinxext.ipython_console_highlighting",
]

sphinx_gallery_conf = {
    "examples_dirs": ["../../examples"],
    "gallery_dirs": ["examples"],
    "filename_pattern": "/plot_",
    "ignore_pattern": "(__init__)",
    "min_reported_time": 1,
}

templates_path = ["_templates"]
html_static_path = ["_static"]

autodoc_default_options = {
    "member-order": "bysource",
    "special-members": True,
    "exclude-members": "__repr__, __str__, __weakref__",
}

exclude_patterns = [
    "_build",
    "build",
    "Thumbs.db",
    ".DS_Store",
    "notebooks/.ipynb_checkpoints",
    "examples/*ipynb",
    "examples/*py"
]

intersphinx_mapping = {
    "haiku": ("https://dm-haiku.readthedocs.io/en/latest/", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
    "numpyro": ("https://num.pyro.ai/en/stable/", None),
}

html_theme = "sphinx_book_theme"

html_theme_options = {
    "repository_url": "https://github.com/ramsey-devs/ramsey",
    "use_repository_button": True,
    "use_download_button": False,
}

html_title = "ramsey"
