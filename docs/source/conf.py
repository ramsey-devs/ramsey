import glob
import os

project = "ramsey"
copyright = "2021, Simon Dirmeier"
author = "Simon Dirmeier"
release = "0.0.1"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "sphinx_gallery.gen_gallery",
    "nbsphinx",
    "IPython.sphinxext.ipython_console_highlighting",
]


sphinx_gallery_conf = {
    "examples_dirs": ["../../examples"],
    "gallery_dirs": ["examples"],
    "filename_pattern": "/plot_",
    "ignore_pattern": "(__init__)",
    "min_reported_time": 1,
}

html_static_path = ["_static"]
templates_path = ["_templates"]

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

html_theme = "sphinx_rtd_theme"

intersphinx_mapping = {
    "haiku": ("https://dm-haiku.readthedocs.io/en/latest/", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
    "numpyro": ("https://num.pyro.ai/en/stable/", None),
}
