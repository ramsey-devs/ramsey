from datetime import date

project = "Ramsey"
copyright = f"{date.today().year}, the Ramsey developers"
author = "the Ramsey developers"

extensions = [
    "nbsphinx",
    "sphinx.ext.autodoc",
    "sphinx_autodoc_typehints",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    'sphinxcontrib.bibtex',
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx_math_dollar",
    "IPython.sphinxext.ipython_console_highlighting",
    "sphinx_design",
]

templates_path = ["_templates"]
html_static_path = ["_static"]
html_css_files = ["theme.css"]

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
    "examples/*py",
]

autosummary_generate = True
autodoc_typehints = 'none'
typehints_fully_qualified = True
always_document_param_types = True

html_theme = "sphinx_book_theme"
html_theme_options = {
    "repository_url": "https://github.com/ramsey-devs/ramsey",
    "use_repository_button": True,
    "use_download_button": False,
    "use_fullscreen_button": False,
    "launch_buttons": {"colab_url": "https://colab.research.google.com"},
}
html_title = "Ramsey"


def skip(app, what, name, obj, would_skip, options):
    if name == "__init__":
        return True
    return would_skip


def setup(app):
    app.connect("autodoc-skip-member", skip)


bibtex_bibfiles = ['references.bib']
bibtex_default_style = 'plain'
bibtex_reference_style = 'label'
