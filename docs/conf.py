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
    "sphinx_immaterial",
    "sphinx.ext.napoleon",
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

html_theme = "sphinx_immaterial"

html_theme_options = {
    "site_url": "https://ramsey.readthedocs.io",
    "repo_url": "https://github.com/ramsey-devs/ramsey",
    "repo_name": "ramsey",
    "extra_navbar": "",
    "launch_buttons": {"colab_url": "https://colab.research.google.com"},
    "icon": {
        "repo": "fontawesome/brands/github",
        "edit": "material/file-edit-outline",
    },
    "globaltoc_collapse": True,
    "features": [
        "navigation.expand",
        "navigation.sections",
        "navigation.top",
        "search.share",
        "toc.follow",
        "toc.sticky",
        "content.tabs.link",
        "announce.dismiss",
    ],
    "palette": [
        {
            "media": "(prefers-color-scheme: light)",
            "scheme": "default",

        },
    ],
    "toc_title_is_page_title": True,
    'globaltoc_depth': 1
}

html_title = "Ramsey 🚀"
