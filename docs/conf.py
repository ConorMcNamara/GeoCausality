import os
import sys

sys.path.insert(0, os.path.abspath("../"))

# Configuration file for the Sphinx documentation builder.

# -- Project information

project = "GeoCausality"
copyright = "2025, Conor McNamara"
author = "Conor McNamara"

release = "0.10.0"
version = "0.10.0"

# -- General configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinxcontrib.bibtex",
]

# sphinxcontrib-bibtex 2.x: the bib file is declared here and the
# ``.. bibliography::`` directive takes no file argument.
bibtex_bibfiles = ["references.bib"]

latex_elements = {"preamble": r"\usepackage{mathtools}"}

# -- Autodoc / autosummary
# The package and its runtime dependencies are installed for the docs build
# (``pip install -e .``), so autodoc imports the real modules — no mocking, which
# would break the ``np.ndarray | None`` annotations.

autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}

napoleon_google_docstring = False
napoleon_numpy_docstring = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]

# -- Options for HTML output

html_theme = "sphinx_book_theme"
html_title = "GeoCausality"
html_theme_options = {
    "repository_url": "https://github.com/ConorMcNamara/GeoCausality",
    "use_repository_button": True,
    "use_issues_button": True,
}

# -- Options for EPUB output
epub_show_urls = "footnote"
