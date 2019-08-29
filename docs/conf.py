# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

import tdub
from pallets_sphinx_themes import ProjectLink, get_version

# -- Project information -----------------------------------------------------

project = "tdub"
copyright = "2019, Doug Davis"
author = "Doug Davis"

version = tdub.__version__
release = tdub.__version__

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinxcontrib.programoutput",
    "pallets_sphinx_themes",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# syntax highlighting style
pygments_style = "default"

# -- Options for HTML output -------------------------------------------------


html_theme = "click"
html_theme_options = {"index_sidebar_logo": False}

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = "sphinx_rtd_theme"

#html_theme_options = {
#    "canonical_url": "https://github.com/douglasdavis/tdub",
#    "display_version": True,
#    "collapse_navigation": True,
#    "sticky_navigation": True,
#    "navigation_depth": 4,
#}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

intersphinx_mapping = {
    "python": ("http://docs.python.org", None),
    "numpy": ("http://docs.scipy.org/doc/numpy", None),
    "dask": ("https://docs.dask.org/en/latest", None),
    "uproot": ("https://uproot.readthedocs.io/en/latest", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
}
