import datetime

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/main/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#

from pyprojroot import here

import os
import sys

sys.path.insert(0, os.path.abspath("."))

import autogalaxy

# -- Project information -----------------------------------------------------

year = datetime.date.today().year
project = "PyAutoGalaxy"
copyright = "2025, James Nightingale, Richard Hayes"
author = "James Nightingale, Richard Hayes"

# The full version, including alpha/beta/rc tags
release = autogalaxy.__version__
master_doc = "index"


# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "numpydoc",
    "sphinx_autodoc_typehints",
    # external
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_inline_tabs",
]

set_type_checking_flag = True  # Enable 'expensive' imports for sphinx_autodoc_typehints
add_module_names = False  # Remove namespaces from class/method signatures

templates_path = ["_templates"]

# -- Options for extlinks ----------------------------------------------------

extlinks = {"pypi": ("https://pypi.org/project/%s/", "")}

# -- Options for intersphinx -------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "sphinx": ("https://www.sphinx-doc.org/en/main", None),
}

# -- Options for TODOs -------------------------------------------------------

todo_include_todos = True

# -- Options for Markdown files ----------------------------------------------

myst_enable_extensions = ["colon_fence", "deflist"]
myst_heading_anchors = 3

autosummary_generate = True
autosummary_imported_members = True
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
autodoc_class_signature = "separated"
autoclass_content = "init"

numpydoc_show_class_members = False
numpydoc_show_inherited_class_members = False
numpydoc_class_members_toctree = True

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "CODE_OF_CONDUCT.md",
    "CONTRIBUTING.md",
    "CITATIONS.rst",
    "README.rst",
    "README.md",
]


# -- Options for HTML output -------------------------------------------------

html_theme = "furo"
html_title = "PyAutoGalaxy"
html_short_title = "PyAutoGalaxy"
html_permalinks_icon = "<span>#</span>"
html_last_updated_fmt = "%b %d, %Y"

html_show_sourcelink = False
html_show_sphinx = True
html_show_copyright = True

# pygments_style = "sphinx"
# pygments_dark_style = "monokai"
add_function_parentheses = False

html_context = {
    "menu_links_name": "Repository",
    "doc_path": "docs",
    "github_project": "pyautogalaxy",
    "github_repo": "pyautogalaxy",
    "github_version": "master",
}
language = "en"

from sphinx.builders.html import StandaloneHTMLBuilder

StandaloneHTMLBuilder.supported_image_types = ["image/gif", "image/png", "image/jpeg"]

typehints_fully_qualified = False
always_document_param_types = False
typehints_document_rtype = False