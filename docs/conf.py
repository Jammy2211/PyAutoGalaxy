import datetime
from typing import Any, Dict
# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#

from pyprojroot import here

workspace_path = str(here())

import os
import sys
import autogalaxy

sys.path.insert(0, os.path.abspath("."))

# -- Project information -----------------------------------------------------

year = datetime.date.today().year
project = "PyAutoGalaxy"
copyright = "2020, James Nightingale, Richard Hayes"
author = "James Nightingale, Richard Hayes"

# The full version, including alpha/beta/rc tags
release = "2022.05.02.1"
master_doc = "index"


# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "numpydoc",
    # External stuff
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_inline_tabs",
]
templates_path = ["_templates"]

# -- Options for extlinks ----------------------------------------------------

extlinks = {
    "pypi": ("https://pypi.org/project/%s/", ""),
}

# -- Options for intersphinx -------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master", None),
}

# -- Options for TODOs -------------------------------------------------------

todo_include_todos = True

# -- Options for Markdown files ----------------------------------------------

myst_enable_extensions = [
    "colon_fence",
    "deflist",
]
myst_heading_anchors = 3

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_default_flags = ["members"]

numpydoc_show_class_members = False
numpydoc_show_inherited_class_members = False
numpydoc_class_members_toctree = False

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

html_theme = "furo"
html_title = "PyAutoGalaxy"
html_short_title = "PyAutoGalaxy"
html_permalinks_icon = '<span>#</span>'
html_last_updated_fmt = "%b %d, %Y"

html_show_sourcelink = False
html_show_sphinx = True
html_show_copyright = True

pygments_style = "sphinx"
pygments_dark_style = "monokai"
add_function_parentheses = False

html_context = {
    "menu_links_name": "Repository",
    "doc_path": "docs",
    "github_project": "pyautogalaxy",
    "github_repo": "pyautogalaxy",
    "github_version": "master",
}
language = "en"

html_static_path = ["_static"]
html_css_files = ["pied-piper-admonition.css"]

html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#7C4DFF",
        "color-brand-content": "#7C4DFF",
    },
}

# html_theme_options: Dict[str, Any] = {
#     "footer_icons": [
#         {
#             "name": "GitHub",
#             "url": "https://github.com/pradyunsg/furo",
#             "html": """
#                 <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
#                     <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
#                 </svg>
#             """,
#             "class": "",
#         },
#     ],
# }

if "READTHEDOCS" in os.environ:
    html_theme_options["announcement"] = (
        "This documentation is hosted on Read the Docs only for testing. Please use "
        "<a href='https://pradyunsg.me/furo/'>the main documentation</a> instead."
    )


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ["_static"]

# -- IMAGES --


from sphinx.builders.html import StandaloneHTMLBuilder

StandaloneHTMLBuilder.supported_image_types = ["image/gif", "image/png", "image/jpeg"]

