"""Sphinx documentation configuration for PECOS quantum error correction framework.

This module configures Sphinx for generating the PECOS documentation, including
API reference, user guides, and examples for the quantum error correction library.
"""

#  =========================================================================  #
#   Copyright 2023 The PECOS Developers
#   Copyright 2018 National Technology & Engineering Solutions of Sandia,
#   LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS,
#   the U.S. Government retains certain rights in this software.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#  =========================================================================  #

# Configuration file for the Sphinx documentation builder.
# --------------------------------------------------------
# -- Path setup --------------------------------------------------------------
# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import sys
from importlib import metadata
from pathlib import Path

sys.path.insert(0, str(Path("../src").resolve()))

# -- Project information -----------------------------------------------------

project = "PECOS"
copyright = (
    "2018-2025, The PECOS Developers. "
    "\xa9 Copyright 2014-2018, National Technology & Engineering Solutions of Sandia, LLC (NTESS)"
)
author = "The PECOS Developers"

# The full version, including alpha/beta/rc tags
try:
    release = metadata.version("quantum-pecos")
except metadata.PackageNotFoundError:
    # If the package isn't installed, fall back to hardcoded version
    release = "0.6.0.dev8"
# The short version
version = ".".join(release.split(".")[:3])


# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
needs_sphinx = "7.1"

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

# TODO: Go through these and confirm they are being used

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    # "sphinx.ext.linkcode",
    "sphinx.ext.doctest",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "sphinx.ext.ifconfig",
    "sphinx.ext.extlinks",
    # 'sphinx.ext.autosectionlabel',
    "sphinx_copybutton",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
source_suffix = ".rst"  # ['.rst', '.md']

# The encoding of source files.
source_encoding = "utf-8-sig"

# The master toctree document.
master_doc = "index"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path .
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "sphinx", "README.md"]

# Highlight the background of code blocks. Default is "python3"
highlight_language = "python3"

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# -- Extension settings  -----------------------------------------------------

# Sphinx-copybutton - add copy button to code blocks
# https://sphinx-copybutton.readthedocs.io/en/latest/index.html
# strip the '>>>' and '...' prompt/continuation prefixes.
copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = "sphinx_rtd_theme"
html_theme = "pydata_sphinx_theme"

html_logo = "images/pecos_large_logo_white.png"

github_root = "https://github.com/PECOS-packages/PECOS"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options = {
    "logo": {
        "image_light": "images/pecos_large_logo.png",
        "image_dark": "images/pecos_large_logo.png",
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": github_root,
            "icon": "fa-brands fa-github",
        },
    ],
    "navbar_end": [
        "theme-switcher",
        # TODO: Enable switcher: https://pydata-sphinx-theme.readthedocs.io/en/stable/user_guide/version-dropdown.html
        # "version-switcher",
        "navbar-icon-links",
    ],
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# The default sidebars (for documents that don't match any pattern) are
# defined by theme itself.  Builtin themes are using these templates by
# default: ``['localtoc.html', 'relations.html', 'sourcelink.html',
# 'searchbox.html']``.
#
# html_sidebars = {}


# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "PECOSdoc"


# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    "papersize": "letterpaper",
    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',
    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, "PECOS.tex", "PECOS Documentation", author, "manual"),
]


# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, "pecos", "PECOS Documentation", [author], 1),
]


# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "PECOS",
        "PECOS Documentation",
        author,
        "PECOS",
        "One line description of project.",
        "Miscellaneous",
    ),
]

# -- External links code:

extlinks = {
    "arxiv": ("https://arxiv.org/abs/%s", "arXiv:%s"),
}

# -- Extension configuration -------------------------------------------------

# -- Options for todo extension ----------------------------------------------

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True
todo_emit_warnings = True
todo_link_only = True

# -- Options for autosummary extension ----------------------------------------------
# Generate subpages for reference docs automatically
autosummary_generate = True

# -- Options for autodoc extension ----------------------------------------------
# Mock modules that are not available during doc build
autodoc_mock_imports = ["cupy", "pecos.slr.std", "pecos.slr.slr"]

# Skip problematic modules
autosummary_mock_imports = [
    "pecos.slr.std",
    "pecos.slr.slr",
    "pecos.simulators.cuquantum_old",
    "pecos.simulators.custatevec",
    "pecos.simulators.cysparsesim",
    "pecos.simulators.cysparsesim_col",
    "pecos.simulators.cysparsesim_row",
]
