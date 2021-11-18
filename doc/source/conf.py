# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("../"))
sys.path.insert(0, os.path.abspath(".../"))


# -- Project information -----------------------------------------------------

import disba

project = "disba"
copyright = "2021, Keurfon Luu"
author = "Keurfon Luu"

version = disba.__version__
release = version


# -- General configuration ---------------------------------------------------

master_doc = "index"

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.napoleon",
    "sphinxarg.ext",
    # "sphinxcontrib.bibtex",
    # "sphinx_gallery.gen_gallery",
]

# Sphinx Gallery settings
from sphinx_gallery.sorting import FileNameSortKey

sphinx_gallery_conf = {
    # "examples_dirs": [
    #     "../../examples/",
    # ],
    # "gallery_dirs": [
    #     "examples/",
    # ],
    "filename_pattern": r"\.py",
    "download_all_examples": False,
    "within_subsection_order": FileNameSortKey,
    "backreferences_dir": None,
    "doc_module": "disba",
    "image_scrapers": (
        "matplotlib",
        # "pyvista",
    ),
    "first_notebook_cell": (
        "%matplotlib inline\n"
    ),
}

# # PyVista settings
# import pyvista

# pyvista.BUILDING_GALLERY = True

# Napoleon settings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
# napoleon_use_admonition_for_examples = False
# napoleon_use_admonition_for_notes = False
# napoleon_use_admonition_for_references = False
# napoleon_use_ivar = True
# napoleon_use_param = True
# napoleon_use_rtype = True

# Numfig settings
numfig = False
numfig_format = {
    "figure": "Figure %s",
}

# Add any paths that contain templates here, relative to this directory.
templates_path = [
    "_templates",
]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

# Class documentation
autoclass_content = "both"


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
import pydata_sphinx_theme

html_theme = "pydata_sphinx_theme"
html_theme_path = [
    "_themes",
]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = [
#     "_static",
# ]