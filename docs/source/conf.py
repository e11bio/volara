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
import os
import sys

# Set the path to the Volara package
sys.path.insert(0, os.path.abspath("../.."))


# -- Project information -----------------------------------------------------

project = "Volara"
copyright = (
    "2024, William Patton, Arlo Sheridan, Jeff Rhoades, Jakob Troidl, Andrew Champion"
)
author = "William Patton, Arlo Sheridan, Jeff Rhoades, Jakob Troidl, Andrew Champion"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "jupyter_sphinx",
    "sphinxcontrib.autodoc_pydantic",  # auto document pydantic models
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx_click",  # auto document cli
    "nbsphinx",
    "myst_parser",  # include md files in rst files
]

# The example scripts are converted to notebooks with jupytext before the build
# (see .github/workflows/docs.yaml), so nbsphinx only reads `.ipynb` files. Do not
# register `.py` as a notebook format here: that makes every Python file under the
# source directory a document, `conf.py` included.
nbsphinx_kernel_name = "volara_env"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    # Output of examples/cremi/cremi_report.py, copied in with the examples; not a
    # page of the site.
    "examples/cremi/RESOURCE_REPORT.md",
]

# The task models have `Coordinate` fields (`block_size`, `context`), which have no
# JSON schema. Render those fields in the schema box as autodoc_pydantic already
# does, without the per-model warning.
autodoc_pydantic_model_show_json_error_strategy = "coerce"

# sphinx-click does `from sphinx.ext.autodoc import mock` and calls it. Under
# Sphinx 9 that name is the `sphinx.ext.autodoc.mock` module once any extension
# has imported it, so the `click` directive fails with "'module' object is not
# callable" and the CLI page is empty. Hand sphinx-click the function. Remove once
# https://github.com/click-contrib/sphinx-click/issues/159 is fixed.
import sphinx_click.ext  # noqa: E402
from sphinx.ext.autodoc.mock import mock as _autodoc_mock  # noqa: E402

if not callable(sphinx_click.ext.mock):
    sphinx_click.ext.mock = _autodoc_mock

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "show_nav_level": 3,  # controls how many levels deep navigation is shown
    "navigation_depth": 3,  # alternative option for some versions
    "collapse_navigation": True,  # Keep navigation expanded
}
html_sidebars = {
    "**": [],
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = [
    "css/custom.css",
]
