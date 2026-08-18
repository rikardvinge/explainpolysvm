# Configuration file for the Sphinx documentation builder.
#
# For the full list of options, see
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from importlib.metadata import PackageNotFoundError, version as package_version

# Make the package importable without installing it, so that autodoc works both
# in a checkout and in an environment where ExplainPolySVM is installed.
sys.path.insert(0, os.path.abspath('../src'))

project = 'ExplainPolySVM'
copyright = '2025, Rikard Vinge'
author = 'Rikard Vinge'

try:
    release = package_version('explainpolysvm')
except PackageNotFoundError:
    release = 'unknown'
version = release

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
]

# The docstrings in the package are written in the numpydoc style.
napoleon_numpy_docstring = True
napoleon_google_docstring = False

autodoc_member_order = 'bysource'
autodoc_typehints = 'description'
autodoc_default_options = {
    'members': True,
    'undoc-members': False,
    'show-inheritance': True,
}

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'sklearn': ('https://scikit-learn.org/stable', None),
    'matplotlib': ('https://matplotlib.org/stable', None),
}

templates_path = ['_templates']
exclude_patterns = ['_build']

html_theme = 'furo'
html_static_path = []
