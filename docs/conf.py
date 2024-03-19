# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sst1mpipe

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'sst1mpipe'
copyright = '2024, SST-1M collaboration'
author = 'SST-1M collaboration'
release = sst1mpipe.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    'sphinx.ext.githubpages', 
    "numpydoc",
    "nbsphinx",
    "sphinx_automodapi.automodapi",
    "sphinxarg.ext",
    "sphinxcontrib.mermaid",
    "sphinx_togglebutton",
    "sphinx_rtd_theme"
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

#import sphinx_rtd_theme
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
