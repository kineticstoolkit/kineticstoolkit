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
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

# -- Project information -----------------------------------------------------

# The full version, including alpha/beta/rc tags
with open('../kineticstoolkit/VERSION', 'r') as fid:
    release = fid.read()

if release == 'master':
    project = 'Kinetics Toolkit (dev)'
else:
    project = 'Kinetics Toolkit'

copyright = '2020-2021, Félix Chénier'
author = 'Félix Chénier'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',  # Document objects using docstrings in API
    'sphinx.ext.autosummary',  # Generate summary table pages (for autodoc)
    'sphinx.ext.napoleon',  # Parse numpy-style docstrings (for autodoc)
    # Type hints in doc instead of signature (for autodoc)
    'sphinx_autodoc_typehints',
    'autodocsumm',  # Add a summary table at the top of each API page
    'sphinx.ext.ifconfig',  # Allow conditional contents for master vs stable versions
    'nbsphinx',
]

autodoc_default_options = {
    'autosummary': True,
    'autodoc_typehints': 'description',
    'autosummary-imported-members': False,
}

nbsphinx_execute = 'never'


# Napoleon settings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# Matplotlib settings
plot_html_show_source_link = False
plot_html_show_formats = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '.ipynb_checkpoints',
                    'api/external.*', 'api/kineticstoolkit.external.*',
                    'api/kineticstoolkit.cmdgui.*']

if release not in ['master']:
    exclude_patterns.append('dev/*')

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Read the docs theme configuration
html_theme_options = {
    'logo_only': True,  # default False
    'display_version': False,
    'prev_next_buttons_location': 'both',  # default 'bottom'
    'style_external_links': False,
    'vcs_pageview_mode': '',
    'style_nav_header_background': 'black',
    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False,
}

html_sidebars = {
    "**": ["globaltoc.html", "localtoc.html", "searchbox.html"]
}

html_show_sourcelink = False
html_copy_source = False
html_logo = '_static/logo_with_text_black.png'
html_css_files = ['css/custom.css']

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

if release == 'master':
    # Modify some parameters to distingate the development site
    html_logo = '_static/logo_development.png'
