# -*- coding: utf-8 -*-
"""Configuration file for Kinetics Toolkit API."""

import os
import sys
from datetime import datetime
from pathlib import Path

with open("../kineticstoolkit/VERSION", "r") as fh:
    release = fh.read()

# -- General setup --------------------------------------------------------------
project = "Kinetics Toolkit"
author = "Félix Chénier"
copyright = f"2020-{datetime.now().year}"

# -- Extension Setup ------------------------------------------------------------
extensions = [
    # Core Sphinx extensions
    "sphinx.ext.autodoc",  # Document objects using docstrings in API
    "sphinx.ext.autosummary",  # Generate summary table pages (for autodoc)
    "sphinx.ext.napoleon",  # Parse numpy/google-style docstrings (for autodoc)
    "sphinx.ext.viewcode",  # Add links to view code on GitHub
    # Custom extensions (Must be registered in your package or installed as a pip package)
    "sphinx_sitemap",
    "sphinxext.opengraph",  # Generate metadata for social media
]


# Configure autodoc to use summary pages and generate them during build
autosummary_generate = True
add_module_names = (
    False  # Remove module names from docstrings for cleaner API look
)

# Configure Napoleon (Google/NumPy style parsing)
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


# -- HTML Options --------------------------------------------------------------
html_theme = "sphinx_book_theme"
html_baseurl = "https://kineticstoolkit.uqam.ca/doc/"
html_favicon = "_static/favicon.ico"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# Theme options for sphinx-book-theme (maps from jupyter-book config)
html_theme_options = {
    "repository_url": "https://github.com/felixchenier/kineticstoolkit",
    "use_repository_button": True,
    "use_issues_button": True,
    "collapse_navigation": True,
    "navigation_depth": 4,
    "show_toc_level": 1,
    "logo": {
        "image_light": "_static/logo.png",
        "image_dark": "_static/logo-dark.png",
    },
}

# Configure Sphinx Sitemap
sitemap_change_freq = "weekly"
sitemap_priority = "0.7"

# Configure OpenGraph (social media cards)
ogp_site_url = "https://kineticstoolkit.uqam.ca/doc/"
ogp_description_length = 200
ogp_site_name = "Kinetics Toolkit"
ogp_image = "https://kineticstoolkit.uqam.ca/doc/_static/logo-social.png"
ogp_image_alt = "Kinetics Toolkit Logo"
ogp_type = "image/png"

# Add custom OpenGraph tags
ogp_custom_meta_tags = [
    '<meta property="og:image:width" content="500" />',
    '<meta property="og:image:height" content="264" />',
]

# Exclude specific patterns if you still need to hide some files (like build artifacts)
exclude_patterns = ["_templates", "_build", "Thumbs.db", ".DS_Store"]


# -- API Type Hint Configuration -----------------------------------------------
# These allow you to reference type hints like 'ArrayLike' or 'TimeSeries' without issues
autodoc_typehints = "description"
autodoc_type_aliases = {
    "ArrayLike": "ArrayLike",
    # 'kineticstoolkit._timeseries.TimeSeries': 'TimeSeries',  # Uncomment if needed
    # 'pandas.core.frame.DataFrame': 'DataFrame',            # Uncomment if needed
    # 'pandas.core.frame.Series': 'Series',                  # Uncomment if needed
}
