---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Installing from git-hub

To help developping Kinetics Toolkits by testing it or extending it, you can install it directly from Git-hub.

This howto assumes you already have a working conda installation.

1. Install the dependencies

```
conda install -c conda-forge python=3.8 matplotlib scipy pandas scikit-learn pyqt ezc3d limitedinteraction git pytest mypy coverage jupyterlab spyder sphinx sphinx_rtd_theme sphinx-autodoc-typehints nbsphinx twine
```

1. Clone the git repository

```
git clone https://github.com/felixchenier/kineticstoolkit.git ktk_develop
```

3. Add ktk_develop to your PYTHON_PATH

Add the `ktk_develop` folder to your PYTHON_PATH according to your favourite IDE and python installation.

4. To keep up to date

```
cd ktk_develop
git pull origin master
```