# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys
import os
import importlib.metadata
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

project = 'Multilayers'
copyright = '2025, Advanced Research Center for Nanolithography (ARCNL)'
author = 'Ester Abram'
release = '2.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# -- Path setup --------------------------------------------------------------
source_dir = Path().resolve()
module_dir = source_dir.parent.parent / "multilayers"

# -- Run sphinx-apidoc automatic ---------------------------------------------
os.environ["SPHINX_APIDOC_OPTIONS"] = "members"
from sphinx.ext import apidoc

exclude_pattern = module_dir / "multilayers.py"
cmd_line = f"sphinx-apidoc -M -f -o {source_dir} {module_dir} "
apidoc.main(cmd_line.split(" ")[1:])


#extensions = []
extensions = [
    "sphinx.ext.autodoc",
    "sphinx_autodoc_defaultargs",
    "sphinx_autodoc_typehints",
]

typehints_use_rtype = False

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
