# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))  
sys.path.insert(0, os.path.abspath('../../src'))

print("Python path:")
for p in sys.path:
    print(p)

try:
    import sampling_planners
    print("成功导入 sampling_planners!")
except ImportError as e:
    print(f"导入失败: {e}")

project = 'sampling_based_planner_library'
copyright = '2025, Chengjin Wang'
author = 'Chengjin Wang'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',    
    'sphinx.ext.napoleon',   
    'sphinx.ext.viewcode',   
    'sphinx.ext.intersphinx',
]

templates_path = ['_templates']
exclude_patterns = []

language = 'English'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
