"""Top-level package for HLR - Hierarchical Linear Regression."""

__author__ = """Toomas Erik Anijärv"""
__email__ = 'toomaserikanijarv@gmail.com'
__version__ = '0.1.0'

from .diagnostic_tests import regression_diagnostics
from .hierarchical_regression import (linear_reg,
                                      calculate_change_stats,
                                      hierarchical_regression)
from .model import HLR_model