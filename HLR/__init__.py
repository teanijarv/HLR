"""HLR - Hierarchical Linear Regression."""

__author__ = """Toomas Erik Anijärv"""
__email__ = 'toomaserikanijarv@gmail.com'
__version__ = '0.2.0'

from .diagnostic_tests import (test_durbin_watson, 
                               test_pearsons_r,
                               test_rainbow, 
                               test_breusch_pagan,
                               test_goldfeld_quandt, 
                               test_pairwise_correlations,
                               test_vif, 
                               test_extreme_standardized_residuals,
                               test_cooks_distance, 
                               test_mean_of_residuals,
                               test_shapiro_wilk)
from .plots import (create_subplot_residuals_vs_fitted, 
                    create_subplot_qq_residuals,
                    create_subplot_influence, 
                    create_subplot_std_residuals,
                    create_subplot_histogram_std_residuals, 
                    create_subplot_partial_regression)
from .regression import HierarchicalLinearRegression
