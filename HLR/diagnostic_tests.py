"""Functions for diagnostic tests to test for assumptions.

Authors
-------
Toomas Erik Anijärv toomaserikanijarv@gmail.com github.com/teanijarv
Rory Boyle rorytboyle@gmail.com github.com/rorytboyle
"""

import scipy.stats
import numpy as np
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats import diagnostic as sm_diagnostic
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

def test_durbin_watson(residuals):
    """Tests the independence of residuals using the Durbin-Watson statistic.

    Args:
        residuals (pd.Series): Residuals from a regression model.

    Returns:
        tuple: A tuple containing the Durbin-Watson statistic and a boolean indicating if the test is passed.
    """
    dw_stat = durbin_watson(residuals, axis=0)
    dw_pass = 1.5 < dw_stat < 2.5
    return dw_stat, dw_pass

def test_pearsons_r(X, y):
    """Tests linearity between each independent variable and the dependent variable using Pearson's r.

    Args:
        X (pd.DataFrame): Independent variables.
        y (pd.Series): Dependent variable.

    Returns:
        dict: A dictionary containing Pearson's r, p-value, and pass status for each independent variable.
    """
    linearity_results = {}
    for var in X.columns:
        r_value, p_value = scipy.stats.pearsonr(X[var], y.iloc[:, 0])
        test_passed = p_value < 0.05
        linearity_results[var] = {'Pearson r': r_value, 'p-value': p_value, 'Passed': test_passed}
    return linearity_results

def test_rainbow(model):
    """Tests linearity between dependent variable and independent variables collectively using Rainbow test.

    Args:
        model (RegressionResults): The fitted OLS model.

    Returns:
        tuple: A tuple containing the Rainbow statistic, p-value, and pass status.
    """
    rainbow_stat, p_value = sm_diagnostic.linear_rainbow(model)
    test_passed = p_value > 0.05
    return rainbow_stat, p_value, test_passed

def test_breusch_pagan(model):
    """Performs the Breusch-Pagan test for homoscedasticity.

    Args:
        model (RegressionResults): The fitted OLS model.

    Returns:
        tuple: A tuple containing lagrange multiplier statistic, p-value and pass status for the Breusch-Pagan test.
    """
    bp_test = sm_diagnostic.het_breuschpagan(model.resid, model.model.exog)
    bp_test_passed = bp_test[1] > 0.05
    return bp_test[0], bp_test[1], bp_test_passed

def test_goldfeld_quandt(model):
    """Performs the Goldfeld-Quandt test for homoscedasticity.

    Args:
        model (RegressionResults): The fitted OLS model.

    Returns:
        tuple: A tuple containing the F-statistic, p-value and pass status for the Goldfeld-Quandt test.
    """
    gq_test = sm_diagnostic.het_goldfeldquandt(model.resid, model.model.exog)
    gq_test_passed = gq_test[1] > 0.05
    return gq_test[0], gq_test[1], gq_test_passed

def test_pairwise_correlations(X):
    """Tests for multicollinearity by checking pairwise correlations between independent variables.

    Args:
        X (pd.DataFrame): Independent variables.

    Returns:
        tuple: A tuple containing a dictionary of pairwise correlations and a boolean indicating if the test is passed.
    """
    if len(X.columns) > 1:
        pairwise_corr = X.corr()
        np.fill_diagonal(pairwise_corr.values, np.nan)  # Excluding self-correlation
        high_corr = pairwise_corr.abs() >= 0.7
        test_passed = not high_corr.values[np.triu_indices_from(high_corr, k=1)].any()

        # Convert correlation matrix to a more readable dictionary format (upper triangle only)
        corr_dict = {}
        for i, col1 in enumerate(pairwise_corr.columns):
            for j, col2 in enumerate(pairwise_corr.columns):
                if i < j and not np.isnan(pairwise_corr.at[col1, col2]):
                    pair = f"{col1}-{col2}"
                    corr_dict[pair] = pairwise_corr.at[col1, col2]

    else:
        # Test automatically passed if there is only one independent variable
        corr_dict = {}
        test_passed = True

    return corr_dict, test_passed

def test_vif(X):
    """Calculates Variance Inflation Factors for each independent variable.

    Args:
        X (pd.DataFrame): Independent variables.

    Returns:
        tuple: A tuple containing a dictionary of VIFs and a boolean indicating if the test is passed.
    """
    if len(X.columns) > 1:
        X_const = add_constant(X)
        vifs = {X_const.columns[i]: variance_inflation_factor(X_const.values, i) for i in range(1, X_const.shape[1])}
        test_passed = all(vif < 5 for vif in vifs.values())
    else:
        vifs = {}
        test_passed = True

    return vifs, test_passed

def test_extreme_standardized_residuals(model):
    """Identifies extreme standardized residuals.

    Args:
        model (RegressionResults): The fitted OLS model.

    Returns:
        list: A list of indices with extreme standardized residuals.
    """
    influence = model.get_influence()
    standardized_residuals = influence.resid_studentized_internal
    extreme_residuals_indices = np.where(np.abs(standardized_residuals) > 3)[0]
    test_passed = len(extreme_residuals_indices) == 0
    return extreme_residuals_indices, test_passed

def test_cooks_distance(model, num_ivs):
    """Identifies high Cook's distance values.

    Args:
        model (RegressionResults): The fitted OLS model.
        num_ivs (int): Number of independent variables in the model.

    Returns:
        list: A list of indices with high Cook's distance.
    """
    influence = model.get_influence()
    cooks_d = influence.cooks_distance[0]

    # Setting Cook's distance threshold based on the number of IVs
    cooks_cutoff = 0.7 if num_ivs == 1 else 0.8 if num_ivs == 2 else 0.85
    high_cooks_indices = np.where(cooks_d > cooks_cutoff)[0]
    test_passed = len(high_cooks_indices) == 0
    return high_cooks_indices, test_passed

def test_mean_of_residuals(residuals):
    """Checks if the mean of the residuals is approximately zero.

    Args:
        residuals (pd.Series): Residuals from a regression model.

    Returns:
        tuple: A tuple containing the mean of residuals and a boolean indicating if the test is passed.
    """
    mean_residuals = residuals.mean()
    test_passed = -0.1 < mean_residuals < 0.1
    return mean_residuals, test_passed

def test_shapiro_wilk(residuals):
    """Performs the Shapiro-Wilk test for normality.

    Args:
        residuals (pd.Series): Residuals from a regression model.

    Returns:
        tuple: A tuple containing the p-value and pass status for the Shapiro-Wilk test.
    """
    shapiro_stat, shapiro_p_value = scipy.stats.shapiro(residuals)
    test_passed = shapiro_p_value > 0.05
    return shapiro_stat, shapiro_p_value, test_passed