"""Functions for running hierarchical linear regression.

Authors
-------
Toomas Erik Anijärv toomaserikanijarv@gmail.com github.com/teanijarv
Rory Boyle rorytboyle@gmail.com github.com/rorytboyle
"""
import pandas as pd
import statsmodels.api as sm
import scipy.stats
import numpy as np
import pingouin as pg
import matplotlib.pyplot as plt

from . import diagnostic_tests
from . import plots

class HierarchicalLinearRegression:
    """Class for performing hierarchical linear regression analysis."""

    def __init__(self, df, ivs_dict, dv, missing_data=None, ols_params=None):
        """Initializes the HierarchicalLinearRegression class.

        Args:
            df (pd.DataFrame): The dataset to be used in the regression.
            ivs_dict (dict): Dictionary mapping model levels to lists of independent variables.
            dv (str): The dependent variable.
            missing_data (str, optional): Handling of missing data in OLS
            ols_params (dict, optional): Optional parameters to pass to the OLS fit method.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Data is not a pandas DataFrame.")
        self.data = df

        if not isinstance(ivs_dict, dict):
            raise ValueError("ivs_dict must be a dictionary.")
        if not all(isinstance(k, int) and isinstance(v, list) for k, v in ivs_dict.items()):
            raise ValueError("ivs_dict keys must be integers and values must be lists.")
        if not all(col in df.columns for predictors in ivs_dict.values() for col in predictors):
            raise ValueError("Predictors in ivs_dict are not columns in DataFrame.")
        self.models = ivs_dict
        
        if dv not in df.columns:
            raise ValueError(f"{dv} is not a column in the DataFrame.")
        self.outcome_var = dv

        if missing_data is not None:
            if missing_data not in ['none', 'drop', 'raise']:
                raise ValueError("missing_data must be either 'none', 'drop', or 'raise'.")
        self.missing_data = missing_data if missing_data is not None else 'none'

        if ols_params is not None:
            if not isinstance(ols_params, dict):
                raise ValueError("ols_params must be a dictionary.")
        self.ols_params = ols_params if ols_params is not None else {}

    def fit_models(self):
        """Fits OLS models for each level of independent variables.

        Returns:
            dict: A dictionary of statsmodels OLS regression results.
        """
        results = {}
        for level, predictors in self.models.items():
            X = self.data[predictors]
            X_const = sm.add_constant(X)
            y = self.data[self.outcome_var]
            model = sm.OLS(y, X_const, missing=self.missing_data).fit(**self.ols_params)
            results[level] = model
        return results

    def calculate_change_stats(self):
        """Calculates change statistics between OLS models.

        Returns:
            dict: A dictionary containing change statistics.
        """
        stats = {}
        prev_model = None
        for level, curr_model in self.fit_models().items():
            if prev_model:
                pred_change = len(curr_model.model.exog[0]) - len(prev_model.model.exog[0])
                r_sq_change = curr_model.rsquared - prev_model.rsquared
                f_change = (r_sq_change / pred_change) / ((1 - curr_model.rsquared) / curr_model.df_resid)
                f_change_pval = scipy.stats.f.sf(f_change, pred_change, curr_model.df_resid)

                stats[level] = {
                    'R-squared change': r_sq_change,
                    'F-value change': f_change,
                    'P-value (F-value change)': f_change_pval
                }
            prev_model = curr_model
        return stats
    
    def calculate_additional_stats(self, model, X, y):
        """Calculates additional statistics for a given OLS model.

        Args:
            model (RegressionResults): The regression model.
            X (pd.DataFrame): Independent variables.
            y (pd.DataFrame): Dependent variable.

        Returns:
            dict: A dictionary containing additional statistics.
        """
        std_x = X.apply(np.std, ddof=1)
        std_y = y.apply(np.std, ddof=1).iloc[0]
        conv_factors = std_x / std_y
        beta_coeffs = pd.Series(model.params).drop('const')
        std_beta_coefs = (beta_coeffs * conv_factors).to_dict()

        Xy_temp = pd.concat([X, y], axis=1)
        partials = {col: pg.partial_corr(data=Xy_temp, x=col, y=y.columns[0], 
                                         covar=[x for x in X.columns if x != col])['r'].iloc[0] for col in X.columns}
        semi_partials = {col: pg.partial_corr(data=Xy_temp, x=col, y=y.columns[0], 
                                              x_covar=[x for x in X.columns if x != col])['r'].iloc[0] for col in X.columns}
        unique_var = {col: value ** 2 * 100 for col, value in semi_partials.items()}

        return {
            'Std Beta coefs': std_beta_coefs,
            'Partial correlations': partials,
            'Semi-partial correlations': semi_partials,
            'Unique variance %': unique_var
        }

    def summary(self):
        """Generates a summary report of the hierarchical linear regression analysis.

        Returns:
            pd.DataFrame: A DataFrame containing the summary report.
        """
        model_res = self.fit_models()
        change_stats = self.calculate_change_stats()

        report = []
        for level, model in model_res.items():
            X = self.data[model.model.exog_names[1:]]
            y = self.data[[self.outcome_var]]

            add_stats = self.calculate_additional_stats(model, X, y)

            results = {
                'Model Level': level,
                'Predictors': model.model.exog_names[1:],
                'N (observations)': model.nobs,
                'DF (residuals)': model.df_resid,
                'DF (model)': model.df_model,
                'R-squared': model.rsquared,
                'F-value': model.fvalue,
                'P-value (F)': model.f_pvalue,
                'SSR': model.ssr,
                'SSTO': model.centered_tss,
                'MSE (model)': model.mse_model,
                'MSE (residuals)': model.mse_resid,
                'MSE (total)': model.mse_total,
                'Beta coefs': model.params.to_dict(),
                'P-values (beta coefs)': model.pvalues.to_dict()
            }
            results.update(add_stats)
            if level in change_stats:
                results.update(change_stats[level])
            report.append(results)

        return pd.DataFrame(report)
    
    def diagnostics(self, verbose=True):
        """Performs diagnostics tests on the fitted models.

        Returns:
            dict: A dictionary containing diagnostics results.
        """
        model_results = self.fit_models()
        diagnostics_results = {}

        for level, model in model_results.items():
            X = self.data[model.model.exog_names[1:]]  # Exclude 'const'
            y = self.data[[self.outcome_var]]
            num_ivs = len(self.models[level])

            # Independence of residuals (Durbin-Watson)
            dw_stat, dw_pass = diagnostic_tests.test_durbin_watson(model.resid)
            # Linearity (Pearson's r and Rainbow test)
            pearsons_r_results = diagnostic_tests.test_pearsons_r(X, y)
            rainbow_stat, rainbow_pal, rainbow_pass = diagnostic_tests.test_rainbow(model)
            # Homoscedasticity (Breusch-Pagan and Goldfeld-Quandt tests)
            bp_stat, bp_pval, bp_pass = diagnostic_tests.test_breusch_pagan(model)
            gq_stat, gq_pval, gq_pass = diagnostic_tests.test_goldfeld_quandt(model)
            # Multicollinearity (Pairwise correlations and VIF)
            pairwise_corr, multicollinearity_pass = diagnostic_tests.test_pairwise_correlations(X)
            vif_dict, vif_pass = diagnostic_tests.test_vif(X)
            # Outliers (Extreme Standardized Residuals and High Cook's Distance)
            extreme_resid_indices, extreme_resid_pass = diagnostic_tests.test_extreme_standardized_residuals(model)
            high_cooks_indices, cooks_pass = diagnostic_tests.test_cooks_distance(model, num_ivs)
            # Normality (Mean of Residuals and Shapiro-Wilk Test)
            mean_residuals, mean_residuals_pass = diagnostic_tests.test_mean_of_residuals(model.resid)
            shapiro_stat, shapiro_pval, shapiro_pass = diagnostic_tests.test_shapiro_wilk(model.resid)

            diagnostics_results[level] = {
                'Independence of residuals (Durbin-Watson test)': {'DW stat': dw_stat, 'Passed': dw_pass},
                'Linearity (Pearson r)': pearsons_r_results,
                'Linearity (Rainbow test)': {'Rainbow Stat': rainbow_stat, 'p-value': rainbow_pal, 'Passed': rainbow_pass},
                'Homoscedasticity (Breusch-Pagan test)': {'Lagrange Stat': bp_stat, 'p-value': bp_pval, 'Passed': bp_pass},
                'Homoscedasticity (Goldfeld-Quandt test)': {'F-Stat': gq_stat, 'p-value': gq_pval, 'Passed': gq_pass},
                'Multicollinearity (pairwise correlations)': {'Correlations': pairwise_corr, 'Passed': multicollinearity_pass},
                'Multicollinearity (Variance Inflation Factors)': {'VIFs': vif_dict, 'Passed': vif_pass},
                'Outliers (extreme standardized residuals)': {'Indices': extreme_resid_indices, 'Passed': extreme_resid_pass},
                'Outliers (high Cooks distance)': {'Indices': high_cooks_indices, 'Passed': cooks_pass},
                'Normality (mean of residuals)': {'Mean': mean_residuals, 'Passed': mean_residuals_pass},
                'Normality (Shapiro-Wilk test)': {'SW Stat': shapiro_stat, 'p-value': shapiro_pval, 'Passed': shapiro_pass}
            }
        
        if verbose:
            for level, results in diagnostics_results.items():
                print(f"\nModel Level {level} Diagnostics:")
                for test_name, test_results in results.items():
                    print(f"  {test_name}:")
                    for key, value in test_results.items():
                        print(f"    {key}: {value}")

        return diagnostics_results
    
    def plot_studentized_residuals_vs_fitted(self):
        """Plots studentized residuals against fitted values for all model levels."""
        model_results = self.fit_models()
        num_levels = len(model_results)
        
        fig, axs = plt.subplots(num_levels, 1, figsize=(8, 4 * num_levels))
        
        if num_levels == 1:
            axs = [axs]  # Ensure axs is iterable even for a single subplot

        for ax, (level, model) in zip(axs, model_results.items()):
            plots.create_subplot_residuals_vs_fitted(ax, model, level)

        plt.tight_layout()
        plt.show()
    
    def plot_qq_residuals(self):
        """Plots Normal QQ Plots for all model levels."""
        model_results = self.fit_models()
        fig, axs = plt.subplots(len(model_results), 1, figsize=(8, 4 * len(model_results)))
        if len(model_results) == 1:
            axs = [axs]

        for ax, (level, model) in zip(axs, model_results.items()):
            plots.create_subplot_qq_residuals(model.resid, ax, level)

        plt.tight_layout()
        plt.show()

    def plot_influence(self):
        """Plots Influence Plots for all model levels."""
        model_results = self.fit_models()
        fig, axs = plt.subplots(len(model_results), 1, figsize=(8, 4 * len(model_results)))
        if len(model_results) == 1:
            axs = [axs]

        for ax, (level, model) in zip(axs, model_results.items()):
            plots.create_subplot_influence(model, ax, level)

        plt.tight_layout()
        plt.show()

    def plot_std_residuals(self):
        """Plots Box Plots of Standardized Residuals for all model levels."""
        model_results = self.fit_models()
        fig, axs = plt.subplots(len(model_results), 1, figsize=(8, 4 * len(model_results)))
        if len(model_results) == 1:
            axs = [axs]

        for ax, (level, model) in zip(axs, model_results.items()):
            influence = model.get_influence()
            standardized_residuals = influence.resid_studentized_internal
            plots.create_subplot_std_residuals(standardized_residuals, ax, level)

        plt.tight_layout()
        plt.show()
    
    def plot_histogram_std_residuals(self):
        """Plots Histogram of Standardized Residuals for all model levels."""
        model_results = self.fit_models()
        fig, axs = plt.subplots(len(model_results), 1, figsize=(8, 4 * len(model_results)))
        if len(model_results) == 1:
            axs = [axs]

        for ax, (level, model) in zip(axs, model_results.items()):
            influence = model.get_influence()
            standardized_residuals = influence.resid_studentized_internal
            plots.create_subplot_histogram_std_residuals(standardized_residuals, ax, level)

        plt.tight_layout()
        plt.show()

    def plot_partial_regression(self):
        """Plots Partial Regression Plots for all model levels."""
        model_results = self.fit_models()
        num_ivs = max(len(ivs) for ivs in self.models.values())
        fig_size = (15, min(10, 5 * num_ivs))

        for level, model in model_results.items():
            plots.create_subplot_partial_regression(model, fig_size, level)