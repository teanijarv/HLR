"""Functions for running hierarchical linear regression.

Authors
-------
Toomas Erik Anijärv toomaserikanijarv@gmail.com github.com/teanijarv
Rory Boyle rorytboyle@gmail.com github.com/rorytboyle
"""

import statsmodels.api as sm
import scipy as scipy
import pandas as pd
import numpy as np
import pingouin as pg
import os

from HLR import diagnostic_tests

def linear_reg(X, X_names, y):
    """Runs a linear regression, extracts results from the model.
    Returns a list of results and the statsmodels.OLS results object.

    Parameters
    ----------
    X : list of Pandas dataframes
        Each list element containing dataframe of predictors in each step.
    X_names : list of lists containing strings
        Corresponding predictor variable names for each step.
    y : Pandas dataframe
        Outcome variable
    
    Returns
    -------
    model_results : list
        List of extracted results from statsmodels.OLS object
    model : statsmodels' OLS results object
        Model object for linear regression
    """
    # Run linear regression & add column of ones to X to serve as intercept
    model = sm.OLS(y, sm.add_constant(X)).fit()
    
    # Extract results from statsmodels.OLS object
    results = [X_names, model.nobs, model.df_resid, model.df_model,
               model.rsquared, model.fvalue, model.f_pvalue, model.ssr,
               model.centered_tss, model.mse_model, model.mse_resid,
               model.mse_total]

    # Deep copy names and add constant. Otherwise results list will contain
    # multiple repetitions of constant (due to below loop)
    namesCopy = X_names[:]
    namesCopy.insert(0, 'Constant')
    
    # Create dicts with name of each parameter in model (i.e. predictor
    # variables) and the beta coefficient and p-value
    coeffs = {}
    p_values = {}
    for ix, coeff in enumerate(model.params):
        coeffs[namesCopy[ix]] = coeff
        p_values[namesCopy[ix]] = model.pvalues[ix]    

    # 26/03/2023 (Jules) - Standardised beta coefficients calculated and appended
    # Std for predictors in step calculated, and std of outcome set to length of predictors
    # Conversion factor created (std x/std y), applied to each coefficient and zipped back to dictionary of predictors
    
    std_x = pd.Series(np.std(X)).to_list() # 25/03/23 dictionary to store standard deviation of predictor variables
    std_y = pd.Series(np.std(y)).to_list()  # 25/03/23 calculate standard deviation of outcome
    std_y = std_y*len(std_x)
    conversion = [x / y for x, y in zip(std_x, std_y)]

    coeffs_list = coeffs.copy()
    coeffs_list.pop('Constant')
    coeffs_keys, coeffs_values = zip(*coeffs_list.items())
    beta_conversion = [b * c for b,c in zip(coeffs_values, conversion)]
    std_coeffs = dict(zip(coeffs_keys, beta_conversion)) # 25/03/23 dictionary to store beta_coefficients

    # 05/04/2023 - Calculate the partial and semi-partial (defined 'part' in SPSS) correlations
    # for each X value with y, while holding other X values as covariates
    # Initialize an array to store partial and semi-partial correlation values
    x_cols = [X.columns.tolist()]
    semi_partial_correlations = np.zeros(len(x_cols[0]))
    partial_correlations = np.zeros(len(x_cols[0]))

    for i in range(len(x_cols[0])):
        # Create a mask to exclude the current variable from the covariates
        mask = [True] * len(x_cols[0])
        mask[i] = False
        
        #Concat X and Y dataframes
        Xy_temp = pd.concat([X, y], axis=1)
        
        # Extract the current variable (x) from xx
        xx = x_cols[0][i]

        # Extract the remaining variables (covariates) from xx
        covariates = [x_cols[0][j] for j in range(len(x_cols[0])) if mask[j]]
 
        # Calculate partial and semi-partial values and assign to array
        semi_partial_corr = pg.partial_corr(data = Xy_temp, x=xx, y=y.columns[0], x_covar=covariates)
        partial_corr = pg.partial_corr(data = Xy_temp, x=xx, y=y.columns[0], covar=covariates)
        semi_partial_correlations[i] = semi_partial_corr['r']
        partial_correlations[i] = partial_corr['r']
        
       
    # Create dict to zip variables names to values
    dict_partial = dict(zip(coeffs_keys, partial_correlations)) # 25/03/23 dictionary to store beta_coefficients
    dict_semi_partial = dict(zip(coeffs_keys, semi_partial_correlations)) # 25/03/23 dictionary to store beta_coefficients

    # Square semi partial values to get unique variance accounted for.\
    unique_var = (semi_partial_correlations*semi_partial_correlations)*100
    dict_unique_var = dict(zip(coeffs_keys, unique_var)) # 25/03/23 dictionary to store beta_coefficients

    # Append to results
    results.append(coeffs)
    results.append(p_values)  
    results.append(std_coeffs)
    results.append(dict_partial)
    results.append(dict_semi_partial)
    results.append(dict_unique_var)

    return results, model

def calculate_change_stats(model_stats):
    """Calculates r-squared change, f change, and p-value of f change for
    hierarchical regression results.

    Parameters
    ----------
    model_stats : list
        Description of the model.
    
    Returns
    -------
    list containing r-squared change value, f change value, and
        p-value for f change

    Code notes
    ----------
    f change is calculated using the formula:
    (r_squared change from Step 1 to Step 2 / no. predictors added in Step 2) / 
    (1 - step 2 r_squared) / (no. observations - no. predictors - 1)
    https://www.researchgate.net/post/What_is_a_significant_f_change_value_in_a_hierarchical_multiple_regression
        
    p-value of f change calculated using the formula:
    f with (num predictors added, n - k - 1) ==> n-k-1 = Residual df for Step 2
    https://stackoverflow.com/questions/39813470/f-test-with-python-finding-the-critical-value
    """
    # get number of steps 
    num_steps = model_stats['Step'].max()
    
    # calculate r-square change (r-sq of current step minus r-sq of previous step)
    r_sq_change = [model_stats.iloc[step+1]['R-squared'] -
                   model_stats.iloc[step]['R-squared'] for step in
                   range(0, num_steps-1)]
    
    # calculate f change - formula from here: 
    f_change = []
    f_change_pval = []
    for step in range(0, num_steps-1):
        # numerator of f change formula
        # 24/03/2023 Predictor change variable created
        predictor_change = len(model_stats.iloc[step+1]['Predictors']) - len(model_stats.iloc[step]['Predictors'])
        
        # (r_sq change / number of predictors added)
        f_change_numerator = r_sq_change[step] / predictor_change
        # denominator of f change formula
        # (1 - step2 r_sq) / (num obs - number of predictors - 1)
        f_change_denominator = ((1 - model_stats.iloc[step+1]['R-squared']) /
                                model_stats.iloc[step+1]['DF (residuals)'])
        # compute f change
        f_change.append(f_change_numerator / f_change_denominator)
        
        # calculate pvalue of f change
        # 24/03/2023 Difference in predictors/step added as df1 (originally had 1)
        # 05/04/2023 p-value for f change fixed (brought function to the for loop)
        f_change_pval_temp = scipy.stats.f.sf(f_change[step], predictor_change, 
                                        model_stats.iloc[step+1]['DF (residuals)'])
        f_change_pval.append(f_change_pval_temp)
    
    return [r_sq_change, f_change, f_change_pval]

def hierarchical_regression(X, X_names, y, diagnostics=True, save_folder='results',
                            showfig=False, verbose=True):
    """Runs hierarchical linear regressions predicting y from X. Uses statsmodels
    OLS to run linear regression for each step. Returns results of regression
    in each step as well as r-squared change, f change, and p-value of f change
    for the change from step 1 to step 2, step 2 to step 3, and so on.
    
    The number of lists contained within names specifies the number of steps of
    hierarchical regressions. If names contains two nested lists of strings,
    e.g. if names = [[variable 1], [variable 1, variable 2]], then a two-step
    hierarchical regression will be conducted.
    
    Parameters
    ----------
    X : list of Pandas dataframes
        Each list element containing dataframe of predictors in each step.
    X_names : list of lists containing strings
        Corresponding predictor variable names for each step.
    y : Pandas dataframe
        Outcome variable
    save_folder : string, optional, default: 'results'
        Folder location in root where to save the results (and diagnostics).
    
    Returns
    -------
    model_stats : Pandas dataframe
        Results from the hierarchical regression model.
    reg_models : list of statsmodels' RegressionResultsWrapper objects
        Model objects for each step in a list.
    
    Code notes
    ----------
    model_stats (rows = nr of steps; cols = 18) contains the following
    information for each step:
        Step = step number
        x = predictor names
        N (observations) = number of observations in model
        DF (residuals) = df of residuals
        DF (model) = df of model
        R-squared = r-squared
        F-value = f-value
        P-value (F) = p-value
        SSE = sum of squares of errors
        SSTO = total sum of squares
        MSE (model) = mean squared error of model
        MSE (residuals) =  mean square error of residuals
        MSE (total) = total mean square error
        Beta coefs = coefficient values for intercept and predictors
        P-values (beta coefs) = p-values for intercept and predictors
        Std Beta coefs = standardised beta coefficients for intercept and predictors
        Partial correlation = partial correlation for each X value with y
        Semi-partial correlation = semi-partial correlation for each X value with y
        Unique Variance % = squared semi-partial correlation for unique variance of predictors
        R-squared change = r-squared change for model (Step 2 r-sq - Step 1 r-sq)
        F-value change = f change for model (Step 2 f - Step 1 f)
        P-value (F change) = p-value of f-change of model
    """
    # Loop through steps and run regressions for each step
    results =[]
    reg_models = []
    for ix, currentX in enumerate(X):
        # Run linear regression for the current step
        currentStepResults, currentStepModel = linear_reg(currentX, X_names[ix], y)

        # Add step number to results
        currentStepResults.insert(0, ix+1)

        # Save the results and model object to results folder
        saveto = save_folder + r'/step' + str(ix+1)
        try:
            os.makedirs(saveto)
        except:
            pass
        modelSave = saveto + "model.pickle"
        currentStepModel.save(modelSave)
        
        # Run diagnostic tests for assumptions
        if diagnostics == True:
            assumptionsToCheck = diagnostic_tests.regression_diagnostics(
                    currentStepModel, currentStepResults, y, currentX, X_names[ix], 
                    saveto=saveto, showfig=showfig, verbose=verbose)
            currentStepResults.append(assumptionsToCheck)
        
        # Add the new results to the results list, same with the model object
        results.append(currentStepResults)
        reg_models.append(['Step ' + str(ix+1), currentStepModel])
        
    # Add the results to model_stats dataframe and name the columns
    # 26/03/2023 (Jules) - Std beta coefs added
    model_stats = pd.DataFrame(results)
    if diagnostics == True:
        model_stats.columns = ['Step', 'Predictors', 'N (observations)', 'DF (residuals)',
                            'DF (model)', 'R-squared', 'F-value', 'P-value (F)', 'SSE', 'SSTO',
                            'MSE (model)', 'MSE (residuals)', 'MSE (total)', 'Beta coefs',
                            'P-values (beta coefs)','Std Beta coefs', 'Partial Correlation', 
                            'Semi-partial Correlation', 'Unique Variance %' , 'Failed assumptions (check!)']
    else:
        model_stats.columns = ['Step', 'Predictors', 'N (observations)', 'DF (residuals)',
                            'DF (model)', 'R-squared', 'F-value', 'P-value (F)', 'SSE', 'SSTO',
                            'MSE (model)', 'MSE (residuals)', 'MSE (total)', 'Beta coefs',
                            'P-values (beta coefs)','Std Beta coefs', 'Partial Correlation', 
                            'Semi-partial Correlation', 'Unique Variance %'] 
    
    # Create change results by calculating r-sq change, f change, p-value of f change between steps
    change_results = calculate_change_stats(model_stats)
    
    # Append step number to the change results
    step_nums = [x+1 for x in [*range(1, len(change_results[0])+1)]]
    change_results.insert(0, step_nums)
    
    # Add the change results to the change stats dataframe
    change_stats = pd.DataFrame(change_results).transpose()
    change_stats.columns = ['Step', 'R-squared change', 'F-value change', 'P-value (F change)'] 
    
    # Merge model_stats and change_stats, creating the final model results dataframe
    model_stats = pd.merge(model_stats, change_stats, on='Step', how='outer')
        
    return model_stats, reg_models