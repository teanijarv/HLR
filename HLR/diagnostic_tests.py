"""Function for diagnostic tests to test for assumptions.

Authors
-------
Toomas Erik Anijärv toomaserikanijarv@gmail.com github.com/teanijarv
Rory Boyle rorytboyle@gmail.com github.com/rorytboyle
"""

import statsmodels.api as sm
import statsmodels.stats.outliers_influence as sm_diagnostics
from statsmodels.tools.tools import add_constant
import statsmodels.stats as sm_stats
import matplotlib.pyplot as plt
import scipy as scipy
import pandas as pd
import seaborn as sns
import os, sys

def regression_diagnostics(model, result, y, X, X_names, saveto='results', showfig=False, verbose=True):
    """Runs formal diagnostic tests for linear regression and creates plots for
    further inspection. Outputs a summary text file listing the failed
    diagnostic tests and a list of assumptions that require further inspection.

    Assumption tested               Diagnostic Test(s) & Plots used

    1. Independence of Residuals    Durbin Watson Test
    
    2. Linearity                    Pearson's Correlations for DV and each IV
                                    Rainbow Test
                                    Plot: Studentised Residuals vs Fitted Values
                                    Plot: Partial Regression Plots
                                    
    3. Homoscedasticity             Breusch Pagan Test
                                    F-test
                                    Goldfeld Quandt Test
                                    Plot: Studentised Residuals vs Fitted Values
                                    
    4. Multicollinearity            Pairwise Correlations between DVs
                                    Variance Inflation Factor
                                    
    5. Outliers/Influence           Standardised Residuals (> -3 & < +3)
                                    Cook's Distance
                                    Plot: Boxplot of Standardised Residuals
                                    Plot: Influence Plot with Cook's Distance
                                    
    6. Normality                    Mean of Residuals (approx = 0)
                                    Shapiro-Wilk Test
                                    Plot: Normal QQ Plot of Residuals

    Parameters
    ----------
    model : statsmodels' RegressionResultsWrapper object
        Current step's regression model.
    result : Pandas series
        One row of results from regression.
    y : Pandas dataframe
        Outcome variable
    X : list of Pandas dataframes
        Each list element containing dataframe of predictors in each step.
    saveto : string, optional, default: 'results'
        A Folder to save diagnostic test results and plots.
    showfig : bool, optional, default: False
        Whether to show diagnostics plots (or only save them).
    verbose : bool, optional, default: True
        Whether to print diagnostic log.

    Returns
    -------
    assumptionsToCheck : list
        Assumptions that require further inspection.
    """
    # Get residuals' values
    influence_df = sm_diagnostics.OLSInfluence(model).summary_frame()
    # Create a dictionary to store diagnostics test info
    diagnostics = {}
    # Create a dictionary to link diagnostic tests to assumptions
    assumptionTests = {}
    # Create a dictionary with formal names of diagnostic tests (for printing warnings)
    formalNames = {}
    # Create a folder for saving diagnostics
    try:
        os.makedirs(saveto)
    except:
        pass
    # Get step number
    step = saveto.split("/")[-1]

    ###### ASSUMPTION I - INDEPENDENCE OF RESIDUALS

    ### (1) Durbin-Watson stat (no autocorrelation)

    # Calculate the Durbin-Watson statistic
    diagnostics['durbin_watson_stat'] = sm_stats.stattools.durbin_watson(
           model.resid, axis=0)
    # If the statistic is between 1.5-2.5, the test is passed
    if diagnostics['durbin_watson_stat'] >= 1.5 and diagnostics[
            'durbin_watson_stat'] <= 2.5:
        diagnostics['durbin_watson_passed'] = 'Yes'
    else:
        diagnostics['durbin_watson_passed'] = 'No'
    # Link test to assumption
    assumptionTests['durbin_watson_passed'] = 'Independence of Residuals'
    formalNames['durbin_watson_passed'] = 'Durbin-Watson Test'

    ###### ASSUMPTION II - LINEARITY

    ### (2) Pearson's r (linearity between DV and each IV)

    # If there are multiple predictors (IVs), find correlation for all with DV
    if len(X.columns) > 1:
        correlations = [scipy.stats.pearsonr(X[var], y.iloc[:, 0])
                        for var in X.columns]
        for ix, corr in enumerate(correlations):
            xName = 'IV_' + X.columns[ix] + '_pearson_'
            diagnostics[xName + 'r'] = corr[0]
            diagnostics[xName + 'p'] = corr[1]
    # If only one predictor (IV), find correlation with DV
    else:
        correlations = scipy.stats.pearsonr(X.iloc[:, 0], y.iloc[:, 0])
    # Go through all Pearson's p-values and if larger than 0.05, add to list
    nonSigLinearIV_toDV = 0
    nonSigLinearVars = []
    for key in diagnostics:
        if key[-9:] == 'pearson_p':
            if diagnostics[key] > 0.05:
                nonSigLinearIV_toDV += 1
                nonSigLinearVars.append(key)
    # If none were added to the non-significant list, the test is passed and vice versa
    if nonSigLinearIV_toDV == 0:
        diagnostics['linear_DVandIVs_passed'] = 'Yes'
    else:
        diagnostics['linear_DVandIVs_passed'] = 'No:' + ', '.join(
                nonSigLinearVars)
    # Link test to assumption
    assumptionTests['linear_DVandIVs_passed'] = 'Linearity'
    formalNames['linear_DVandIVs_passed'] = 'Non-sig. linear relationship between DV and each IV'

    ### (3) Rainbow test (linearity between DV and IVs collectively)

    # Apply rainbow test for linearity, where 0-hypothesis = model has linear fit
    diagnostics['rainbow_linearity'] = sm_stats.diagnostic.linear_rainbow(
            model)[1]
    # If p>0.05 (rejecting 0-hypothesis), the test is passed, and vice versa
    if diagnostics['rainbow_linearity'] > 0.05:
        diagnostics['rainbow_linearity_passed'] = 'Yes'
    else:
        diagnostics['rainbow_linearity_passed'] = 'No'
    # Link test to assumption
    assumptionTests['rainbow_linearity_passed'] = 'Linearity'
    formalNames['rainbow_linearity_passed'] = 'Rainbow Test'

    ###### ASSUMPTION III - HOMOSCEDASTICITY
    
    ### (4) Breusch-Pagan Lagrange Multiplier test

    # Apply Breusch-Pagan Lagrange test for residual variance dependance
    breusch_pagan_test = sm_stats.diagnostic.het_breuschpagan(
            model.resid, model.model.exog)
    diagnostics['breusch_pagan_p'] = breusch_pagan_test[1]
    diagnostics['f_test_p'] = breusch_pagan_test[3]
    # If lagrange multiplier test is  p<0.05, the test is failed and vice versa
    if diagnostics['breusch_pagan_p'] < .05:
        diagnostics['breusch_pagan_passed'] = 'No'
    else:
        diagnostics['breusch_pagan_passed'] = 'Yes'
    # Link test to assumption
    assumptionTests['breusch_pagan_passed'] = 'Homoscedasticity'
    formalNames['breusch_pagan_passed'] = 'Bruesch Pagan Test'
    # If F-test is p<0.05, the test is failed and vice versa (better for small samples)
    if diagnostics['f_test_p'] < .05:
        diagnostics['f_test_passed'] = 'No'
    else:
        diagnostics['f_test_passed'] = 'Yes'
    # Link test to assumption
    assumptionTests['f_test_passed'] = 'Homoscedasticity'
    formalNames['f_test_passed'] = 'F-test for residual variance'

    ### (5) Goldfeld-Quandt homoscedasticity test

    # Apply Goldfeld-Quandt test to check if residual variance is same in 2 subsamples
    goldfeld_quandt_test = sm_stats.api.het_goldfeldquandt(
            model.resid, model.model.exog)
    diagnostics['goldfeld_quandt_p'] = goldfeld_quandt_test[1]
    # If p<0.05 (rejecting 0-hypothesis), the test is failed, and vice versa
    if diagnostics['goldfeld_quandt_p'] < .05:
        diagnostics['goldfeld_quandt_passed'] = 'No'
    else:
        diagnostics['goldfeld_quandt_passed'] = 'Yes'
    # Link test to assumption
    assumptionTests['goldfeld_quandt_passed'] = 'Homoscedasticity'
    formalNames['goldfeld_quandt_passed'] = 'Goldfeld Quandt Test'

    ###### ASSUMPTION IV - MULTICOLLINEARITY

    ### (6) Pairwise correlations < 0.7

    # If there are multiple predictors (IVs), find pairwise correlation between IVs
    if len(X.columns) > 1:
        pairwise_corr = X.corr()
        pairwise_corr = pairwise_corr[pairwise_corr != 1]  # make diagonals=nan
        # If pairwise correlations < 0.7, the test is passed and vice versa
        high_pairwise_corr = pairwise_corr[pairwise_corr >= 0.7]
        if high_pairwise_corr.isnull().all().all():
            diagnostics['high_pairwise_correlations_passed'] = 'Yes'
        else:
            diagnostics['high_pairwise_correlations_passed'] = 'No'
    # If only one predictor (IV), the test is passed (no correlation calculation)
    else:
        diagnostics['high_pairwise_correlations_passed'] = 'Yes'
    # Link test to assumption
    assumptionTests['high_pairwise_correlations_passed'] = 'Multicollinearity'
    formalNames['high_pairwise_correlations_passed'] = 'High Pairwise correlations'

    ### (7) Variance Inflation Factors (VIF) < 5

    # If there are multiple predictors (IVs), find VIFs between IVs
    if len(X.columns) > 1:
        X_withconst = add_constant(X)
        vif = pd.DataFrame()
        vif['VIF'] = [sm_stats.outliers_influence.variance_inflation_factor(
                X_withconst.values, i) for i in range(X_withconst.shape[1])]
        vif['features'] = X_withconst.columns

        # If no predictors have VIF > 5, the test is passed
        if ((vif['VIF'] < 5).all()):
            diagnostics['VIF_passed'] = 'Yes'
            diagnostics['VIF_predictorsFailed'] = []
        # If not, the test is not passed and predictor name is added to diagnostics
        else:
            diagnostics['VIF_passed'] = 'No'
            diagnostics['VIF_predictorsFailed'] = vif[vif['VIF'] > 5].to_string(
                    index=False, header=False)
    # If only one predictor (IV), the test is passed (no VIF calculation)
    else:
        diagnostics['VIF_passed'] = 'Yes'
    # Link test to assumption
    assumptionTests['VIF_passed'] = 'Multicollinearity'
    formalNames['VIF_passed'] = 'High Variance Inflation Factor'
    
    ###### ASSUMPTION V - OUTLIERS

    ### (8) Extreme Strandardised Residuals

    # Find outliers with std residuals above 3 or below -3
    highOutliers = influence_df[
            influence_df['standard_resid'] < -3].index.tolist()
    lowOutliers = influence_df[
            influence_df['standard_resid'] > 3].index.tolist()
    diagnostics['outlier_index'] = highOutliers + lowOutliers
    # If not found any outliers, the test is passed and vice versa
    if not diagnostics['outlier_index']:
        diagnostics['outliers_passed'] = 'Yes'
    else:
        diagnostics['outliers_passed'] = 'No'
    # Link test to assumption
    assumptionTests['outliers_passed'] = 'Outliers/Leverage/Influence'
    formalNames['outliers_passed'] = 'Extreme Standardised Residuals'

    ### (9) Cook's Distance (assessing influence)

    # Cooks cut-off for one predictor is 0.7, two is 0.8, and more is 0.85
    if len(X.columns) == 1:
        cooks_cutOff = 0.7
    elif X.shape[1] == 2:
        cooks_cutOff = 0.8
    elif X.shape[1] > 2:
        cooks_cutOff = 0.85
    # Find outliers with residuals larger than Cook's cut-off
    diagnostics['influence_largeCooksD_index'] = influence_df[
            influence_df['cooks_d'] > cooks_cutOff].index.tolist()
    # If not found any outliers, the test is passed and vice versa
    if not diagnostics['influence_largeCooksD_index']:
        diagnostics['influence_passed'] = 'Yes'
    else:
        diagnostics['influence_passed'] = 'No'
    # Link test to assumption
    assumptionTests['influence_passed'] = 'Outliers/Leverage/Influence'
    formalNames['influence_passed'] = "Large Cook's Distance"

    ###### ASSUMPTION VI - NORMALITY

    ### (10) Normal distribution of residuals

    # Check if mean is 0 (between -0.1 and 0.1) and errors normally distributed
    diagnostics['meanOfResiduals'] = model.resid.mean()
    if diagnostics['meanOfResiduals'] < .1 and diagnostics[
            'meanOfResiduals'] > -.1:
        diagnostics['meanOfResiduals_passed'] = 'Yes'
    else:
        diagnostics['meanOfResiduals_passed'] = 'No'
    # Link test to assumption
    assumptionTests['meanOfResiduals_passed'] = 'Normality'
    formalNames['meanOfResiduals_passed'] = "Mean of residuals not approx = 0"

    ### (11) Shapiro-Wilk test

    # Perform Shapiro-Wilk test for normality on residuals
    diagnostics['shapiroWilks_p'] = scipy.stats.shapiro(model.resid)[1]
    # If p>0.05, the test is passed
    if diagnostics['shapiroWilks_p'] > 0.05:
        diagnostics['shapiroWilks_passed'] = 'Yes'
    else:
        diagnostics['shapiroWilks_passed'] = 'No'
    # Link test to assumption
    assumptionTests['shapiroWilks_passed'] = 'Normality'
    formalNames['shapiroWilks_passed'] = 'Shapiro-Wilk Test'

    ###### SUMMARISE DIAGNOSTIC TEST INFORMATION

    # Check whether diagnostic tests are passed.
    # If all tests passed, then print message telling user that model is ok.
    # If any test failed, print message telling user that
    # model may not satisfy assumptions, check plots, and investigate further.

    diagnostic_tests = 0
    diagnosticsPassed = 0
    violated = []

    # Go through all the assumptions and tests
    if verbose==True: print('---\nDiagnostic tests - ' + step + '\n')

    for key in diagnostics:
        if key[-6:] == 'passed':
            diagnostic_tests += 1
            if diagnostics[key] == 'Yes':
                if verbose==True:
                    # Print the assumption and test which has passed
                    print(assumptionTests[key] + ' = PASSED (' +
                        formalNames[key] + ')')
                # Add the assumption to diagnostics passed
                diagnosticsPassed += 1
            else:
                if verbose==True:
                    # Print the assumption and test which has failed
                    print(assumptionTests[key] + ' = FAILED (' +
                        formalNames[key] + ')')
                # Add the assumption to possible violations
                violated.append(assumptionTests[key])
    if verbose==True: print(' ')

    # Summarise how many tests passed/failed for each assumption
    assumptionList = [i for i in assumptionTests.values()]
    assumptions = list(set(assumptionList))
    summaryTextList = []
    summarySentence = ' tests passed for assumption - '
    # For each assumption that didn't pass all tests, print warning
    for assumption in assumptions:
        testsFailed = violated.count(assumption)
        testsPerformed = assumptionList.count(assumption)
        testsPassed = testsPerformed - testsFailed
        sentence = str(testsPassed) + '/' + str(
                testsPerformed) + summarySentence + assumption
        if testsFailed > 0 and verbose==True:
            print("FURTHER INSPECTION REQUIRED -> " +
                sentence)
        summaryTextList.append(sentence)

    ###### SAVE DIAGNOSTICS INFO AND SUMMARY OF DIAGNOSTIC TESTS

    # Write out text file with summary of tests
    summaryFile = saveto + '/' + step + '_testSummary.txt'
    with open(summaryFile, 'w') as f:
        for item in summaryTextList:
            f.write("%s\n" % item)

    # Saves a CSV with 28 rows and two columns (long but easily readable)
    csvName = saveto + '/' + step + '_diagnostic_results.csv'
    pd.Series(diagnostics).to_csv(csvName)

    # Save a CSV of pairwise correlations (only if there are multiple predictors)
    if len(X.columns) > 1:
        pairwiseCorrName = saveto + '/' + step + '_pairwise_correlations.csv'
        high_pairwise_corr.to_csv(pairwiseCorrName)

    ###### MAKE AND SAVE PLOTS
    #29/03/2023 - 2 x 2 figure subplot removed and figures now saved individually

    ### PLOT 1 - STUDENTISED RESIDUALS VS FITTED VALUES 
    # Used to inspect linearity and homoscedasticity
    # Get values for the plot 1
    student_resid = influence_df['student_resid']
    fitted_vals = model.fittedvalues
    df_residfitted = pd.concat([student_resid, fitted_vals], axis=1)
    df_residfitted = df_residfitted.set_axis(['student_resid', 'fitted_vals'],
                                             axis=1)
    # Plot with a LOWESS (Locally Weighted Scatterplot Smoothing) line 
    # A relativelty straight LOWESS line indicates a linear model is reasonable
    fig, ax = plt.subplots()
    sns.residplot(ax=ax, data=df_residfitted,
            x='fitted_vals', y='student_resid', lowess=True,
            scatter_kws={'alpha': 0.8},
            line_kws={'color': 'red', 'lw': 1, 'alpha': 1})
    ax.set(ylim=(-3.5, 3.5))
    ax.set_title('Residuals vs Fitted')
    ax.set_xlabel('Fitted values')
    ax.set_ylabel('Studentised Residuals')

    figName = saveto + '/' + step + '_studresid_plots.png'
    plt.savefig(figName, dpi=300, bbox_inches="tight")

    ### PLOT 2 - NORMAL QQ PLOT OF RESIDUALS
    # Used to inspect normality
    fig, ax = plt.subplots()
    sm.qqplot(ax=ax, data=model.resid, fit=True, line='45')
    ax.set_title('Normal QQ Plot of Residuals')
    figName = saveto + '/' + step + '_QQ_plots.png'
    plt.savefig(figName, dpi=300, bbox_inches="tight")

    ### PLOT 3 - INFLUENCE PLOT WITH COOK'S DISTANCE
    # Used to inspect influence
    fig, ax = plt.subplots()
    sm.graphics.influence_plot(model, ax=ax, criterion="cooks")
    ax.set_title('Influence plot')
    ax.set_xlabel('H leverage')
    ax.set_ylabel('Studentised Residuals')
    figName = saveto + '/' + step + '_influence_plots.png'
    plt.savefig(figName, dpi=300, bbox_inches="tight")

    ### PLOT 4 - BOX PLOT OF STANDARDISED RESIDUALS
    # Used to inspect outliers (residuals)
    fig, ax = plt.subplots()
    outlier_fig = sns.boxplot(ax=ax, y=influence_df['standard_resid'])
    outlier_fig = sns.swarmplot(ax=ax, y=influence_df['standard_resid'], color="red")
    outlier_fig.axes.set(title = 'Boxplot of  Standardised Residuals', ylabel='Standardized Residuals', ylim=(-3.5, 3.5))
    residBoxplot = outlier_fig.get_figure()  # get figure to save
    figName = saveto + '/' + step + '_box_plots.png'
    plt.savefig(figName, dpi=300, bbox_inches="tight")
    plt.show()  

    ### PLOT 5 - HISTOGRAM OF STANDARDISED RESIDUAL
    plot = sns.histplot(data=influence_df, x='standard_resid', kde=True, bins=16) # you may select the no. of bins
    plot.set(title = 'Histogram of Standardised Residuals', xlabel='Standardized Residuals', ylabel='Frequency', xlim=(-3.5, 3.5))

    figName = saveto + '/' + step + '_histogram_plots.png'
    plt.savefig(figName, dpi=300, bbox_inches="tight")

    if showfig==True:
        plt.show()
    else:
        plt.close()
    
    ### PLOT 6 - PARTIAL REGRESSION PLOTS
    # Used to inspect linearity
    if len(X.columns) == 1:
        fig_partRegress = plt.figure(figsize=(15, 5))
    elif len(X.columns) <= 3:
        fig_partRegress = plt.figure(figsize=(15, 10))
    elif len(X.columns) <= 5:
        fig_partRegress = plt.figure(figsize=(15, 15))
    elif len(X.columns) > 5:
        fig_partRegress = plt.figure(figsize=(15, 20))
    sys.stdout = open(os.devnull, 'w') # disable print for following function
    fig_partRegress = sm.graphics.plot_partregress_grid(model, fig=fig_partRegress)
    sys.stdout = sys.__stdout__ 
    suptitle = fig_partRegress.suptitle('Partial Regression Plots for {} - {}'.format(step, X_names), y=1)
    fig_partRegress.tight_layout()
    figName = saveto + '/' + step + '_partial_regression_plots.png'
    fig_partRegress.savefig(figName, dpi=300, bbox_extra_artists=(suptitle,), bbox_inches="tight")
    if showfig==True:
        plt.show()
    else:
        plt.close()

    # Return list of assumptions that require further inspection (i.e. failed
    # at least 1 diagnostic test)
    assumptionsToCheck = list(set(violated))
    return assumptionsToCheck