# HLR - Hierarchical Linear Regression in Python

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7683809.svg)](https://doi.org/10.5281/zenodo.7683809) [![image](https://img.shields.io/pypi/v/HLR.svg)](https://pypi.python.org/pypi/HLR) [![CI testing](https://github.com/teanijarv/HLR/actions/workflows/testing.yml/badge.svg)](https://github.com/teanijarv/HLR/actions/workflows/testing.yml) [![Documentation Status](https://readthedocs.org/projects/hlr-hierarchical-linear-regression/badge/?version=latest)](https://hlr-hierarchical-linear-regression.readthedocs.io/en/latest/?version=latest)

HLR is a simple Python package for running hierarchical linear regression.

## Features
It is built to work with Pandas dataframes, uses SciPy, statsmodels and pingouin under the hood, and runs diagnostic tests for testing assumptions while plotting figures with matplotlib and seaborn.

## Installation
HLR is meant to be used with Python 3.x and has been tested on Python 3.9-3.12.

#### Dependencies
- [NumPy](https://numpy.org/)
- [SciPy](https://www.scipy.org/)
- [Pandas](https://pandas.pydata.org/)
- [statsmodels](https://www.statsmodels.org/)
- [pingouin](https://pingouin-stats.org/)
- [matplotlib](https://matplotlib.org/)
- [seaborn](https://seaborn.pydata.org/)

#### User installation
To install HLR, run this command in your terminal:

`pip install hlr`

This is the preferred method to install HLR, as it will always install the most recent stable release.

If you don’t have [pip](https://pip.pypa.io/) installed, this [Python installation guide](http://docs.python-guide.org/en/latest/starting/installation/) can guide you through the process.

## Usage

Importing the module and running hierarchical linear regression and summarising the results.

```python
import pandas as pd
from HLR import HierarchicalLinearRegression

# Example dataframe which includes some columns which are also mentioned below
nba = pd.read_csv('example/NBA_train.csv')

# Define the models for hierarchical regression including predictors for each model
X = {1: ['PTS'], 
     2: ['PTS', 'ORB'], 
     3: ['PTS', 'ORB', 'BLK']}

# Define the outcome variable
y = 'W'

# Initiate the HLR object
hreg = HierarchicalLinearRegression(df, X, y)

# Generate a summarised report as a dataframe which shows all linear regression models parameters and difference between the models
summary_report = hreg.summary()
display(summary_report)

# Run diagnostics on all the models (displayed output below only shows the first model)
hreg.diagnostics(verbose=True)

# Different plots
hreg.plot_studentized_residuals_vs_fitted()
hreg.plot_qq_residuals()
hreg.plot_influence()
hreg.plot_std_residuals()
hreg.plot_histogram_std_residuals()
hreg.plot_partial_regression()
```
Output:
|   | Model Level |                           Predictors | N (observations) | DF (residuals) | DF (model) | R-squared |   F-value |  P-value (F) |           SSE |     SSTO |  MSE (model) | MSE (residuals) | MSE (total) |                                        Beta coefs |                             P-values (beta coefs) |                       Failed assumptions (check!) | R-squared change | F-value change | P-value (F change) |
|---|-----:|-------------------------------------:|-----------------:|---------------:|-----------:|----------:|----------:|-------------:|--------------:|---------:|-------------:|----------------:|------------:|--------------------------------------------------:|--------------------------------------------------:|--------------------------------------------------:|-----------------:|---------------:|-------------------:|
| 0 |    1 |                             [PTS]	 |            835.0 |          833.0 |        1.0 |  0.089297 | 81.677748 | 1.099996e-18 | 123292.827686 | 135382.0 | 12089.172314 |      148.010597 |  162.328537 | {'Constant': -13.846261266053896, 'points': 0.... | {'Constant': 0.023091997486255577, 'points': 1... |                     [Homoscedasticity, Normality] |              NaN |            NaN |                NaN |
| 1 |    2 |         [PTS, ORB] |            835.0 |          832.0 |        2.0 |  0.168503 | 84.302598 | 4.591961e-34 | 112569.697267 | 135382.0 | 11406.151367 |      135.300117 |  162.328537 | {'Constant': -14.225561767669713, 'points': 0.... | {'Constant': 0.014660145903221372, 'points': 1... |                    [Normality, Multicollinearity] |         0.079206 |      79.254406 |       3.372595e-18 |
| 2 |    3 | [PTS, ORB, BLK] |            835.0 |          831.0 |        3.0 |  0.210012 | 73.638176 | 3.065838e-42 | 106950.174175 | 135382.0 |  9477.275275 |      128.700571 |  162.328537 | {'Constant': -21.997353037483723, 'points': 0.... | {'Constant': 0.00015712851466562279, 'points':... | [Normality, Multicollinearity, Outliers/Levera... |         0.041509 |      43.663545 |       6.962046e-11 |

```
Model Level 1 Diagnostics:
  Independence of residuals (Durbin-Watson test):
    DW stat: 1.9913212248708367
    Passed: True
  Linearity (Pearson r):
    PTS: {'Pearson r': 0.29882561440469596, 'p-value': 1.099996182226575e-18, 'Passed': True}
  Linearity (Rainbow test):
    Rainbow Stat: 0.9145095390107386
    p-value: 0.8189528030224006
    Passed: True
  Homoscedasticity (Breusch-Pagan test):
    Lagrange Stat: 5.183865793060617
    p-value: 0.022797547646224846
    Passed: False
  Homoscedasticity (Goldfeld-Quandt test):
    F-Stat: 1.0462467498084154
    p-value: 0.3225733517317874
    Passed: True
  Multicollinearity (pairwise correlations):
    Correlations: {}
    Passed: True
  Multicollinearity (Variance Inflation Factors):
    VIFs: {}
    Passed: True
  Outliers (extreme standardized residuals):
    Indices: []
    Passed: True
  Outliers (high Cooks distance):
    Indices: []
    Passed: True
  Normality (mean of residuals):
    Mean: 4.465782367986833e-14
    Passed: True
  Normality (Shapiro-Wilk test):
    SW Stat: 0.9873111844062805
    p-value: 1.2462886616049218e-06
    Passed: False

Model Level 2 Diagnostics:
...
```

![diagnostic_plot1](https://i.imgur.com/22kFc0F.jpeg)  |  ![diagnostic_plot2](https://i.imgur.com/j8l6qJs.png)
:-------------------------:|:-------------------------:

#### Documentation (WIP)
Docs is currently outdated - it currently displays the old version of the package. See the Usage above for all available functionality.
 <https://hlr-hierarchical-linear-regression.readthedocs.io>

## Citation
Please use Zenodo DOI for citing the package in your work.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7683809.svg)](https://doi.org/10.5281/zenodo.7683809)

#### Example

Toomas Erik Anijärv, & Rory Boyle. (2023). teanijarv/HLR: v0.2.0 (v0.2.0). Zenodo. https://doi.org/10.5281/zenodo.7683808
```
@software{toomas_erik_anijarv_2024_7683808,
  author       = {Toomas Erik Anijärv and
                  Rory Boyle},
  title        = {teanijarv/HLR: v0.2.0},
  month        = mar,
  year         = 2024,
  publisher    = {Zenodo},
  version      = {v0.2.0},
  doi          = {10.5281/zenodo.7683808},
  url          = {https://doi.org/10.5281/zenodo.7683808}
}
```

## Development
HLR was created by [Toomas Erik Anijärv](https://www.toomaserikanijarv.com) using original code by [Rory Boyle](https://github.com/rorytboyle). The package is maintained by Toomas during his spare time, thereby contributions are more than welcome!

This program is provided with no warranty of any kind and it is still under development. However, this code has been checked and validated against multiple same analyses conducted in SPSS.

#### To-do
Would be great if someone with more experience with packages would contribute with testing and the whole deployment process. Also, if someone would want to write documentation, that would be amazing.
- docs
- testing
- dict valus within df hard to read
- ability to change OLS parameters
- add t stats for coefficients
- give option for output only some columns not all

#### Contributors
[Toomas Erik Anijärv](https://github.com/teanijarv)
[Rory Boyle](https://github.com/rorytboyle)
[Jules Mitchell](https://github.com/JulesMitchell)
[Cate Scanlon](https://github.com/catescanlon)