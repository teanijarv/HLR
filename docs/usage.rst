=====
Usage
=====

Quick start guide to use hierarchical linear regression using HLR package.

Fetch example data
------------------

Let's first fetch some data and initiate the HLR object. We'll use the `penguins` dataset from `seaborn` for our example.

.. code-block:: python

     import seaborn as sns
     import pandas as pd

     # Load the example penguins dataset
     df = sns.load_dataset('penguins')
     df.dropna(inplace=True)
     df = df[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']]

Initialize HLR & generate summary report
----------------------------------------

.. code-block:: python

     from HLR import HierarchicalLinearRegression

     # Define the independent variables for each model level
     ivs_dict = {
          1: ['bill_length_mm'],
          2: ['bill_length_mm', 'bill_depth_mm'],
          3: ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm']
     }

     # Define the dependent variable
     dv = 'body_mass_g'

     # Initialize the HierarchicalLinearRegression class
     hlr = HierarchicalLinearRegression(df, ivs_dict, dv)
     hlr.summary()

Output:

.. raw:: html
   :file: images/hlr_summary.html

Run diagnostics for testing assumptions
---------------------------------------

.. code-block:: python

     diagnostics_dict = hlr.diagnostics(verbose=True)

Output:

.. code-block:: text

     Model Level 1 Diagnostics:
     Independence of residuals (Durbin-Watson test):
     DW stat: 0.8450671190941991
     Passed: False
     Linearity (Pearson r):
     bill_length_mm: {'Pearson r': 0.5894511101769488, 'p-value': 1.5386135144860176e-32, 'Passed': True}
     Linearity (Rainbow test):
     Rainbow Stat: 0.845825915500362
     p-value: 0.8589217163587981
     Passed: True
     Homoscedasticity (Breusch-Pagan test):
     Lagrange Stat: 76.51043993569607
     p-value: 2.1905189444330245e-18
     Passed: False
     Homoscedasticity (Goldfeld-Quandt test):
     F-Stat: 3.298385120028286
     p-value: 5.1841847326260096e-14
     Passed: False
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
     Mean: -2.403469482162693e-13
     Passed: True
     Normality (Shapiro-Wilk test):
     SW Stat: 0.9912192354166119
     p-value: 0.04492289320888261
     Passed: False

     Model Level 2 Diagnostics:
     ...

Plotting options for all model levels
-------------------------------------

.. code-block:: python

     hlr.plot_studentized_residuals_vs_fitted()

Output:

.. image:: /images/plot_studentized_residuals_vs_fitted.png
   :alt: plot_studentized_residuals_vs_fitted
   :align: center
   :width: 50%

.. code-block:: python

     hlr.plot_qq_residuals()

Output:

.. image:: /images/plot_qq_residuals.png
   :alt: plot_qq_residuals
   :align: center
   :width: 50%

.. code-block:: python

     hlr.plot_influence()

Output:

.. image:: /images/plot_influence.png
   :alt: plot_influence
   :align: center
   :width: 50%

.. code-block:: python

     hlr.plot_std_residuals()

Output:

.. image:: /images/plot_std_residuals.png
   :alt: plot_std_residuals
   :align: center
   :width: 50%

.. code-block:: python

     hlr.plot_histogram_std_residuals()

Output:

.. image:: /images/plot_histogram_std_residuals.png
   :alt: plot_histogram_std_residuals
   :align: center
   :width: 50%

.. code-block:: python

     hlr.plot_partial_regression()

Output:

.. image:: /images/plot_partial_regression.png
   :alt: plot_partial_regression
   :align: center
   :width: 50%

(only Model Level 1 displayed, but actual output would plot all levels)