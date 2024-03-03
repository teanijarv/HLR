=====
Usage
=====

Quick start guide to use hierarchical linear regression using HLR package.

Initialising the HLR object
---------------------------

Let's first fetch some data and initiate the HLR object. We'll use the `penguins` dataset from `seaborn` for our example.

.. code-block:: python

     import seaborn as sns
     import pandas as pd

     # Load the example penguins dataset
     df = sns.load_dataset('penguins')
     df.dropna(inplace=True)
     df = df[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']]

Initialising the HLR object and generating summary report
---------------------------------------------------------

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

Run diagnostics for testing assumptions
---------------------------------------

.. code-block:: python
     diagnostics_dict = hlr.diagnostics(verbose=True)

Plotting options for all model levels
-------------------------------------

.. code-block:: python
     hlr.plot_studentized_residuals_vs_fitted()
     hlr.plot_qq_residuals()
     hlr.plot_influence()
     hlr.plot_std_residuals()
     hlr.plot_histogram_std_residuals()
     hlr.plot_partial_regression()