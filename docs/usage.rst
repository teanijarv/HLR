=====
Usage
=====

See GitHub repository for the example dataset and Jupyter Notebook.

To use HLR - Hierarchical Linear Regression in a project::

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