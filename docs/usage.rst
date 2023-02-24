=====
Usage
=====

See GitHub repository for the example dataset and Jupyter Notebook.

To use HLR - Hierarchical Linear Regression in a project::

    import pandas as pd
    import os
    import HLR

    nba = pd.read_csv('example/NBA_train.csv')

    # List of dataframes of predictor variables for each step
    X = [nba[['PTS']],
        nba[['PTS', 'ORB']],
        nba[['PTS', 'ORB', 'BLK']]]

    # List of predictor variable names for each step
    X_names = [['points'],
            ['points', 'offensive_rebounds'], 
            ['points', 'offensive_rebounds', 'blocks']]

    # Outcome variable as dataframe
    y = nba[['W']]

    model = HLR.HLR_model(diagnostics=True, showfig=True, save_folder='results', verbose=True)
    model_results, reg_models = model.run(X=X, X_names=X_names, y=y)
    model.save_results(filename='nba_results', show_results=True)