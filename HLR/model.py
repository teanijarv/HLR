"""HLR - base object which defines the model.

Authors
-------
Toomas Erik Anijärv toomaserikanijarv@gmail.com github.com/teanijarv
Rory Boyle rorytboyle@gmail.com github.com/rorytboyle
"""

import numpy as np
import pandas as pd
import scipy as scipy

from HLR import hierarchical_regression

class HLR_model():
    """Run Hierarchical Linear Regression (HLR) with optionally diagnostics tests for
    Independence of Residuals, Linearity, Homoscedasticity, Multicollinearity,
    Outliers/Influence, and Normality.

    Parameters
    ----------
    diagnostics : bool, optional, default: True
        Whether to run diagnostic tests for assumptions or not.
    save_folder : string, optional, default: 'results'
        Folder location in root where to save the results (and diagnostics).
    showfig : bool, optional, default: False
        Whether to show diagnostics plots or not.
    verbose : bool, optional, default: True
        Whether to print diagnostic log.
    
    Attributes
    ----------
    has_data : bool
        Whether data is loaded to the object.
    has_model : bool
        Whether model results are available in the object.
    n_steps : int
        The number of steps in the HLR model.

    """
    def __init__(self, diagnostics=True, save_folder='results', showfig=False, verbose=True):
        """Initialize object with main settings set to default values"""
        self.diagnostics = diagnostics
        self.save_folder = save_folder
        self.showfig = showfig
        self.verbose = verbose

        # Reset the data and results
        self._reset_data_results(True, True)
    
    @property
    def has_data(self):
        """Indicator for if the object contains data."""

        return True if self.X else False
    
    @property
    def has_results(self):
        """Indicator for if the object contains model results."""

        return True if np.all(self.model_results) else False # needs testing
    
    @property
    def n_steps(self):
        """Indicator for how many steps was there in the model."""

        return len(self.X) if self.has_results else None
    
    def _reset_data_results(self, clear_data=False, clear_results=False):
        """Reset the input data and model results to empty.
        
        Parameters
        ----------
        clear_data : bool, optional, default: False
            Whether to clear data attributes.
        clear_results : bool, optional, default: False
            Whether to clear model results attributes.
        """
        if clear_data:
            self.X = None
            self.X_names = None
            self.y = None
        if clear_results:
            self.model_results = None
            self.reg_models = None
        
    def add_data(self, X, X_names, y):
        """Add data (predictors and outcome variable) to the current object.

        Parameters
        ----------
        X : list of Pandas dataframes
            Each list element containing dataframe of predictors in each step.
        X_names : list of lists containing strings
            Corresponding predictor variable names for each step.
        y : Pandas dataframe
            Outcome variable
        """
        # If already have data, then reset first
        if self.X:
            self._reset_data_results(True, True)
        
        # Check if the inputted data matches the required conditions
        self.X, self.X_names, self.y = \
            self._prepare_data(X, X_names, y)
    
    def run(self, X, X_names, y):
        """Run the hierarchical linear regression with an option to run
        diagnostics (i.e., testing for assumptions).

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
        model_results : Pandas dataframe
            Results from the hierarchical regression model.
        reg_models : list of statsmodels' RegressionResultsWrapper objects
            Model objects for each step in a list.
        """
        # If data given, add data to the class
        if X is not None and X_names is not None and y is not None:
            self.add_data(X, X_names, y)
        
        # If data not given, cannot proceed
        if not self.has_data:
            raise ValueError("No data available to run, cannot proceed.")
        
        self.model_results, self.reg_models = \
            hierarchical_regression(X, X_names, y, diagnostics=self.diagnostics,
                                    save_folder=self.save_folder,
                                    showfig=self.showfig, verbose=self.verbose)
        
        return self.model_results, self.reg_models
    
    def save_results(self, filename='hlr_model_results', show_results=False):
        """Save the results as an Excel file (and optionally display).

        Parameters
        ----------
        filename : string, optional, default: 'hlr_model_results'
            Name for the exported Excel file.
        show_results : bool, optional, default: False
            Display the results table
        """
        if not self.has_results:
            raise ValueError("Do not have results to export.")
        if show_results==True:
            print(self.model_results)
        
        self.model_results.to_excel(self.save_folder+'/'+filename+'.xlsx')

    @staticmethod
    def _prepare_data(X, X_names, y):
        """Check if the input data matches the data requirements.

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
        X : list of Pandas dataframes
            Each list element containing dataframe of predictors in each step.
        X_names : list of lists containing strings
            Corresponding predictor variable names for each step.
        y : Pandas dataframe
            Outcome variable

        Raises
        ------
        ValueError
            If the input variables are not in correct format.
        """

        # Check that X and X_names are the right datatypes (lists)
        if not isinstance(X, list) or not isinstance(X_names, list):
            raise ValueError("X and X_names must be in lists.")
        
        # Check if X and X_names are the same length
        if len(X) != len(X_names):
            raise ValueError("X and X_names must be equal length.")
        
        # Check that y is the right datatype (dataframe)
        if not isinstance(y, pd.core.frame.DataFrame):
            raise ValueError("y must be a dataframe.")

        # Check if list X consists of dataframes and their lengths match y
        for s, step in enumerate(X):
            if not isinstance(step, pd.core.frame.DataFrame):
                raise ValueError("X at index {} is not a dataframe.".format(s))
            if step.shape[0] != y.shape[0]:
                raise ValueError("Dataframes X[{}] and y are not the same length.")

        # Check if list X_names consists of lists which have strings in them
        for s, step in enumerate(X_names):
            if not isinstance(step, list):
                raise ValueError("X_names at index {} is not a list.".format(s))
            for j, string in enumerate(step):
                if not isinstance(string, str):
                    raise ValueError("X_names[{}] does not contain string at index {}".format(s, j))

        return X, X_names, y