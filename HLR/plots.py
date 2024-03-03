import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

def create_subplot_residuals_vs_fitted(ax, model, level):
    influence = model.get_influence()
    student_resid = influence.resid_studentized_internal
    fitted_vals = model.fittedvalues
    df_residfitted = pd.DataFrame({'Studentized Residuals': student_resid, 'Fitted Values': fitted_vals})

    sns.residplot(ax=ax, data=df_residfitted, x='Fitted Values', y='Studentized Residuals', lowess=True,
                  scatter_kws={'alpha': 0.8}, line_kws={'color': 'red', 'lw': 1, 'alpha': 1})
    ax.set(ylim=(-3.5, 3.5))
    ax.set_title(f'Residuals vs Fitted (Model Level {level})')
    ax.set_xlabel('Fitted values')
    ax.set_ylabel('Studentized Residuals')

def create_subplot_qq_residuals(residuals, ax, level):
    sm.qqplot(residuals, line='45', fit=True, ax=ax)
    ax.set_title(f'Normal QQ Plot of Residuals (Model Level {level})')

def create_subplot_influence(model, ax, level):
    sm.graphics.influence_plot(model, ax=ax, criterion="cooks")
    ax.set_title(f'Influence Plot (Model Level {level})')
    ax.set_xlabel('H Leverage')
    ax.set_ylabel('Studentized Residuals')

def create_subplot_std_residuals(residuals, ax, level):
    sns.boxplot(y=residuals, ax=ax)
    sns.swarmplot(y=residuals, color="red", ax=ax)
    ax.set_title(f'Boxplot of Standardized Residuals (Model Level {level})')
    ax.set_ylabel('Standardized Residuals')
    ax.set_ylim(-3.5, 3.5)

def create_subplot_histogram_std_residuals(residuals, ax, level):
    sns.histplot(residuals, kde=True, bins=16, ax=ax)
    ax.set_title(f'Histogram of Standardized Residuals (Model Level {level})')
    ax.set_xlabel('Standardized Residuals')
    ax.set_ylabel('Frequency')
    ax.set_xlim(-3.5, 3.5)

def create_subplot_partial_regression(model, fig_size, level):
    fig = plt.figure(figsize=fig_size)
    sm.graphics.plot_partregress_grid(model, fig=fig)
    fig.suptitle(f'Partial Regression Plots (Model Level {level})', y=1)

    plt.tight_layout()
    plt.show()