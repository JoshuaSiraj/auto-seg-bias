"""
Code from https://github.com/pmcdi/jarvais/blob/main/src/jarvais/explainer/bias.py
"""
from typing import List

import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
from tabulate import tabulate

import numpy as np
import statsmodels.api as sm

def generate_violin(data: pd.DataFrame, sensitive_feature: str, bias_metric: np.ndarray, bias_metric_name) -> None:
    """Generate a violin plot for the bias metric."""
    plt.figure(figsize=(8, 6)) 
    sns.set_theme(style="whitegrid")  

    sns.violinplot(
        x=data[sensitive_feature], 
        y=bias_metric, 
        palette="muted",  
        inner="quart", 
        linewidth=1.25 
    )

    plt.title(f'{bias_metric_name.title()} Distribution by {sensitive_feature}', fontsize=16, weight='bold')  
    plt.xlabel(f'{sensitive_feature}', fontsize=14)  
    plt.ylabel(f'{bias_metric_name.title()} per Patient', fontsize=14) 
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()  
    plt.show()

def subgroup_analysis_OLS(data:pd.DataFrame, sensitive_features: str | List[str], bias_metric:np.ndarray) -> float:
    """Fit a statsmodels OLS model to the bias metric data, using the sensitive feature and print summary based on p_val."""
    one_hot_encoded = pd.get_dummies(data[sensitive_features], prefix=sensitive_features)
    X_columns = one_hot_encoded.columns

    X = one_hot_encoded.values  
    y = bias_metric  

    X = sm.add_constant(X.astype(float), has_constant='add')
    model = sm.OLS(y, X).fit()

    # if model.f_pvalue < 0.05:
    #     output = []

    #     print(f"⚠️  **Possible Bias Detected in {sensitive_features}** ⚠️\n")
    #     output.append(f"=== Subgroup Analysis for '{sensitive_features}' Using OLS Regression ===\n")

    #     output.append("Model Statistics:")
    #     output.append(f"    R-squared:                  {model.rsquared:.3f}")
    #     output.append(f"    F-statistic:                {model.fvalue:.3f}")
    #     output.append(f"    F-statistic p-value:        {model.f_pvalue:.4f}")
    #     output.append(f"    AIC:                        {model.aic:.2f}")
    #     output.append(f"    Log-Likelihood:             {model.llf:.2f}")

    #     summary_df = pd.DataFrame({
    #         'Feature': ['const'] + X_columns.tolist(),     # Predictor names (includes 'const' if added)
    #         'Coefficient': model.params,    # Coefficients
    #         'Standard Error': model.bse     # Standard Errors
    #     })
    #     table_output = tabulate(summary_df, headers='keys', tablefmt='fancy_grid', showindex=False, floatfmt=".3f")
    #     output.append("Model Coefficients:")
    #     output.append('\n'.join(['    ' + line for line in table_output.split('\n')]))

    #     output_text = '\n'.join(output)
    #     print(output_text)

    return model.f_pvalue
