import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu, kruskal
import statsmodels.api as sm
import numpy as np

def generate_violin(
        sensitive_features: pd.DataFrame,
        endpoint: pd.Series,
        output_dir: Path | str,
        show_figure: bool = True, 
        save_figure = True, 
        tag: str = '',
    ) -> None:
    """
    Generates violin plots to visualize the distribution of `endpoint` across `sensitive_features`.

    This function creates violin plots for each feature in `sensitive_features` to compare the 
    distribution of `endpoint` across different groups. The plots are saved in `output_dir`.

    Args:
        sensitive_features (pd.DataFrame): DataFrame containing categorical sensitive features.
        endpoint (pd.Series): Series representing the bias metric or outcome variable.
        output_dir (Path | str): Directory where plots should be saved.
        show_figure (bool, optional): Whether to display the figure. Defaults to True.
        save_figure (bool, optional): Whether to save the figure. Defaults to True.
        tag (str, optional): Optional tag for filename customization. Defaults to ''.

    Returns:
        None
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for feature_name, feature_series in sensitive_features.items():
        plt.figure(figsize=(8, 6)) 
        sns.set_theme(style="whitegrid")  

        sns.violinplot(
            x=feature_series, 
            y=endpoint, 
            # palette="muted",  
            inner="point", 
            linewidth=1.25 
        )

        # Perform statistical test based on number of unique groups
        unique_groups = feature_series.dropna().unique()

        if len(unique_groups) == 2:
            # Mann-Whitney U Test (for 2 groups)
            group_1 = endpoint[feature_series == unique_groups[0]]
            group_2 = endpoint[feature_series == unique_groups[1]]

            if len(group_1) > 1 and len(group_2) > 1:
                u_stat, p_value = mannwhitneyu(group_1, group_2, alternative='two-sided')
                test_text = f"p-value: {p_value:.4f}"
                if p_value < 0.05:
                    print(f"    Significant difference between {unique_groups[0]} and {unique_groups[1]}")
            else:
                test_text = ""

        elif len(unique_groups) > 2:
            # Kruskal-Wallis H Test (for 3+ groups)
            group_data = [endpoint[feature_series == group] for group in unique_groups]

            if all(len(g) > 1 for g in group_data):  # Ensure all groups have at least 2 values
                h_stat, p_value = kruskal(*group_data)
                test_text = f"p-value: {p_value:.4f}"
                if p_value < 0.05:
                    print(f"    Significant difference between {unique_groups[0]} and {unique_groups[1]}")
            else:
                test_text = ""

        else:
            test_text = ""

        # Add p-value as text on the plot
        plt.text(0.05, 0.95, test_text, transform=plt.gca().transAxes,
                 fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))


        title_tag = f" ({tag})" if tag else ""
        filename_tag = f"_{tag}" if tag else ""

        plt.title(f'{endpoint.name} Distribution by {feature_name} {title_tag}', fontsize=16, weight='bold')  
        plt.xlabel(f'{feature_name}', fontsize=14)  
        plt.ylabel(f'{endpoint.name} per Patient', fontsize=14) 
        plt.xticks(rotation=45, ha='right')

        # plt.ylim(0, 5000)

        plt.tight_layout()
        
        if save_figure:
            plt.savefig(output_dir / f'{feature_name}_vs_{endpoint.name}{filename_tag}.png')  
        if show_figure:
            plt.show()

def subgroup_analysis_OLS(data:pd.DataFrame, sensitive_features: str | list[str], bias_metric:np.ndarray) -> float:
    """Fit a statsmodels OLS model to the bias metric data, using the sensitive feature and print summary based on p_val."""
    one_hot_encoded = pd.get_dummies(data[sensitive_features], prefix=sensitive_features)
    X_columns = one_hot_encoded.columns

    X = one_hot_encoded.values  
    y = bias_metric  

    X = sm.add_constant(X.astype(float), has_constant='add')
    model = sm.OLS(y, X).fit()

    return model.f_pvalue
