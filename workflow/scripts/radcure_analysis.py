import pandas as pd
from utils import subgroup_analysis_OLS, generate_violin
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
from damply import dirs


def radcure_analysis():
    df = pd.read_csv('./data/radcure_nnunet_with_clinical.csv', index_col=0)
    df_sensitive = df[['OAR', 'Sex', 'Age', 'Ds Site', 'Stage', 'HPV', 'N', 'T', 'Chemo? ', 'Smoking Status', 'APL']]

    bins = [0, 40, 60, 80, float('inf')]
    labels = ["≤40", "41-60", "61-80", "80+"]

    # Apply categorization
    df_sensitive['Age'] = pd.cut(df_sensitive['Age'], bins=bins, labels=labels, right=True)

    columns_to_use = [col for col in df_sensitive.columns if not col in ["OAR", 'APL']]
    column_pairs = list(combinations(columns_to_use, 2))

    for column_pair in column_pairs:
        for oar in df_sensitive["OAR"].unique():
            _df = df_sensitive[df_sensitive["OAR"] == oar]
            metric = _df["APL"].to_numpy()

            # Create a DataFrame to store p-values
            pval_matrix = pd.DataFrame(index=columns_to_use, columns=columns_to_use, dtype=float)

            for col1, col2 in column_pairs:
                pval = subgroup_analysis_OLS(_df, [col1, col2], metric)
                pval_matrix.loc[col1, col2] = pval
                pval_matrix.loc[col2, col1] = pval  # Ensure symmetry

            # Convert to numeric (useful if any NaNs are present)
            pval_matrix = pval_matrix.apply(pd.to_numeric)
            mask = np.triu(np.ones_like(pval_matrix, dtype=bool)) # Keep only lower triangle
            np.fill_diagonal(mask, False)

            # Plot heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(pval_matrix, annot=True, cmap="coolwarm_r", fmt=".3f", linewidths=0.5, cbar=True, mask=mask)
            plt.title(f"P-value Heatmap for OAR: {oar}", fontsize=14, weight="bold")
            plt.xticks(rotation=45, ha="right")
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            # Save the figure
            plt.savefig(dirs.RESULTS / "radcure_analysis" / f"pval_heatmap_{oar}.png")

    for oar in df_sensitive["OAR"].unique():
        _df = df_sensitive[df_sensitive["OAR"] == oar]
        generate_violin(
            _df, 
            _df["APL"], 
            dirs.RESULTS / "radcure_analysis" / f"violin_{oar}.png",
            show_figure=True,
            save_figure=False,
        )

if __name__ == "__main__":
    radcure_analysis()