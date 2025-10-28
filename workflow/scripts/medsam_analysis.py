from numpy.__config__ import show
import pandas as pd
import numpy as np
from damply import dirs
from utils import generate_violin

def radcure_analysis(
    organ_at_risk: str,
):
    print(f"Performing auto-seg bias analysis for {organ_at_risk}\n")

    df = pd.read_csv(dirs.RAWDATA / "medsam2_inference" / f"radcure_{organ_at_risk}.csv")
    clinical_data = pd.read_csv(dirs.RAWDATA / "RADCURE_Clinical_v04_20241219.csv")

    df = df.merge(clinical_data, left_on='ID', right_on='patient_id', how='inner')

    df_sensitive = df[[ 'Sex', 'Age', 'Ds Site', 'Stage', 'HPV', 'N', 'T', 'Smoking Status']]

    bins = [0, 40, 60, 80, float('inf')]
    labels = ["≤40", "41-60", "61-80", "80+"]

    # Apply categorization
    df_sensitive = df_sensitive.rename(columns={'age at dx': 'Age'})

    df_sensitive['Age'] = pd.cut(df_sensitive['Age'], bins=bins, labels=labels, right=True)

    generate_violin(
        df_sensitive,
        df['Added_Path_Length'],
        dirs.RESULTS / 'medsam2_inference' / f'radcure_{organ_at_risk}_figures',
        show_figure=False,
    ) 

if __name__ == "__main__":
    radcure_analysis('brainstem')
    radcure_analysis('mandible')
    radcure_analysis('brachialplex_l')
    radcure_analysis('brachialplex_r')
    radcure_analysis("cochlea_l")
    radcure_analysis("cochlea_r")
    radcure_analysis("esophagus")
    radcure_analysis("eye_l")
    radcure_analysis("eye_r")
    radcure_analysis("larynx")
    radcure_analysis("lens_l")
    radcure_analysis("lens_r")
    radcure_analysis("lips")
    radcure_analysis("nrv_optic_l")
    radcure_analysis("nrv_optic_r")
    radcure_analysis("opticchiasm")
    radcure_analysis("parotid_l")
    radcure_analysis("parotid_r")
    radcure_analysis("spinalcord")

    # radcure_analysis("gtvp")


