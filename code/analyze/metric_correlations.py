"""
This file computes the correlation strength of different metrics measuring gender probabilty in speech with each other
"""
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import pearsonr

data_folder = "../../data/bootstrapped_data"
n_iter = 10000
MIN_FREQ = 20

def analyze_correlation(lst):
    mean = np.mean(lst)
    p_val = len([x for x in lst if x <= 0]) / len(lst)

    return f"{mean}, p={p_val}"

if __name__ == "__main__":

    gp_OR_corrs = []
    gp_LOR_corrs = []
    OR_LOR_corrs = []

    for i in tqdm(range(n_iter)):

        df_cds = pd.read_csv(f"{data_folder}/df_cds_agg_bs{i}.csv")
        df_cs = pd.read_csv(f"{data_folder}/df_cs_agg_bs{i}.csv")

        df_cds = df_cds[df_cds["freq"] >= MIN_FREQ]
        df_cs = df_cs[df_cs["freq"] >= MIN_FREQ]

        for df in (df_cds, df_cs):
            gp_OR_corrs.append(pearsonr(df["pfgw"], df["OR"])[0])
            gp_LOR_corrs.append(pearsonr(df["pfgw"], df["LOR"])[0])
            OR_LOR_corrs.append(pearsonr(df["OR"], df["LOR"])[0])

    print(f"pfgw-OR: {analyze_correlation(gp_OR_corrs)}")
    print(f"pfgw-LOR: {analyze_correlation(gp_LOR_corrs)}")
    print(f"OR-LOR: {analyze_correlation(OR_LOR_corrs)}")
