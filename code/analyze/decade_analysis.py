import pandas as pd
from association_utils import get_correlations_by_decade_boostrap
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--hist", action="store_true")

# folders for reading and writing data
data_folder = "../../data/bootstrapped_data"
output_folder = "../../data/results/decade"

decades = ("70s", "80s", "90s")

# number of bootstrapping iterations
n_iter = 10000

if __name__ == "__main__":
    # determine whether to use the historical word embeddings.
    args = parser.parse_args()
    use_hist = args.hist

    if use_hist:
        embedding_types = ("Hist",)
    else:
        embedding_types = ("Word2Vec", "GloVe", "fastText")

    # create lists of all the bootstrapped dataframes
    all_dfs_cds = [] 
    all_dfs_cs = []
    for i in range(n_iter):
        df_cds = pd.read_csv(f"{data_folder}/df_cds_decade_bs{i}.csv")
        df_cs = pd.read_csv(f"{data_folder}/df_cs_decade_bs{i}.csv")
        all_dfs_cds.append(df_cds)
        all_dfs_cs.append(df_cs)

    # run analysis and save results without names
    df_results_by_decade = get_correlations_by_decade_boostrap(all_dfs_cds, all_dfs_cs, decades, embedding_types, include_names=False,
        hist_embeddings=use_hist)
    if use_hist:
        df_results_by_decade.to_csv(f"{output_folder}/bootstrapped_results_by_decade_nonames_hist.csv")
    else: 
        df_results_by_decade.to_csv(f"{output_folder}/bootstrapped_results_by_decade_nonames.csv")

    # run analysis and save results with names
    df_results_by_decade = get_correlations_by_decade_boostrap(all_dfs_cds, all_dfs_cs, decades, embedding_types, include_names=True,
        hist_embeddings=use_hist)
    if use_hist:
        df_results_by_decade.to_csv(f"{output_folder}/bootstrapped_results_by_decade_hist.csv")
    else:
        df_results_by_decade.to_csv(f"{output_folder}/bootstrapped_results_by_decade.csv")
