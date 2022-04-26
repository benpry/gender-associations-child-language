import pandas as pd
from association_utils import get_all_correlations_bootstrap

# folders for reading and writing data
data_folder = "../../data/bootstrapped_data"
output_folder = "../../data/results/aggregate"

# types of embeddings and association tests to use
embedding_types = ("Word2Vec", "GloVe", "fastText")
association_types = ("WEAT", "PROJ")

# number of bootstrapping iterations
n_iter = 10000

if __name__ == "__main__":
    # create lists of all the bootstrapped dataframes
    all_dfs_cds = [] 
    all_dfs_cs = []
    for i in range(n_iter):
        df_cds = pd.read_csv(f"{data_folder}/df_cds_agg_bs{i}.csv")
        df_cs = pd.read_csv(f"{data_folder}/df_cs_agg_bs{i}.csv")
        all_dfs_cds.append(df_cds)
        all_dfs_cs.append(df_cs)

    # version of the analysis that does not include names
    df_aggregate_results_nonames = get_all_correlations_bootstrap(all_dfs_cds, all_dfs_cs, embedding_types, association_types, include_names=False)
    df_aggregate_results_nonames.to_csv(f"{output_folder}/bootstrapped_aggregate_corrs_nonames.csv")
    # compute and save a results dataframe 
    df_aggregate_results = get_all_correlations_bootstrap(all_dfs_cds, all_dfs_cs, embedding_types, association_types, include_names=True)
    df_aggregate_results.to_csv(f"{output_folder}/bootstrapped_aggregate_corrs.csv")
