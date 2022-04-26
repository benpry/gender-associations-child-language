import pandas as pd
from association_utils import get_all_correlations_bootstrap

# folders for reading and writing data
data_folder = "../../data/santa_barbara_data/bootstrapped"
output_folder = "../../data/results/santa_barbara"

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
        df_cs = pd.read_csv(f"{data_folder}/df_santabarbara_bs{i}.csv")
        df_cds = pd.DataFrame(columns=df_cs.columns) # empty dataframe
        all_dfs_cds.append(df_cds)
        all_dfs_cs.append(df_cs)

    # compute results with name and save them to csv
    df_aggregate_results_nonames = get_all_correlations_bootstrap(all_dfs_cds, all_dfs_cs, embedding_types, association_types, include_names=False)
    df_aggregate_results_nonames.to_csv(f"{output_folder}/bootstrapped_corrs_santabarbara_nonames.csv")

    # compute results without name and save them to csv
    df_aggregate_results = get_all_correlations_bootstrap(all_dfs_cds, all_dfs_cs, embedding_types, association_types, include_names=True)
    df_aggregate_results.to_csv(f"{output_folder}/bootstrapped_corrs_santabarbara.csv")
