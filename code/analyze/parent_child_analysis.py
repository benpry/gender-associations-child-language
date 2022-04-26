import pandas as pd
from association_utils import get_parent_child_results_bootstrap

# folders for reading and writing data
data_folder = "../../data/bootstrapped_data"
output_folder = "../../data/results/parent_child"

# types of embeddings and association tests to use
embedding_types = ("Word2Vec", "GloVe", "fastText")
association_types = ("WEAT", "PROJ")

# number of bootstrapping iterations
n_iter = 10000

if __name__ == "__main__":
    # create lists of all the bootstrapped dataframes
    all_dfs_fathers = []
    all_dfs_mothers = [] 
    for i in range(n_iter):
        df_fathers = pd.read_csv(f"{data_folder}/df_fathers_bs{i}.csv")
        df_mothers = pd.read_csv(f"{data_folder}/df_mothers_bs{i}.csv")
        all_dfs_fathers.append(df_fathers)
        all_dfs_mothers.append(df_mothers)

    # run analysis and save results without names
    df_yearly_results = get_parent_child_results_bootstrap(all_dfs_fathers, all_dfs_mothers, embedding_types, include_names=False)
    df_yearly_results.to_csv(f"{output_folder}/parent_child_corrs_nonames.csv")

    # run analysis and save results with names
    df_yearly_results = get_parent_child_results_bootstrap(all_dfs_fathers, all_dfs_mothers, embedding_types, include_names=True)
    df_yearly_results.to_csv(f"{output_folder}/parent_child_corrs.csv")
