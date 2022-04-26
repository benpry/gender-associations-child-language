import pandas as pd
from association_utils import get_all_correlations_by_class_bootstrap

# folders for reading and writing data
data_folder = "../../data/bootstrapped_data"
output_folder = "../../data/results/class"

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
        df_cds = pd.read_csv(f"{data_folder}/df_cds_class_bs{i}.csv")
        df_cs = pd.read_csv(f"{data_folder}/df_cs_class_bs{i}.csv")
        all_dfs_cds.append(df_cds)
        all_dfs_cs.append(df_cs)

    # analysis without names and without the conjunctive set of words
    df_results_by_class_nonames = get_all_correlations_by_class_bootstrap(all_dfs_cds, all_dfs_cs, embedding_types, association_types, include_names=False, use_conjset=False)
    df_results_by_class_nonames.to_csv(f"{output_folder}/bootstrapped_class_corrs_nonames.csv")

    # analysis with names and without using the conjunctive set of words
    df_results_by_class = get_all_correlations_by_class_bootstrap(all_dfs_cds, all_dfs_cs, embedding_types, association_types, include_names=True, use_conjset=False)
    df_results_by_class.to_csv(f"{output_folder}/bootstrapped_class_corrs.csv")

    # analysis with names and with the conjunctive set of words
    df_results_by_class_conjset = get_all_correlations_by_class_bootstrap(all_dfs_cds, all_dfs_cs, embedding_types, association_types, include_names=True, use_conjset=True)
    df_results_by_class_conjset.to_csv(f"{output_folder}/bootstrapped_class_corrs_conjset.csv")

    # analysis without names and with the conjunctive set of words
    df_results_by_class_nonames_conjset = get_all_correlations_by_class_bootstrap(all_dfs_cds, all_dfs_cs, embedding_types, association_types, include_names=False, use_conjset=True)
    df_results_by_class_nonames_conjset.to_csv(f"{output_folder}/bootstrapped_class_corrs_nonames_conjset.csv")
