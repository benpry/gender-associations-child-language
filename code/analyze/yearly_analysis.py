import pandas as pd
from association_utils import get_yearly_correlations_bootstrap
from gensim.models import KeyedVectors
import fasttext

# folders for reading and writing data
data_folder = "../../data/bootstrapped_data"
output_folder = "../../data/results/yearly"

# types of embeddings and association tests to use
embedding_types = ("Word2Vec", "GloVe", "fastText")

# number of bootstrapping iterations
n_iter = 10000

if __name__ == "__main__":
    # create lists of all the bootstrapped dataframes
    all_dfs_cds = [] 
    all_dfs_cs = []
    age_range = range(1,6)

    for i in range(n_iter):
        df_cds = pd.read_csv(f"{data_folder}/df_cds_yearly_bs{i}.csv")
        df_cs = pd.read_csv(f"{data_folder}/df_cs_yearly_bs{i}.csv")
        all_dfs_cds.append(df_cds)
        all_dfs_cs.append(df_cs)

    # yearly analysis without names
    df_yearly_results_nonames = get_yearly_correlations_bootstrap(all_dfs_cds, all_dfs_cs, embedding_types, age_range, include_names=False)
    df_yearly_results_nonames.to_csv(f"{output_folder}/bootstrapped_yearly_corrs_nonames.csv")

    # yearly analysis with names
    df_yearly_results = get_yearly_correlations_bootstrap(all_dfs_cds, all_dfs_cs, embedding_types, age_range, include_names=True)
    df_yearly_results.to_csv(f"{output_folder}/bootstrapped_yearly_corrs.csv")
