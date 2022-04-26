import pickle
import numpy as np
from tqdm import tqdm
from processing_utils import make_aggregate_df

raw_data_folder = "../../data/switchboard_data"
output_folder = "../../data/switchboard_data/bootstrapped"

# the number of bootstrapped instances to create
n_iter = 10000

if __name__ == "__main__":
    
    # random seed for reproducibility
    np.random.seed(5432)


    with open(f"{raw_data_folder}/m_words.p", "rb") as fp:
        m_words = pickle.load(fp)
    with open(f"{raw_data_folder}/f_words.p", "rb") as fp:
        f_words = pickle.load(fp)

    downsample_n = (len(m_words) + len(f_words)) // 2

    for iteration in tqdm(range(n_iter)):

        m_words_i = np.random.choice(m_words, size=downsample_n, replace=True)
        f_words_i = np.random.choice(f_words, size=downsample_n, replace=True)

        df_i = make_aggregate_df(m_words_i, f_words_i)
        df_i.to_csv(f"{output_folder}/df_bs{iteration}.csv")
