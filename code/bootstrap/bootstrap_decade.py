import pandas as pd
import pickle
import numpy as np
from tqdm import tqdm
import os, sys, inspect
from collections import defaultdict
from processing_utils import make_aggregate_df, make_yearly_df, make_class_df, make_df_by_decade, make_parent_df

MIN_FREQ = 20
# folders for reading and writing data
raw_data_folder = '../../data/raw_data/'
output_folder = '../../data/bootstrapped_data/'

# number of bootstrapped instances to create
n_iter = 10000

def filter_min_freq(df):
    return df[df["freq"] >= MIN_FREQ].reset_index(drop=True)

if __name__ == "__main__":
    # random seed for reproducibility
    np.random.seed(7403)

    # ANALYSIS BY DECADE

    # read data
    with open(raw_data_folder + 'm_words_cds_bydecade.p', 'rb') as fp:
        m_words_cds = pickle.load(fp)
    with open(raw_data_folder + 'f_words_cds_bydecade.p', 'rb') as fp:
        f_words_cds = pickle.load(fp)
    with open(raw_data_folder + 'm_words_cs_bydecade.p', 'rb') as fp:
        m_words_cs = pickle.load(fp)
    with open(raw_data_folder + 'f_words_cs_bydecade.p', 'rb') as fp:
        f_words_cs = pickle.load(fp)

    # get total word count
    all_words = []
    downsample_n_cds = 0
    downsample_n_cs = 0
    for decade in ("70s", "80s", "90s"):
        all_words += m_words_cds[decade]
        all_words += f_words_cds[decade]
        all_words += m_words_cs[decade]
        all_words += f_words_cs[decade]
        downsample_n_cds += len(f_words_cds[decade]) + len(m_words_cds[decade])
        downsample_n_cs += len(f_words_cs[decade]) + len(m_words_cs[decade])

    # compute number of words per decade-gender pair
    downsample_n_cds = int(downsample_n_cds / 6)
    downsample_n_cs = int(downsample_n_cs / 6)
    print(f"decade CDS downsample number: {downsample_n_cds}")
    print(f"decade CS downsample number: {downsample_n_cs}")

    for iteration in tqdm(range(n_iter)):
        m_cds_ds = {}
        f_cds_ds = {}
        m_cs_ds = {}
        f_cs_ds = {}
        
        # downsample equally for each decade-age-gender pair
        for decade in ("70s", "80s", "90s"):
            m_cds_ds[decade] = np.random.choice(m_words_cds[decade], size=downsample_n_cds, replace=True)
            f_cds_ds[decade] = np.random.choice(f_words_cds[decade], size=downsample_n_cds, replace=True)
            m_cs_ds[decade] = np.random.choice(m_words_cs[decade], size=downsample_n_cs, replace=True)
            f_cs_ds[decade] = np.random.choice(f_words_cs[decade], size=downsample_n_cs, replace=True)

        # compute aggregate statistics per decade
        df_cds = make_df_by_decade(m_cds_ds, f_cds_ds)
        df_cs = make_df_by_decade(m_cs_ds, f_cs_ds)

        # filter out the words that have fewer instances than the minimum frequency
        df_cds = filter_min_freq(df_cds)
        df_cs = filter_min_freq(df_cs)

        # save the data
        df_cds.to_csv(f'{output_folder}/df_cds_decade_bs{iteration}.csv')
        df_cs.to_csv(f'{output_folder}/df_cs_decade_bs{iteration}.csv')
