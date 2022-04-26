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
    np.random.seed(3937)

    # AGGREGATE AND YEARLY ANALYSIS

    with open(raw_data_folder + 'm_words_cds.p', 'rb') as fp:
        m_words_cds = pickle.load(fp)
    with open(raw_data_folder + 'f_words_cds.p', 'rb') as fp:
        f_words_cds = pickle.load(fp)
    with open(raw_data_folder + 'm_words_cs.p', 'rb') as fp:
        m_words_cs = pickle.load(fp)
    with open(raw_data_folder + 'f_words_cs.p', 'rb') as fp:
        f_words_cs = pickle.load(fp)

    # get total word count
    all_words = []
    downsample_n_cds = 0
    downsample_n_cs = 0
    for age in range(1,6):
        all_words += m_words_cds[str(age)]
        all_words += f_words_cds[str(age)]
        all_words += m_words_cs[str(age)]
        all_words += f_words_cs[str(age)]
        downsample_n_cds += len(f_words_cds[str(age)]) + len(m_words_cds[str(age)])
        downsample_n_cs += len(f_words_cs[str(age)]) + len(m_words_cs[str(age)])

    # compute the number of words to include per age-gender pair
    downsample_n_cds = int(downsample_n_cds / 10)
    downsample_n_cs = int(downsample_n_cs / 10)
    print(f"CDS downsample number: {downsample_n_cds}")
    print(f"CS downsample number: {downsample_n_cs}")

    for iteration in tqdm(range(n_iter)):
        m_cds_ds = {}
        f_cds_ds = {}
        m_cs_ds = {}
        f_cs_ds = {}
        
        # downsample equally for each age-gender pair
        for age in range(1,6):
            m_cds_ds[str(age)] = np.random.choice(m_words_cds[str(age)], size=downsample_n_cds, replace=True)
            f_cds_ds[str(age)] = np.random.choice(f_words_cds[str(age)], size=downsample_n_cds, replace=True)
            m_cs_ds[str(age)] = np.random.choice(m_words_cs[str(age)], size=downsample_n_cs, replace=True)
            f_cs_ds[str(age)] = np.random.choice(f_words_cs[str(age)], size=downsample_n_cs, replace=True)

        # compute yearly word statistics
        df_cds_yearly = make_yearly_df(m_cds_ds, f_cds_ds)
        df_cs_yearly = make_yearly_df(m_cs_ds, f_cs_ds)

        # compute aggregate word statistics
        df_cds_agg = make_aggregate_df(m_cds_ds, f_cds_ds)
        df_cs_agg = make_aggregate_df(m_cs_ds, f_cs_ds)

        # filter out the words that have fewer instances than the minimum frequency
        df_cds_yearly = filter_min_freq(df_cds_yearly)
        df_cs_yearly = filter_min_freq(df_cs_yearly)
        df_cds_agg = filter_min_freq(df_cds_agg)
        df_cs_agg = filter_min_freq(df_cs_agg)

        # save data to csv
        df_cds_yearly.to_csv(f'{output_folder}/df_cds_yearly_bs{iteration}.csv')
        df_cs_yearly.to_csv(f'{output_folder}/df_cs_yearly_bs{iteration}.csv')
        df_cds_agg.to_csv(f'{output_folder}/df_cds_agg_bs{iteration}.csv')
        df_cs_agg.to_csv(f'{output_folder}/df_cs_agg_bs{iteration}.csv')
