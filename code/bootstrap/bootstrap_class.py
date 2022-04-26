import pandas as pd
import pickle
from tqdm import tqdm
import numpy as np
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
    np.random.seed(6345)
    # ANALYSIS BY CLASS

    # read data
    with open(raw_data_folder + 'm_words_cds_class.p', 'rb') as fp:
        m_words_cds = pickle.load(fp)
    with open(raw_data_folder + 'm_words_cs_class.p', 'rb') as fp:
        m_words_cs = pickle.load(fp)
    with open(raw_data_folder + 'f_words_cds_class.p', 'rb') as fp:
        f_words_cds = pickle.load(fp)
    with open(raw_data_folder + 'f_words_cs_class.p', 'rb') as fp:
        f_words_cs = pickle.load(fp)

    # get the word count
    all_words = []
    downsample_n_cds = 0
    downsample_n_cs = 0
    for cla in ('Black,WC', 'Black,UC', 'White,WC', 'White,UC'):
        all_words += m_words_cds[cla]
        all_words += f_words_cds[cla]
        all_words += m_words_cs[cla]
        all_words += f_words_cs[cla]
        downsample_n_cds += len(f_words_cds[cla]) + len(m_words_cds[cla])
        downsample_n_cs += len(f_words_cs[cla]) + len(m_words_cs[cla])


    # sample from each class-gender pair evenly
    downsample_n_cds = int(downsample_n_cds / 8)
    downsample_n_cs = int(downsample_n_cs / 8)
    print(f"class CDS bootstrap number: {downsample_n_cds}")    
    print(f"class CS bootstrap number: {downsample_n_cs}")

    for iteration in tqdm(range(n_iter)):
        m_cds_ds = defaultdict(dict)
        f_cds_ds = defaultdict(dict)
        m_cs_ds = defaultdict(dict)
        f_cs_ds = defaultdict(dict)
        
        for cla in ('Black,WC', 'Black,UC', 'White,WC', 'White,UC'):
            m_cds_ds[cla] = np.random.choice(m_words_cds[cla], size=downsample_n_cds, replace=True)
            f_cds_ds[cla] = np.random.choice(f_words_cds[cla], size=downsample_n_cds, replace=True)
            m_cs_ds[cla] = np.random.choice(m_words_cs[cla], size=downsample_n_cs, replace=True)
            f_cs_ds[cla] = np.random.choice(f_words_cs[cla], size=downsample_n_cs, replace=True)

        # make a dataframe by class
        df_cds = make_class_df(m_cds_ds, f_cds_ds)
        df_cs = make_class_df(m_cs_ds, f_cs_ds)

        # filter out the words that have fewer instances than the minimum frequency
        df_cds = filter_min_freq(df_cds)
        df_cs = filter_min_freq(df_cs)

        # save to csv
        df_cds.to_csv(f"{output_folder}/df_cds_class_bs{iteration}.csv")
        df_cs.to_csv(f"{output_folder}/df_cs_class_bs{iteration}.csv")