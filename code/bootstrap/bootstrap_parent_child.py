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
    np.random.seed(3904)

    # PARENT-CHILD ANALYSIS

    # read the parent-child data
    with open(raw_data_folder + 'm_m_cds.p', 'rb') as fp:
        m_m_cds = pickle.load(fp)
    with open(raw_data_folder + 'f_m_cds.p', 'rb') as fp:
        f_m_cds = pickle.load(fp)
    with open(raw_data_folder + 'm_f_cds.p', 'rb') as fp:
        m_f_cds = pickle.load(fp)
    with open(raw_data_folder + 'f_f_cds.p', 'rb') as fp:
        f_f_cds = pickle.load(fp)

    # get the total word count
    all_words = []
    for age in range(1,6):
        all_words += m_m_cds[str(age)]
        all_words += f_m_cds[str(age)]
        all_words += m_f_cds[str(age)]
        all_words += f_f_cds[str(age)]

    # compute the number of words to sample per age-gender pair
    downsample_n = int(len(all_words) / 20)
    print(f"parent-child bootstrap number: {downsample_n}")

    for iteration in tqdm(range(n_iter)):
        m_m_cds_ds = {}
        f_m_cds_ds = {}
        m_f_cds_ds = {}
        f_f_cds_ds = {}

        for age in range(1,6):
            m_m_cds_ds[str(age)] = np.random.choice(m_m_cds[str(age)], size=downsample_n, replace=True)
            f_m_cds_ds[str(age)] = np.random.choice(f_m_cds[str(age)], size=downsample_n, replace=True)
            m_f_cds_ds[str(age)] = np.random.choice(m_f_cds[str(age)], size=downsample_n, replace=True)
            f_f_cds_ds[str(age)] = np.random.choice(f_f_cds[str(age)], size=downsample_n, replace=True)

        # make dataframes for speech from mothers and speech from fathers
        df_mothers = make_parent_df(f_m_cds_ds, f_f_cds_ds)
        df_fathers = make_parent_df(m_m_cds_ds, m_f_cds_ds)

        # filter out the words that have fewer instances than the minimum frequency
        df_mothers = filter_min_freq(df_mothers)
        df_fathers = filter_min_freq(df_fathers)

        # save the data to csv
        df_mothers.to_csv(f"{output_folder}/df_mothers_bs{iteration}.csv")
        df_fathers.to_csv(f"{output_folder}/df_fathers_bs{iteration}.csv")

