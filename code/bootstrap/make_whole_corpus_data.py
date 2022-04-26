"""
This file computes statistics for the entire corpus without bootstrapping.

This file is used for the psycholinguistic correlates and linear regression analysis
"""
import pickle
from processing_utils import make_aggregate_df

# folders for reading and writing data
raw_data_folder = '../../data/raw_data/'
output_folder = '../../data/whole_corpus/'

if __name__ == "__main__":
    # AGGREGATE ANALYSIS

    with open(raw_data_folder + 'm_words_cds.p', 'rb') as fp:
        m_words_cds = pickle.load(fp)
    with open(raw_data_folder + 'f_words_cds.p', 'rb') as fp:
        f_words_cds = pickle.load(fp)
    with open(raw_data_folder + 'm_words_cs.p', 'rb') as fp:
        m_words_cs = pickle.load(fp)
    with open(raw_data_folder + 'f_words_cs.p', 'rb') as fp:
        f_words_cs = pickle.load(fp)

    # compute aggregate word statistics
    df_cds_agg = make_aggregate_df(m_words_cds, f_words_cds)
    df_cs_agg = make_aggregate_df(m_words_cs, f_words_cs)

    df_cds_agg.to_csv(f'{output_folder}/df_cds_agg.csv')
    df_cs_agg.to_csv(f'{output_folder}/df_cs_agg.csv')
