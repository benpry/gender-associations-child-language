"""
This file runs linear regression analysis on the entire corpus to predict gender probability using word embedding
associations and other measures
"""
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine as cos
import statsmodels.api as sm


def compute_effect_size(w, wv, A, B):
    return np.mean([cos(wv[w], wv[a]) for a in A]) - np.mean([cos(wv[w], wv[b]) for b in B])


def word_weat(w, wv):
    m_words = ['brother', 'father', 'uncle', 'grandfather', 'son', 'he', 'his', 'him']
    f_words = ['sister', 'mother', 'aunt', 'grandmother', 'daughter', 'she', 'hers', 'her']
    for word in m_words + f_words:
        if word not in wv:
            print(word, 'not in wv')
    if w not in wv:
        print(w, 'not in wv')

    return compute_effect_size(w, wv, m_words, f_words) / (np.std([cos(wv[w], wv[x]) for x in m_words + f_words]))


# the folder for the whole corpus (i.e. no bootstrapping)
whole_corpus_folder = "../../data/whole_corpus/"
psycholing_data_folder = "../../data/psycholing_data/"

if __name__ == "__main__":
    # read data
    df_cds = pd.read_csv(whole_corpus_folder + "df_cds_agg.csv").dropna()
    df_cs = pd.read_csv(whole_corpus_folder + "df_cs_agg.csv").dropna()

    # read concreteness and valence data
    df_conc = pd.read_csv(psycholing_data_folder + "concretewords.csv")
    df_conc = df_conc[['Word', 'Conc.M']]
    df_conc.columns = ['word', 'conc']
    df_val = pd.read_csv(psycholing_data_folder + "valencewords.csv")
    df_val.columns = ['word', 'val']

    dfs = {"CS": df_cs, "CDS": df_cds}

    for speech_type in ("CDS", "CS"):
        df = dfs[speech_type]

        # add psycholinguistic variables
        df['length'] = df['word'].apply(lambda x: len(x))
        df['log_freq'] = df.apply(lambda x: np.log(x['freq']), axis=1)
        df = pd.merge(df, df_conc, on='word', how='inner')
        df = pd.merge(df, df_val, on='word', how='inner')

        # fit a linear regression with the psycholinguistic variables
        lr = sm.OLS(df['pfgw'], df[['length', 'log_freq', 'conc', 'val']])
        res = lr.fit()
        print(speech_type)
        print(res.summary())
