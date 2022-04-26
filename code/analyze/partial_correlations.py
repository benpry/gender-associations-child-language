"""
This file computes the correlations between gender probability and word embedding associations
controlling for psycholinguistic measures and not
"""

from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine as cos
from scipy.stats import pearsonr
import pingouin as pg
import gensim
import fasttext
from association_utils import word_weat
from gensim.models import KeyedVectors


# The filepaths for the Word2Vec, GloVe, and fastText embeddings
W2V_FILEPATH = "/hal9000/datasets/wordembeddings/google_news_w2v.bin"
GLOVE_FILEPATH = "/hal9000/datasets/wordembeddings/glove_w2v_format.txt"
FASTTEXT_FILEPATH = "/hal9000/datasets/wordembeddings/english_fasttext.bin"

# the folder for the whole corpus (i.e. no bootstrapping)
whole_corpus_folder = "../../data/whole_corpus/"
psycholing_data_folder = "../../data/psycholing_data/"

# Threshold for the number of word occurrences
THRESHOLD = 20

if __name__ == "__main__":
    # read data
    df_cds = pd.read_csv(whole_corpus_folder + "df_cds_agg.csv")
    df_cs = pd.read_csv(whole_corpus_folder + "df_cs_agg.csv").dropna()

    # read concreteness and valence data
    df_conc = pd.read_csv(psycholing_data_folder + "concretewords.csv")
    df_conc = df_conc[['Word', 'Conc.M']]
    df_conc.columns = ['word', 'conc']
    df_val = pd.read_csv(psycholing_data_folder + "valencewords.csv")
    df_val.columns = ['word', 'val']

    dfs = {"CS": df_cs, "CDS": df_cds}
    for embeddings in ("Word2Vec", "GloVe", "fastText"):
        # load the word embeddings
        if embeddings == 'Word2Vec':
            wv = KeyedVectors.load_word2vec_format(W2V_FILEPATH, binary=True)
            vocab = wv.vocab
        elif embeddings == 'GloVe':
            wv = KeyedVectors.load_word2vec_format(GLOVE_FILEPATH)
            vocab = wv.vocab
        else:
            wv = fasttext.load_model(FASTTEXT_FILEPATH)
            vocab = wv.get_words()

        corr_words = list(set(df_cds[df_cds['freq'] >= THRESHOLD]['word']) & set(vocab) & \
                          set(df_cs[df_cs['freq'] >= THRESHOLD]['word']))

        for speech_type in ("CDS", "CS"):
            df = dfs[speech_type]

            # drop the words not in corr_words
            df["drop"] = df["word"].apply(lambda x: x in corr_words)
            df = df[df["drop"] == True]

            # compute psycholinguistic data
            df['length'] = df['word'].apply(lambda x: len(x))
            df['log_freq'] = df.apply(lambda x: np.log(x['freq']), axis=1)
            df = pd.merge(df, df_conc, on='word', how='inner')
            df = pd.merge(df, df_val, on='word', how='inner')

            df['WEAT'] = [word_weat(w, wv) for w in df['word']]

            print(f"{embeddings}-{speech_type} full")
            print(pg.corr(df['WEAT'], df['pfgw']))
            print(f"{embeddings}-{speech_type} partial")
            print(pg.partial_corr(df, 'WEAT', 'pfgw', ['length', 'log_freq', 'conc', 'val']))
