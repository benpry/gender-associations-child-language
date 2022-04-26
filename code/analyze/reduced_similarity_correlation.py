import numpy as np
import pandas as pd
from association_utils import compute_similarity_correlations

n_runs = 10000
embedding_types = ("Word2Vec", "GloVe", "fastText")
OUTPUT_FOLDER = "../../data/results"
CS_FILEPATH = "../../data/whole_corpus/df_cs_agg.csv"
words_to_exclude = {"uhoh", "uhuh", "mhm"}

if __name__ == "__main__":

    # read in the child speech data
    df_cs = pd.read_csv(CS_FILEPATH)

    # select the 60 words to be used for the t-SNE plot
    df_cs = df_cs[df_cs["word"].apply(lambda x: str(x) == str(x).lower() and str(x).isalpha() and str(x) not in words_to_exclude)]
    df_cs = df_cs[df_cs["freq"] >= 500].sort_values(by="pfgw")
    words_to_track = list(df_cs.head(30)["word"]) + list(df_cs.tail(30)["word"])

    df_corrs = compute_similarity_correlations(words_to_track, embedding_types, n_runs=n_runs)
    df_corrs.to_csv(f"{OUTPUT_FOLDER}/similarity_correlations.csv")
