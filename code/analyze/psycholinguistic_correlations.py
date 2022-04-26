import pandas as pd
import numpy as np
from scipy.stats import pearsonr

whole_corpus_folder = "../../data/whole_corpus/"
psycholing_data_folder = "../../data/psycholing_data/"

THRESHOLD = 20

if __name__ == "__main__":
    df_cds = pd.read_csv(whole_corpus_folder + "df_cds_agg.csv").dropna()
    df_cs = pd.read_csv(whole_corpus_folder + "df_cs_agg.csv").dropna()

    # compute length and frequency information
    df_cds['length'] = df_cds['word'].apply(lambda x: len(x))
    df_cds['log_freq'] = df_cds.apply(lambda x: np.log(x['freq']), axis=1)
    df_cs['length'] = df_cs['word'].apply(lambda x: len(x))
    df_cs['log_freq'] = df_cs.apply(lambda x: np.log(x['freq']), axis=1)

    # read concreteness and valence data
    df_conc = pd.read_csv(psycholing_data_folder + "concretewords.csv")
    df_conc = df_conc[['Word', 'Conc.M']]
    df_conc.columns = ['word', 'conc']
    df_val = pd.read_csv(psycholing_data_folder + "valencewords.csv")
    df_val.columns = ['word', 'val']

    # compute the correlation coefficient for each speech type and variable
    dfs = {"CDS": df_cds, "CS": df_cs}
    for speech_type in ("CDS", "CS"):
        df_curr = dfs[speech_type]

        # merge in valence and concreteness information
        df = pd.merge(df_curr, df_conc, on="word", how="inner")
        df = pd.merge(df, df_val[["word", "val"]], on="word", how="inner")
        df_corr = df[df["freq"] >= THRESHOLD]

        for psycholing_measure in ("length", "log_freq", "conc", "val"):

            print(f"{speech_type}-{psycholing_measure}: {pearsonr(df_corr['pfgw'], df_corr[psycholing_measure])}")
