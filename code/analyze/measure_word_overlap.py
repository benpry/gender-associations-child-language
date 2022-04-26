import numpy as np
import pandas as pd
from pyprojroot import here
from collections import defaultdict

MIN_FREQ = 20  # the minimum number of occurrences needed to include a word
speech_types = ("CS", "CDS", "SB", "SW")
n_bs = 10000

if __name__ == "__main__":
    # compare whole corpora
    total_words = {}
    for speech_type in ("CS", "CDS"):
        df_type = pd.read_csv(here(f"data/whole_corpus/df_{speech_type.lower()}_agg.csv"))
        total_words[speech_type] = len(df_type[df_type["freq"] >= MIN_FREQ])
        print(f"total words in {speech_type}: {total_words[speech_type]}")

    # compare boostrapped instances of corpora
    bootstrapped_words = defaultdict(list)
    for speech_type in speech_types:
        for i in range(n_bs):
            if speech_type == "SB":
                df_i = pd.read_csv(here(f"data/santa_barbara_data/bootstrapped/df_santabarbara_bs{i}.csv"))
            elif speech_type == "SW":
                df_i = pd.read_csv(here(f"data/switchboard_data/bootstrapped/df_bs{i}.csv"))
            else:
                df_i = pd.read_csv(here(f"data/bootstrapped_data/df_{speech_type.lower()}_agg_bs{i}.csv"))
            bootstrapped_words[speech_type].append(list(df_i[df_i["freq"] >= MIN_FREQ]["word"]))

    # print median numbers of words in bootstrapped iterations
    for speech_type in speech_types:
        ns_words = [len(x) for x in bootstrapped_words[speech_type]]
        med_words = np.median(ns_words)
        mean_words = np.mean(ns_words)
        print(f"median words in bootstrapped {speech_type}: {med_words}")
        print(f"mean words in bootstrapped {speech_type}: {mean_words}")

    ns_shared_words = []
    # look at the overlap
    for i in range(n_bs):
        conj_words = None
        for speech_type in speech_types:
            if conj_words == None:
                conj_words = set(bootstrapped_words[speech_type][i])
            else:
                conj_words = conj_words & set(bootstrapped_words[speech_type][i])
        n_shared_words = len(conj_words)
        ns_shared_words.append(n_shared_words)
    print(f"median number of shared words: {np.median(ns_shared_words)}")
    print(f"mean number of shared words: {np.mean(ns_shared_words)}")

    ns_shared_words = []
    # overlap without SB   
    for i in range(n_bs):
        conj_words = None
        for speech_type in speech_types:
            if speech_type == "SB":
                continue
            if conj_words == None:
                conj_words = set(bootstrapped_words[speech_type][i])
            else:
                conj_words = conj_words & set(bootstrapped_words[speech_type][i])
        n_shared_words = len(conj_words)
        ns_shared_words.append(n_shared_words)
    print(f"median number of shared words (no SB): {np.median(ns_shared_words)}")
    print(f"mean number of shared words (no SB): {np.mean(ns_shared_words)}")

    ns_shared_words = []
    # overlap without SB   
    for i in range(n_bs):
        conj_words = None
        for speech_type in ("CS", "CDS"):
            if conj_words == None:
                conj_words = set(bootstrapped_words[speech_type][i])
            else:
                conj_words = conj_words & set(bootstrapped_words[speech_type][i])
        n_shared_words = len(conj_words)
        ns_shared_words.append(n_shared_words)
    print(f"median number of shared words (CS/CDS): {np.median(ns_shared_words)}")
    print(f"mean number of shared words (CS/CDS): {np.mean(ns_shared_words)}")
