"""
This file turns the results of hypothesis tests into LaTex-formatted tables to avoid manually copying results.
"""
import re
import numpy as np
import pandas as pd

ROOT_FOLDER = "../../data/results/hypothesis_tests"

if __name__ == "__main__":

    # Aggregate results
    print("AGGREGATE RESULTS\n")
    agg_str = ""
    df_agg = pd.read_csv(f"{ROOT_FOLDER}/aggregate.csv")
    df_agg = df_agg[df_agg["speech_metric"] == "pfgw"]
    for index, row in df_agg.iterrows():
        mean_stat = re.sub(r"0\.", ".", str(np.round(row.mean_stat, 2))).ljust(3, "0")
        p_val = "$< .01$" if row.p < 0.01 else re.sub(r"0\.", ".", str(np.round(row.p, 2))).ljust(3, "0")
        ci = re.sub(r"0\.", ".", row.ci)
        agg_str += f"{row.speech_type} & {row.embeddings} & {row.test} & {mean_stat} {ci} \\\\ \n"

    print(agg_str)

    # Results by age
    print("\nRESULTS BY AGE\n")
    age_str = ""
    df_age = pd.read_csv(f"{ROOT_FOLDER}/age.csv")
    for speech_type in ("CDS", "CS"):
        for embeddings in ("Word2Vec", "GloVe", "fastText"):
            for test in ("WEAT", "PROJ"):
                df_i = df_age[(df_age["speech_type"] == speech_type) & (df_age["embeddings"] == embeddings) &
                              (df_age["test"] == test)]
                age_str += f"{speech_type} & {embeddings} & {test} "
                for age in range(1, 6):
                    row = df_i[df_i["age"] == age].iloc[0]
                    mean_stat = re.sub(r"0\.", ".", str(np.round(row.mean_corr, 2))).ljust(3, "0")
                    p_val = "$< .01$" if row.p < 0.01 else re.sub(r"0\.", ".", str(np.round(row.p, 2))).ljust(3, "0")
                    ci = re.sub(r"0\.", ".", row.ci)
                    age_str += f"& {mean_stat} ({p_val}) {ci} "
                age_str += "\\\\ \n"

    print(age_str)

    # Class tests - one-sample
    print("\nRESULTS BY CLASS (one-sample)\n")
    class_onesample_str = ""
    df_class_1s = pd.read_csv(f"{ROOT_FOLDER}/class_onesample.csv")
    df_class_1s = df_class_1s[df_class_1s["speech_metric"] == "pfgw"]
    for speech_type in ("CDS", "CS"):
        for embeddings in ("Word2Vec", "GloVe", "fastText"):
            for test in ("WEAT", "PROJ"):
                df_i = df_class_1s[(df_class_1s["speech_type"] == speech_type) &
                                   (df_class_1s["embeddings"] == embeddings) &
                                   (df_class_1s["test"] == test)]
                class_onesample_str += f"{speech_type} & {embeddings} & {test} "
                for cla in ("Black,WC", "Black,UC", "White,WC", "White,UC"):
                    row = df_i[df_i["class"] == cla].iloc[0]
                    mean_stat = re.sub(r"0\.", ".", str(np.round(row.mean_corr, 2))).ljust(3, "0")
                    p_val = "$< .01$" if row.p < 0.01 else re.sub(r"0\.", ".", str(np.round(row.p, 2))).ljust(3, "0")
                    ci = re.sub(r"0\.", ".", row.ci)
                    class_onesample_str += f"& {mean_stat} ({p_val}) {ci} "
                class_onesample_str += "\\\\ \n"

    print(class_onesample_str)

    # Class tests - two-sample
    print("\nRESULTS BY CLASS (two-sample)\n")
    class_twosample_str = ""
    df_class_2s = pd.read_csv(f"{ROOT_FOLDER}/class_twosample.csv")
    df_class_2s = df_class_2s[df_class_2s["speech_metric"] == "pfgw"]
    for speech_type in ("CDS", "CS"):
        for embeddings in ("Word2Vec", "GloVe", "fastText"):
            for test in ("WEAT", "PROJ"):
                df_i = df_class_2s[(df_class_2s["speech_type"] == speech_type) &
                                   (df_class_2s["embeddings"] == embeddings) &
                                   (df_class_2s["test"] == test)]
                class_twosample_str += f"{speech_type} & {embeddings} & {test} "
                for comparison in ("WC/MC", "Black/White"):
                    row = df_i[df_i["comparison"] == comparison].iloc[0]
                    mean_stat = re.sub(r"0\.", ".", str(np.round(row.mean_effect, 2))).ljust(3, "0")
                    p_val = "$< .01$" if row.p < 0.01 else re.sub(r"0\.", ".", str(np.round(row.p, 2))).ljust(3, "0")
                    ci = re.sub(r"0\.", ".", row.ci)
                    class_twosample_str += f"& {mean_stat} ({p_val}) {ci} "
                class_twosample_str += "\\\\ \n"

    print(class_twosample_str)

    # changes by decade
    print("\nRESULTS BY DECADE\n")
    decade_str = ""
    df_decade = pd.read_csv(f"{ROOT_FOLDER}/decade_twosample.csv")
    df_decade = df_decade[df_decade["speech_metric"] == "pfgw"]
    for speech_type in ("CDS", "CS"):
        for embeddings in ("Word2Vec", "GloVe", "fastText"):
            for test in ("WEAT", "PROJ"):
                df_i = df_decade[(df_decade["speech_type"] == speech_type) &
                                 (df_decade["embeddings"] == embeddings) &
                                 (df_decade["test"] == test)]
                decade_str += f"{speech_type} & {embeddings} & {test} "
                for decades in ("70s-80s", "80s-90s", "70s-90s"):
                    row = df_i[df_i["decades"] == decades].iloc[0]
                    mean_stat = re.sub(r"0\.", ".", str(np.round(row.mean_effect, 2))).ljust(3, "0")
                    p_val = "$< .01$" if row.p < 0.01 else re.sub(r"0\.", ".", str(np.round(row.p, 2))).ljust(3, "0")
                    ci = re.sub(r"0\.", ".", row.ci)
                    decade_str += f"& {mean_stat} ({p_val}) {ci} "
                decade_str += "\\\\ \n"

    print(decade_str)
