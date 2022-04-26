"""
This file runs all the hypothesis tests using the bootstrapped data and prints the results
"""
import numpy as np
import pandas as pd
from ast import literal_eval
from argparse import ArgumentParser
from hypothesis_testing_utils import one_sample_hypothesis_test, two_sample_hypothesis_test

bonferroni_vals = {
    "agg": 12,  # 2 x 2 x 3
    "age": 60,  # 5 x 2 x 2 x 3
    "class_onesample": 48,  # 4 x 2 x 2 x 3
    "class_twosample": 24,  # 2 x 2 x 2 x 3
    "decade": 36,  # 3 x 2 x 2 x 3
    "parent_child": 24  # 4 x 2 x 3
}


def age_hypothesis_testing(df):
    """
    Run the hypothesis tests for each age
    """
    results = []
    for age in df["Age"].drop_duplicates():
        for speech_type in df["Speech Type"].drop_duplicates():
            for embeddings in df["Embeddings"].drop_duplicates():
                for test in df["Test"].drop_duplicates():
                    df_subset = df[(df["Age"] == age) &
                                   (df["Test"] == test) &
                                   (df["Speech Type"] == speech_type) &
                                   (df["Embeddings"] == embeddings)]

                    print(f"{age}-{speech_type}-{embeddings}-{test}: {one_sample_hypothesis_test(df_subset['Pearson r'], bonferroni=bonferroni_vals['age'], string=True)}")
                    r, p, ci = one_sample_hypothesis_test(df_subset['Pearson r'], bonferroni=bonferroni_vals['age'])
                    results.append({"age": age, "speech_type": speech_type, "embeddings": embeddings, "test": test,
                                    "mean_corr": r, "p": p, "ci": ci})

    return pd.DataFrame(results)


def class_hypothesis_testing(df):
    """
    Run the hypothesis tests comparing the strength of gender associations for different classes
    """
    # test the correlation against a null hypothesis of 0
    results_onesample = []
    print("CLASS - AGAINST 0")
    for speech_metric in df["Speech Metric"].drop_duplicates():
        for test in df["Test"].drop_duplicates():
            for speech_type in df["Speech Type"].drop_duplicates():
                for embeddings in df["Embeddings"].drop_duplicates():
                    df_subset = df[(df["Speech Metric"] == speech_metric) &
                                   (df["Test"] == test) &
                                   (df["Speech Type"] == speech_type) &
                                   (df["Embeddings"] == embeddings)]

                    for cla in df_subset["class"].drop_duplicates():
                        corrs = list(df_subset[df_subset["class"] == cla]["Correlation"])
                        print(f"{speech_metric}-{test}-{speech_type}-{embeddings} {cla}: {one_sample_hypothesis_test(corrs, bonferroni=bonferroni_vals['class_onesample'], string=True)}")
                        r, p, ci = one_sample_hypothesis_test(corrs, bonferroni=bonferroni_vals['class_onesample'])
                        results_onesample.append({"class": cla, "speech_metric": speech_metric, "speech_type": speech_type,
                                            "test": test, "embeddings": embeddings, "mean_corr": r, "p": p, "ci": ci})

    df_onesample = pd.DataFrame(results_onesample)

    # compare correlation strengths
    results_twosample = []
    print("CLASS - COMPARISON")
    for speech_metric in df["Speech Metric"].drop_duplicates():
        for test in df["Test"].drop_duplicates():
            for speech_type in df["Speech Type"].drop_duplicates():
                for embeddings in df["Embeddings"].drop_duplicates():
                    df_subset = df[(df["Speech Metric"] == speech_metric) &
                                   (df["Test"] == test) &
                                   (df["Speech Type"] == speech_type) &
                                   (df["Embeddings"] == embeddings)]

                    black_wc = list(df_subset[df_subset["class"] == "Black,WC"]["Correlation"])
                    black_mc = list(df_subset[df_subset["class"] == "Black,UC"]["Correlation"])
                    white_wc = list(df_subset[df_subset["class"] == "White,WC"]["Correlation"])
                    white_mc = list(df_subset[df_subset["class"] == "White,UC"]["Correlation"])

                    all_wc, all_mc, all_black, all_white = [], [], [], []
                    for i in range(len(black_wc)):
                        all_wc.append((black_wc[i] + white_wc[i]) / 2)
                        all_mc.append((black_mc[i] + white_mc[i]) / 2)
                        all_black.append((black_wc[i] + black_mc[i]) / 2)
                        all_white.append((white_wc[i] + white_mc[i]) / 2)

                    print(f"{speech_metric}-{test}-{speech_type}-{embeddings} wc vs mc: {two_sample_hypothesis_test(all_wc, all_mc, bonferroni=bonferroni_vals['class_twosample'], string=True)}")
                    r, p, ci = two_sample_hypothesis_test(all_wc, all_mc, bonferroni=bonferroni_vals['class_twosample'])
                    results_twosample.append({"speech_metric": speech_metric, "test": test, "speech_type": speech_type,
                                              "embeddings": embeddings, "comparison": "WC/MC", "mean_effect": r,
                                              "p": p, "ci": ci})
                    print(f"{speech_metric}-{test}-{speech_type}-{embeddings} black vs white: {two_sample_hypothesis_test(all_black, all_white, bonferroni=bonferroni_vals['class_twosample'], string=True)}")
                    r, p, ci = two_sample_hypothesis_test(all_black, all_white, bonferroni=bonferroni_vals['class_twosample'])
                    results_twosample.append({"speech_metric": speech_metric, "test": test, "speech_type": speech_type,
                                              "embeddings": embeddings, "comparison": "Black/White", "mean_effect": r,
                                              "p": p, "ci": ci})

    df_twosample = pd.DataFrame(results_twosample)
    return df_onesample, df_twosample


def decade_significance_tests(df):
    """
    Run significance tests for differences between subsequent decades for each speech type, set of
    embeddings, and association test.
    """
    results_onesample = []
    print("against 0")
    for speech_metric in df["Speech Metric"].drop_duplicates():
        for speech_type in df["Speech Type"].drop_duplicates():
            for embeddings in df['Embeddings'].drop_duplicates():
                for test in df["Test"].drop_duplicates():
                    for decade in df["Decade"].drop_duplicates():
                        df_subset = df[(df["Speech Metric"] == speech_metric) &
                                    (df['Speech Type'] == speech_type) &
                                    (df['Embeddings'] == embeddings) &
                                    (df['Test'] == test) &
                                    (df['Decade'] == decade)]
                        corrs = df_subset['Correlation']

                        print(f"{speech_metric}-{speech_type}-{embeddings}-{test} vs. 0: {one_sample_hypothesis_test(corrs, bonferroni=bonferroni_vals['decade'], string=True)}")
                        r, p, ci = one_sample_hypothesis_test(corrs, bonferroni=bonferroni_vals['decade'])
                        results_onesample.append({"speech_metric": speech_metric, "speech_type": speech_type,
                                                  "embeddings": embeddings, "test": test, "decade": decade,
                                                  "mean_effect": r, "p": p, "ci": ci})
    df_onesample = pd.DataFrame(results_onesample)

    results_twosample = []
    print("70s-80s")
    for speech_metric in df["Speech Metric"].drop_duplicates():
        increased = 0
        for speech_type in df["Speech Type"].drop_duplicates():
            for embeddings in df['Embeddings'].drop_duplicates():
                for test in df["Test"].drop_duplicates():
                    df_subset = df[(df["Speech Metric"] == speech_metric) &
                                   (df['Speech Type'] == speech_type) &
                                   (df['Embeddings'] == embeddings) &
                                   (df['Test'] == test)]
                    seventies_corrs = df_subset[df_subset['Decade'] == "70s"]['Correlation']
                    eighties_corrs = df_subset[df_subset['Decade'] == "80s"]['Correlation']

                    hyp_test = two_sample_hypothesis_test(seventies_corrs, eighties_corrs, bonferroni=bonferroni_vals['decade'])
                    print(f"{speech_metric}-{speech_type}-{embeddings}-{test} 70s vs 80s: {two_sample_hypothesis_test(seventies_corrs, eighties_corrs, bonferroni=bonferroni_vals['decade'], string=True)}")
                    r, p, ci = two_sample_hypothesis_test(seventies_corrs, eighties_corrs, bonferroni=bonferroni_vals['decade'])
                    results_twosample.append({"speech_metric": speech_metric, "speech_type": speech_type,
                                              "embeddings": embeddings, "test": test, "decades": "70s-80s",
                                              "mean_effect": r, "p": p, "ci": ci})
                    if hyp_test[0] < 0:
                        increased += 1
        print(f"{increased} increased")

    print("80s-90s")
    for speech_metric in df["Speech Metric"].drop_duplicates():
        increased = 0
        for speech_type in df["Speech Type"].drop_duplicates():
            for embeddings in df['Embeddings'].drop_duplicates():
                for test in df["Test"].drop_duplicates():
                    df_subset = df[(df["Speech Metric"] == speech_metric) &
                                   (df['Speech Type'] == speech_type) &
                                   (df['Embeddings'] == embeddings) &
                                   (df['Test'] == test)]
                    eighties_corrs = df_subset[df_subset['Decade'] == "80s"]['Correlation']
                    nineties_corrs = df_subset[df_subset['Decade'] == "90s"]['Correlation']

                    hyp_test = two_sample_hypothesis_test(eighties_corrs, nineties_corrs, bonferroni=bonferroni_vals['decade'])
                    print(f"{speech_metric}-{speech_type}-{embeddings}-{test} 80s vs 90s: {two_sample_hypothesis_test(eighties_corrs, nineties_corrs, bonferroni=bonferroni_vals['decade'], string=True)}")
                    r, p, ci = two_sample_hypothesis_test(eighties_corrs, nineties_corrs, bonferroni=bonferroni_vals['decade'])
                    results_twosample.append({"speech_metric": speech_metric, "speech_type": speech_type, "test": test,
                                              "embeddings": embeddings, "decades": "80s-90s", "mean_effect": r, "p": p, "ci": ci})
                    if hyp_test[0] < 0:
                        increased += 1
        print(f"{increased} increased")

    print("70s-90s")
    for speech_metric in df["Speech Metric"].drop_duplicates():
        increased = 0
        for speech_type in df["Speech Type"].drop_duplicates():
            for embeddings in df['Embeddings'].drop_duplicates():
                for test in df["Test"].drop_duplicates():
                    df_subset = df[(df["Speech Metric"] == speech_metric) &
                                    (df['Speech Type'] == speech_type) &
                                    (df['Embeddings'] == embeddings) &
                                    (df['Test'] == test)]
                    seventies_corrs = df_subset[df_subset['Decade'] == "70s"]['Correlation']
                    nineties_corrs = df_subset[df_subset['Decade'] == "90s"]['Correlation']

                    hyp_test = two_sample_hypothesis_test(seventies_corrs, nineties_corrs, bonferroni=bonferroni_vals['decade'])
                    print(f"{speech_metric}-{speech_type}-{embeddings}-{test} 70s vs 90s: {two_sample_hypothesis_test(seventies_corrs, nineties_corrs, bonferroni=bonferroni_vals['decade'], string=True)}")
                    r, p, ci = two_sample_hypothesis_test(seventies_corrs, nineties_corrs, bonferroni=bonferroni_vals['decade'])
                    results_twosample.append({"speech_metric": speech_metric, "speech_type": speech_type, "test": test,
                                              "embeddings": embeddings, "decades": "70s-90s", "mean_effect": r, "p": p,
                                              "ci": ci})
                    if hyp_test[0] < 0:
                        increased += 1
        print(f"{increased} increased")

    df_twosample = pd.DataFrame(results_twosample)

    return df_onesample, df_twosample


def parent_child_hypothesis_testing(df):
    """
    Test the significance of the differences between father-daughter and mother-son
    correlations
    """
    results = []
    for metric in df["Test"].drop_duplicates():
        for embeddings in df['Embeddings'].drop_duplicates():
            df_subset = df[(df["Test"] == metric) & (df['Embeddings'] == embeddings)]
            father_son_assocs = list(df_subset[(df_subset['Speech Type'] == 'father_son')]["Association"])
            father_daughter_assocs = list(df_subset[(df_subset['Speech Type'] == 'father_daughter')]["Association"])
            mother_son_assocs = list(df_subset[(df_subset['Speech Type'] == 'mother_son')]["Association"])
            mother_daughter_assocs = list(df_subset[(df_subset['Speech Type'] == 'mother_daughter')]["Association"])

            # create a list of the mean father and mother association
            father_assocs, mother_assocs, daughter_assocs, son_assocs = [], [], [] , []
            for i in range(len(father_son_assocs)):
                father_assocs.append((father_son_assocs[i] + father_daughter_assocs[i]) / 2)
                mother_assocs.append((mother_son_assocs[i] + mother_daughter_assocs[i]) / 2)
                daughter_assocs.append((father_daughter_assocs[i] + mother_daughter_assocs[i]) / 2)
                son_assocs.append((father_son_assocs[i] + mother_son_assocs[i]) / 2)

            print(f"{metric}-{embeddings}: father vs mother: {two_sample_hypothesis_test(father_assocs, mother_assocs, bonferroni=bonferroni_vals['parent_child'], string=True)}")
            r, p, ci = two_sample_hypothesis_test(father_assocs, mother_assocs, bonferroni=bonferroni_vals['parent_child'])
            results.append({"metric": metric, "embeddings": embeddings, "comparison": "father-mother", "mean_stat": r,
                            "p": p, "ci": ci})
            print(f"{metric}-{embeddings}: son vs daughter: {two_sample_hypothesis_test(son_assocs, daughter_assocs, bonferroni=bonferroni_vals['parent_child'], string=True)}")
            r, p, ci = two_sample_hypothesis_test(son_assocs, daughter_assocs, bonferroni=bonferroni_vals['parent_child'])
            results.append({"metric": metric, "embeddings": embeddings, "comparison": "son-daughter", "mean_stat": r,
                            "p": p, "ci": ci})

    df_results = pd.DataFrame(results)
    return df_results


if __name__ == "__main__":

    print("AGGREGATE ANALYSIS\n")
    agg_results = []
    agg_data_file = "../../data/results/aggregate/bootstrapped_aggregate_corrs_nonames.csv"
    df = pd.read_csv(agg_data_file)
    df.columns = ["Unnamed: 0", "bootstrap_num", "Speech Type", "Embeddings", "Speech Metric", "Test", "Correlation", "p"]

    for metric in ('pfgw', 'OR', 'LOR'):
        df_subset = df[df["Speech Metric"] == metric]
        for speech_type in ("CDS", "CS"):
            for embeddings in ("Word2Vec", "GloVe", "fastText"):
                for test in ("WEAT", "PROJ"):
                    df_i = df_subset[(df_subset["Speech Type"] == speech_type) & (df_subset["Embeddings"] == embeddings) & (df_subset["Test"] == test)]

                    print(f"{metric}-{speech_type}-{test}-{embeddings}: {one_sample_hypothesis_test(df_i['Correlation'], string=True)}")
                    r, p, ci = one_sample_hypothesis_test(df_i["Correlation"], bonferroni=bonferroni_vals["agg"])
                    agg_results.append({"speech_metric": metric, "embeddings": embeddings, "speech_type": speech_type,
                                          "test": test, "mean_stat": r, "p": p, "ci": ci})

    df_agg_results = pd.DataFrame(agg_results)
    df_agg_results.to_csv("../../data/results/hypothesis_tests/aggregate.csv")

    print("\nYEARLY ANALYSIS\n")
    yearly_data_file = "../../data/results/yearly/bootstrapped_yearly_corrs_nonames.csv"
    df = pd.read_csv(yearly_data_file)
    df.columns = ['Unnamed: 0', 'bootstrap_num', 'Speech Type', 'Embeddings', 'Age', 'Metric', "Test", 'Pearson r', "p"]
    df_age = age_hypothesis_testing(df)
    df_age.to_csv("../../data/results/hypothesis_tests/age.csv")

    print("\nADULT ANALYSIS\n")
    # adult analysis
    adult_results = []
    for corpus in ("santa_barbara", "switchboard"):
        print(f"\n{corpus.upper()}\n")
        data_file = f"../../data/results/{corpus}/bootstrapped_corrs_{corpus.replace('_', '')}_nonames.csv"

        # load the data
        df = pd.read_csv(data_file).dropna()
        df.columns = ['Unnamed: 0', 'bootstrap_num', 'Speech Type', 'Embeddings', 'speech_metric', 'Test', 'Correlation', 'p']
        df = df[df['Speech Type'] == "CS"]

        for speech_metric in ("pfgw", "OR", "LOR"):
            # make the plot
            df_subset = df[df['speech_metric'] == speech_metric]

            for embeddings in ("Word2Vec", "GloVe", "fastText"):
                for test in ("WEAT", "PROJ"):
                    print(f"metric: {speech_metric}, embeddings: {embeddings}")
                    df_i = df_subset[(df_subset['Test'] == test) & (df_subset['Embeddings'] == embeddings)]
                    print(df_i['Correlation'].mean())
                    print(one_sample_hypothesis_test(df_i["Correlation"], bonferroni=bonferroni_vals["agg"]))
                    r, p, ci = one_sample_hypothesis_test(df_i["Correlation"], bonferroni=bonferroni_vals["agg"])
                    adult_results.append({"corpus": corpus, "speech_metric": speech_metric, "embeddings": embeddings,
                                          "test": test, "mean_stat": r, "p": p, "ci": ci})

    df_adult = pd.DataFrame(adult_results)
    df_adult.to_csv("../../data/results/hypothesis_tests/adult.csv")

    print("\nDECADE ANALYSIS (non-historical embeddings)\n")
    decade_data_file = "../../data/results/decade/bootstrapped_results_by_decade_nonames.csv"
    df = pd.read_csv(decade_data_file)
    df.columns = ["Unnamed: 0", "bootstrap_num", "Speech Type", "Embeddings", 'Decade', "Speech Metric",
                  "Test", "Correlation", "p"]
    df_decade_1s, df_decade_2s = decade_significance_tests(df)
    df_decade_1s.to_csv("../../data/results/hypothesis_tests/decade_onesample.csv")
    df_decade_2s.to_csv("../../data/results/hypothesis_tests/decade_twosample.csv")

    print("\nDECADE ANALYSIS (historical embeddings)\n")
    decade_data_file = "../../data/results/decade/bootstrapped_results_by_decade_nonames_hist.csv"
    df = pd.read_csv(decade_data_file)
    df.columns = ["Unnamed: 0", "bootstrap_num", "Speech Type", "Embeddings", "Decade", "Speech Metric",
                  "Test", "Correlation", "p"]

    df_decade_1s_hist, df_decade_2s_hist = decade_significance_tests(df)
    df_decade_1s_hist.to_csv("../../data/results/hypothesis_tests/decade_hist_onesample.csv")
    df_decade_2s_hist.to_csv("../../data/results/hypothesis_tests/decade_hist_twosample.csv")

    print("\nPARENT-CHILD ANALYSIS\n")
    parent_child_data_file = "../../data/results/parent_child/parent_child_corrs_nonames.csv"
    df = pd.read_csv(parent_child_data_file)
    df.columns = ["Unnamed: 0", "bootstrap_num", "Speech Type", "Embeddings", "Test", "Association"]
    df_pc = parent_child_hypothesis_testing(df)
    df_pc.to_csv("../../data/results/hypothesis_tests/parent_child.csv")

    print("\nCLASS ANALYSIS\n")
    class_data_file = "../../data/results/class/bootstrapped_class_corrs_nonames.csv"
    df = pd.read_csv(class_data_file)
    df.columns = ["Unnamed: 0", "bootstrap_num", "Speech Type", "Embeddings", 'class', "Speech Metric", "Test", "Correlation", "p"]
    df_class_1s, df_class_2s = class_hypothesis_testing(df)
    df_class_1s.to_csv("../../data/results/hypothesis_tests/class_onesample.csv")
    df_class_2s.to_csv("../../data/results/hypothesis_tests/class_twosample.csv")
