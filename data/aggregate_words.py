import pandas as pd

if __name__ == "__main__":

    all_words = set()

    for i in range(100):
        df_cds_i = pd.read_csv(f"./bootstrapped_data/df_cds_agg_bs{i}.csv")
        df_cs_i = pd.read_csv(f"./bootstrapped_data/df_cs_agg_bs{i}.csv")
        all_words = all_words | set([w for w in df_cds_i[df_cds_i['freq'] >= 20]["word"] if w == w.lower() and w.isalpha()])
        all_words = all_words | set([w for w in df_cs_i[df_cs_i['freq'] >= 20]["word"] if w == w.lower() and w.isalpha()])

    with open("all_words.txt", "w") as fp:
        for w in all_words:
            fp.write(w + "\n")
    print(len(all_words))
