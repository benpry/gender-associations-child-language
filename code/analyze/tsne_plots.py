"""
Reduce the dimensionality of vectors with t-SNE and make a nice plot
"""
import numpy as np
import pandas as pd
from adjustText import adjust_text
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from gensim.models import KeyedVectors
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine as cos

output_folder = "../../figures"

# a list of all explicitly gendered words, manually selected
gendered_words = []
with open("../../data/gendered_words.txt", "r") as fp:
    for word in fp.readlines():
        gendered_words.append(word.replace("\n", ""))


if __name__ == "__main__":

    wv = KeyedVectors.load_word2vec_format("../../data/word_embeddings/GoogleNews-vectors-negative300.bin", binary=True)

    df_assoc = pd.read_csv(f"../../data/whole_corpus/df_cds_agg.csv")
    df_assoc = df_assoc[df_assoc["word"].apply(lambda x: str(x) == str(x).lower() and str(x).isalpha() and x not in gendered_words)]
    df_assoc["keep"] = df_assoc.apply(lambda x: x["freq"] >= 500 and x["word"] in wv, axis=1)
    df_assoc = df_assoc[df_assoc["keep"]].sort_values(by="pfgw")

    words_to_track = list(df_assoc.head(30)["word"]) + list(df_assoc.tail(30)["word"])

    tsne = TSNE(n_components=2, perplexity=8)
    transformed_vectors = tsne.fit_transform([wv[w] for w in words_to_track])

    x_points = [v[0] for v in transformed_vectors]
    y_points = [v[1] for v in transformed_vectors]

    plt.rc('font', size=18)
    plt.rcParams['figure.figsize'] = (12,6)

    for speech_type in ("CDS", "CS"):
        df_yearly = pd.read_csv(f"../../data/bootstrapped_data/df_{speech_type.lower()}_yearly_bs0.csv")
        for age in (1, 3, 5):
            df_year = df_yearly[df_yearly["age"] == age]
            pfgws = df_year.set_index("word").loc[words_to_track]["pfgw"]

            fig, ax = plt.subplots()
            ax.set_xticks(())
            ax.set_yticks(())
            sc = ax.scatter(x_points, y_points, s=60, marker='o', c=[pfgws[w] for w in words_to_track], cmap=plt.get_cmap('seismic'), vmin=0, vmax=1)
            ax.set_title(f'Reduced Word Embeddings by Gender Association at Age {age} ({speech_type})', fontsize=16)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')
            ax.set_xlabel('Dimension 1 (x)')
            ax.set_ylabel('Dimension 2 (y)')
            texts = [ax.text(x_points[i], y_points[i], words_to_track[i], fontsize=16) for i in range(len(words_to_track))]
            cbar = fig.colorbar(sc)
            fig.axes[1].set_ylabel(f'Gender Probability (p(f|w))')
            cbar.set_ticks([0., 0.2, 0.4, 0.6, 0.8, 1.])
            adjust_text(texts)
            fig.savefig(f'{output_folder}/reduced_word_embeddings_by_gender_{speech_type.lower()}_age{age}.pdf', bbox_inches='tight')