import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
from scipy.spatial.distance import cosine as cos
import matplotlib.pyplot as plt
from adjustText import adjust_text
plt.rcParams['figure.figsize'] = (9,9)

# Attribute words for association testing
f_words_proj = ['she', 'her', 'woman', 'Mary', 'herself', 'daughter', 'mother', 'gal', 'girl', 'female']
m_words_proj = ['he', 'his', 'man', 'John', 'himself', 'son', 'father', 'guy', 'boy', 'male']
m_words_weat = ['brother', 'father', 'uncle', 'grandfather', 'son', 'he', 'his', 'him']
f_words_weat = ['sister', 'mother', 'aunt', 'grandmother', 'daughter', 'she', 'hers', 'her']

# a list of all explicitly gendered words, manually selected
gendered_words = ["miss", "mister", "blonde", "blond", "he", "she", "prince", "princess", "moms", "dads", "mama",
                  "dada", "papa", "auntie", "aunt", "uncle", "mothers", "fathers", "momma", "mother", "father",
                  "gramma", "grampa", "grandmother", "grandfather", "girls", "boys", "brother", "sister", "mommies",
                  "daddies", "woman", "man", "policeman", "policewoman", "mummie", "daddie", "lady", "gentleman",
                  "lord", "guy", "gal", "guys", "gals", "daughter", "son", "missus", "women", "men", "cowboy", "cowgirl", "stepsisters",
                  "stepbrothers", "grandma", "grandpa", "grandfather", "grandmother", "ladies", "gentlemen", "himself",
                  "herself", "boyfriend", "girlfriend", "female", "male", "girl", "boy", "godmother", "godfather",
                  "husband", "wife", "fireman", "firewoman", "mum", "pop", "his", "her", "hers", "stewardess",
                  "steward", "daughters", "sons", "mommy", "daddy"]

words_to_exclude = set(f_words_proj + m_words_proj + m_words_weat + f_words_weat + gendered_words)

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
    
    return compute_effect_size(w, wv, m_words, f_words)/(np.std([cos(wv[w], wv[x]) for x in m_words + f_words]))

if __name__ == "__main__":

    # random seed for reproducibility
    np.random.seed(41650)

    df_cds = pd.read_csv('../../data/whole_corpus/df_cds_agg.csv')
    df_cs = pd.read_csv('../../data/whole_corpus/df_cs_agg.csv')
    
    wv = KeyedVectors.load_word2vec_format("../../data/word_embeddings/GoogleNews-vectors-negative300.bin", binary=True)
    
    corr_words = list(set(df_cds[df_cds['freq'] >= 50]['word']) & set(wv.vocab) & set(df_cs[df_cs['freq'] >= 50]['word']))
    
    # create samples of words in each quadrant of CHILDES
    word_sample_topright = [w for w in corr_words if word_weat(w, wv) > 0.2 and float(df_cds[df_cds['word'] == w]['pfgw']) > 0.6 and w not in words_to_exclude and w == w.lower() and w.isalpha() and len(w) > 1]
    word_sample_topleft = [w for w in corr_words if word_weat(w, wv) < -0.2 and float(df_cds[df_cds['word'] == w]['pfgw']) > 0.6 and w not in words_to_exclude and w == w.lower() and w.isalpha() and len(w) > 1]
    word_sample_bottomright = [w for w in corr_words if word_weat(w, wv) > 0.2 and float(df_cds[df_cds['word'] == w]['pfgw']) < 0.4 and w not in words_to_exclude and w == w.lower() and w.isalpha() and len(w) > 1]
    word_sample_bottomleft = [w for w in corr_words if word_weat(w, wv) < -0.2 and float(df_cds[df_cds['word'] == w]['pfgw']) < 0.4 and w not in words_to_exclude and w == w.lower() and w.isalpha() and len(w) > 1]
    
    # choose the specific set of words
    words = list(np.random.choice(word_sample_topright, 10, replace=False)) + \
        list(np.random.choice(word_sample_topleft, 10, replace=False)) + \
        list(np.random.choice(word_sample_bottomright, 10, replace=False)) + \
        list(np.random.choice(word_sample_bottomleft, 10, replace=False))
    
    x_points = [word_weat(w, wv)/2 for w in words]
    y_points = [float(df_cds[df_cds['word'] == w]['pfgw']) for w in words]
    
    # make the plot and save it
    plt.rc('font', size=18)
    fig, ax = plt.subplots()
    ax.scatter(x_points, y_points)
    ax.set_title("p(f|w) and WEAT Comparison")
    ax.set_xlabel("WEAT Result (Word2Vec)")
    ax.set_ylabel("p(f|w)")
    ax.set_xticks(np.arange(-1, 1.01, 0.4))
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    texts = [ax.text(x_points[i], y_points[i], words[i], fontsize=18) for i in range(len(words))]
    adjust_text(texts)
    ax.plot((-1,1), (0.5,0.5), ls="--", c=".3")
    ax.plot((0,0), (0,1), ls="--", c=".3")
    # labels
    ax.text(-0.95, 0.475, "M", fontsize=32)
    ax.text(0.925, 0.475, "F", fontsize=32)
    ax.text(-0.04, 0.9, "F", fontsize=32)
    ax.text(-0.06, 0.05, "M", fontsize=32)

    fig.savefig("../../figures/word_sample_comparison_pfgw.pdf", bbox_inches='tight')