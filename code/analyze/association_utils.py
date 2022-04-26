import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import fasttext
import pickle
from tqdm import tqdm
from scipy.spatial.distance import cosine as cos
from gensim.models import KeyedVectors
from gensim.test.utils import datapath, get_tmpfile
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from adjustText import adjust_text
from collections import defaultdict
from scipy.stats import spearmanr

# The filepaths for the Word2Vec, GloVe, and fastText embeddings
W2V_FILEPATH = "/hal9000/datasets/wordembeddings/google_news_w2v.bin"
GLOVE_FILEPATH = "/hal9000/datasets/wordembeddings/glove_w2v_format.txt"
FASTTEXT_FILEPATH = "/hal9000/datasets/wordembeddings/english_fasttext.bin"

# The minimum frequency with which words must be said to children to be
# included in the list of words used in the correlation.
MIN_FREQ = 20

# Attribute words for association testing
f_words_proj = ['she', 'her', 'woman', 'Mary', 'herself', 'daughter', 'mother', 'gal', 'girl', 'female']
m_words_proj = ['he', 'his', 'man', 'John', 'himself', 'son', 'father', 'guy', 'boy', 'male']
m_words_weat = ['brother', 'father', 'uncle', 'grandfather', 'son', 'he', 'his', 'him']
f_words_weat = ['sister', 'mother', 'aunt', 'grandmother', 'daughter', 'she', 'hers', 'her']

# a list of all explicitly gendered words, manually selected
gendered_words = []
with open("../../data/gendered_words.txt", "r") as fp:
    for word in fp.readlines():
        gendered_words.append(word.replace("\n", ""))

words_to_exclude = set(f_words_proj + m_words_proj + m_words_weat + f_words_weat + gendered_words)


def doPCA(pairs, embedding, num_components = 10):
    """
    This fucntion does principal component analysis

    Code from Tolga Bolukbasi: https://github.com/tolga-b/debiaswe/blob/master/debiaswe/we.py
    """
    matrix = []
    for a, b in pairs:
        center = (embedding[a] + embedding[b])/2
        matrix.append(embedding[a] - center)
        matrix.append(embedding[b] - center)
    matrix = np.array(matrix)
    pca = PCA(n_components = num_components)
    pca.fit(matrix)
    # bar(range(num_components), pca.explained_variance_ratio_)
    return pca


def projection(vec, basis):
    """
    Scalar projection of a vector onto a basis,
    used for PROJ analysis
    """
    return vec.dot(basis)/basis.dot(basis)


def compute_effect_size(w, wv, A, B):
    """
    Computes the effect size for WEAT, the difference between the mean cosine similarity
    between word w and words in set A and words in set B
    """
    return np.mean([cos(wv[w], wv[a]) for a in A]) - np.mean([cos(wv[w], wv[b]) for b in B])


def word_weat(w, wv):
    """
    run WEAT on an individual word
    """
    for word in m_words_weat + f_words_weat:
        if word not in wv:
            print(word, 'not in wv')
    if w not in wv:
        print(w, 'not in wv')

    return compute_effect_size(w, wv, m_words_weat, f_words_weat)/(np.std([cos(wv[w], wv[x]) for x in m_words_weat + f_words_weat]))


def get_gender_basis(wv, embeddings_name):
    if embeddings_name == "Hist":
        mw_proj = m_words_proj[:3] + m_words_proj[4:]
        fw_proj = f_words_proj[:3] + f_words_proj[4:]
    else:
        mw_proj = m_words_proj
        fw_proj = f_words_proj

    female_vectors = [wv[word] for word in fw_proj]
    male_vectors = [wv[word] for word in mw_proj]
    gendered_pairs = [(fw_proj[i], mw_proj[i]) for i in range(len(fw_proj))]
    pca = doPCA(gendered_pairs, wv)
    gender_basis = pca.components_[0]
    if np.mean([projection(wv[w], gender_basis) for w in fw_proj]) < np.mean([projection(wv[w], gender_basis) for w in mw_proj]):
        gender_basis = - gender_basis


    return gender_basis


def get_correlation(df_cds, df_cs, embeddings, speech_metric, assoc_type, wv, include_names, conjset=None):
    """
    Compute the strength of the correlation between word embedding associations and gender probability in the
    given dataframes of speech statistics
    """
    # read the word embeddings (if they were not passed in as an argument)
    if embeddings == 'Word2Vec':
        if wv is None:
            wv = KeyedVectors.load_word2vec_format(W2V_FILEPATH, binary=True)
        vocab = wv.vocab
    elif embeddings == 'GloVe':
        if wv is None:
            wv = KeyedVectors.load_word2vec_format(GLOVE_FILEPATH)
        vocab = wv.vocab
    elif embeddings == 'fastText':
        if wv is None:
            wv = fasttext.load_model(FASTTEXT_FILEPATH)
        vocab = wv.get_words()
    elif embeddings == 'Hist':
        vocab = [w for w in wv.keys() if np.linalg.norm(wv[w]) != 0]
    else:
        print(f"unknown embedding type: {embeddings}")
        return None

    # either use a conjunctive set of words (passed in as an argument),
    # or compute the set of words to be used to measure the correlation
    if conjset:
        corr_words = list(set(conjset) & set(vocab))
    else:
        if len(df_cds) > 0:
            corr_words = list(set(df_cds[df_cds['freq'] >= MIN_FREQ]['word']) & set(vocab) & set(df_cs[df_cs['freq'] >= MIN_FREQ]['word']) \
                              & set(df_cds[~np.isnan(df_cds['LOR'])]['word']) & set(df_cs[~np.isnan(df_cs['LOR'])]['word']))
        else:
            corr_words = list(set(df_cs[df_cs['freq'] >= MIN_FREQ]['word']) & set(vocab) & set(df_cs[~np.isnan(df_cs['LOR'])]['word']))

    # drop all names if specified
    if not include_names:
        corr_words = [x for x in corr_words if x == x.lower() and x not in f_words_proj + m_words_proj +
                      m_words_weat + f_words_weat + gendered_words]

    # get the word embedding associations in a list
    if len(df_cds) > 0:
        cds_assocs = [float(df_cds[df_cds['word'] == word][speech_metric]) for word in corr_words]
    cs_assocs = [float(df_cs[df_cs['word'] == word][speech_metric]) for word in corr_words]

    # get a list of word embedding associations, depending on the association test
    if assoc_type == 'WEAT':
        wv_associations = [word_weat(w, wv) for w in corr_words]
    elif assoc_type == 'PROJ':
        # Skip "John" and "Mary" if using the HistWords embeddings (since they don't appear in HistWords)
        if embeddings == "Hist":
            mw_proj = m_words_proj[:3] + m_words_proj[4:]
            fw_proj = f_words_proj[:3] + f_words_proj[4:]
        else:
            mw_proj = m_words_proj
            fw_proj = f_words_proj
        female_vectors = [wv[word] for word in fw_proj]
        male_vectors = [wv[word] for word in mw_proj]
        gendered_pairs = [(fw_proj[i], mw_proj[i]) for i in range(len(fw_proj))]
        pca = doPCA(gendered_pairs, wv)
        gender_basis = pca.components_[0]
        # make sure the gender basis is pointing in the female direction
        if np.mean([projection(wv[w], gender_basis) for w in fw_proj]) < np.mean([projection(wv[w], gender_basis) for w in mw_proj]):
            gender_basis = - gender_basis
        wv_associations = [projection(wv[w], gender_basis) for w in corr_words]
    else:
        print(f"unknown association type: {assoc_type}")
        return None

    # compute the correlation strength in CDS and CS
    correlation = {}
    if len(df_cds) > 0:
        correlation['CDS'] = pearsonr(cds_assocs, wv_associations)
    else:
        correlation['CDS'] = "NA"
    correlation['CS'] = pearsonr(cs_assocs, wv_associations)
    return correlation


def get_all_correlations(df_cds, df_cs, embedding_types, association_types, include_names=False, conjset=None):
    """
    Compute the correlations using all the specified embeddings and association tests
    if include_names is false, all names (defined as words that include captial letters) are dropped.
    If conjset is not none, it should be a list of all the words that should be used to compute the correlation.
    """
    all_corrs = []

    for embeddings in embedding_types:
        # load the embeddings
        print(f"loading {embeddings}")
        if embeddings == 'Word2Vec':
            wv = KeyedVectors.load_word2vec_format(W2V_FILEPATH, binary=True)
        elif embeddings == 'GloVe':
            wv = KeyedVectors.load_word2vec_format(GLOVE_FILEPATH)
        elif embeddings == 'fastText':
            wv = fasttext.load_model(FASTTEXT_FILEPATH)
        print(f"loaded {embeddings}")

        # for each association type, compute the correlation in child speech and child-directed speech
        for assoc_type in association_types:
            corr = get_correlation(df_cds, df_cs, embeddings, "pfgw", assoc_type, wv=wv, include_names=include_names, conjset=conjset)
            all_corrs.append({"speech_type": "CDS", "embeddings": embeddings, "assoc": assoc_type, "corr": corr["CDS"]})
            all_corrs.append({"speech_type": "CS", "embeddings": embeddings, "assoc": assoc_type, "corr": corr["CS"]})

    # make a dataframe out of the computed correlations
    df_corrs =  pd.DataFrame(all_corrs)

    return df_corrs


def get_corrs_fast(df_speech, word_scores, corr_words):

    # compute the correlation for each association type and speech metric
    corrs = []
    df_words = df_speech[df_speech["word"].isin(corr_words)].reset_index()
    for assoc_type in ("WEAT", "PROJ"):
        for speech_metric in ("pfgw", "OR", "LOR"):
            speech_assocs = df_words[speech_metric]
            wv_assocs = [word_scores[assoc_type][w] for w in df_words["word"]]
            r, p = pearsonr(speech_assocs, wv_assocs)
            corrs.append({"speech_metric": speech_metric, "assoc": assoc_type, "corr": r, "p": p})

    return corrs


def get_embeddings(embeddings):
    print(f"Loading {embeddings}...")
    if embeddings == 'Word2Vec':
        wv = KeyedVectors.load_word2vec_format(W2V_FILEPATH, binary=True)
        vocab = wv.vocab
    elif embeddings == 'GloVe':
        wv = KeyedVectors.load_word2vec_format(GLOVE_FILEPATH)
        vocab = wv.vocab
    elif embeddings == 'fastText':
        wv = fasttext.load_model(FASTTEXT_FILEPATH)
        vocab = wv.get_words()
    print(f"Loaded {embeddings}!")

    return wv, vocab


def compute_word_scores(words, wv, embeddings):
    word_scores = {"WEAT": {}, "PROJ": {}}
    gender_basis = get_gender_basis(wv, embeddings)
    print("computing word scores")
    for word in tqdm(words):
        word_scores["WEAT"][word] = word_weat(word, wv)
        word_scores["PROJ"][word] = projection(wv[word], gender_basis)

    return word_scores


def compute_all_words(dfs_cds, dfs_cs, no_min_freq=False):
    if no_min_freq:
        min_f = -1
    else:
        min_f = MIN_FREQ

    all_words = set()
    print("computing all words")
    for i in tqdm(range(len(dfs_cds))):
        for df_words in (dfs_cds[i], dfs_cs[i]):
            if len(df_words) == 0:
                continue
            df_words = df_words[df_words["freq"] >= min_f]
            all_words = all_words | set(df_words["word"])

    all_words = list(all_words)
    return all_words


def get_all_correlations_bootstrap(all_dfs_cds, all_dfs_cs, embedding_types, association_types, include_names=False):
    """
    This is the (hopefully) much faster version of the bootstrapped correlations
    """
    all_corrs = []

    # compute a list of all the words that we will need to compute association metrics for
    all_words = compute_all_words(all_dfs_cds, all_dfs_cs)

    for embeddings in embedding_types:
        # load the relevant embeddings
        wv, vocab = get_embeddings(embeddings)

        # filter out the words that are not in the vocabulary
        embedding_words = [w for w in all_words if w in vocab]

        # compute WEAT and PROJ scores for every word
        word_scores = compute_word_scores(embedding_words, wv, embeddings)

        # create a smaller version of the word embedding vocabulary that's easier to check
        vocab_small = set(embedding_words)

        # compute the correlations for each bootstrapped iteration
        print("running bootstrapping")
        for i in tqdm(range(len(all_dfs_cds))):
            df_cds, df_cs = all_dfs_cds[i], all_dfs_cs[i]

            if len(df_cds) > 0:
                corr_words = set(df_cds[df_cds["freq"] >= MIN_FREQ]["word"]) \
                    & set(df_cs[df_cs["freq"] >= MIN_FREQ]["word"]) \
                    & set(df_cds[~np.isnan(df_cds['LOR'])]['word']) \
                    & set(df_cs[~np.isnan(df_cs['LOR'])]['word']) \
                    & vocab_small
            else:
               corr_words = set(df_cs[df_cs["freq"] >= MIN_FREQ]["word"]) \
                    & set(df_cs[~np.isnan(df_cs['LOR'])]['word']) \
                    & vocab_small

            # drop all names if specified
            if not include_names:
                corr_words = set([w for w in corr_words if w == w.lower() and w not in words_to_exclude])

            # compute the correlations here
            if len(df_cds) > 0:
                cds_corrs = get_corrs_fast(all_dfs_cds[i], word_scores, corr_words)
                cds_corrs = [{"bootstrap_num": i, "speech_type": "CDS", "embeddings": embeddings, **corr} for corr in cds_corrs]
                all_corrs.extend(cds_corrs)

            cs_corrs = get_corrs_fast(all_dfs_cs[i], word_scores, corr_words)
            cs_corrs = [{"bootstrap_num": i, "speech_type": "CS", "embeddings": embeddings, **corr} for corr in cs_corrs]
            all_corrs.extend(cs_corrs)

    df_corrs = pd.DataFrame(all_corrs)
    return df_corrs


def get_class_word_list(df, cla):
    """
    Get the list of words said at least MIN_FREQ times to children of class cla
    """
    df_cla = df[df['class'] == cla]
    words = []
    for index, row in df_cla.iterrows():
        if row['freq'] >= MIN_FREQ and not np.isnan(row['LOR']) and not np.isinf(row['LOR']) and not np.isnan(row['OR']) and not np.isinf(row['OR']):
            words.append(row['word'])

    return words


def get_all_correlations_by_class_bootstrap(all_dfs_cds, all_dfs_cs, embedding_types, association_types, include_names=False, use_conjset=False):
    """
    Compute all the correlations between gender associations in word embeddings and speech by class
    """
    all_corrs = []

    # get all the words
    all_words = compute_all_words(all_dfs_cds, all_dfs_cs)

    for embeddings in embedding_types:
        # load the relevant word embeddings
        wv, vocab = get_embeddings(embeddings)

        # filter out the words that are not in the vocabulary
        embedding_words = [w for w in all_words if w in vocab]

        word_scores = compute_word_scores(embedding_words, wv, embeddings)

        # create a smaller version of the word embedding vocabulary that's easier to check
        vocab_small = set(embedding_words)

        # iterate over bootstrap iterations
        print("doing bootstrapping")
        for i in tqdm(range(len(all_dfs_cds))):
            df_cds = all_dfs_cds[i]
            df_cs = all_dfs_cs[i]

            # if we are using the conjunctive set of words, compute the set of words said at least 20 times
            # to and by children of each class
            if use_conjset:
                # get lists of all the words with sufficient frequency
                word_lists = []
                for cla in df_cds['class'].drop_duplicates():
                    cds_words = get_class_word_list(df_cds, cla)
                    word_lists.append(cds_words)
                    cs_words = get_class_word_list(df_cs, cla)
                    word_lists.append(cs_words)

                # compute the conjunctive set of all the word lists
                conj_set = set(word_lists[0])
                for word_list in word_lists[1:]:
                    conj_set = conj_set & set(word_list)
                conj_set = list(conj_set & vocab_small)
            else:
                conj_set = None

            # run the analysis with each class
            for cla in df_cds['class'].drop_duplicates():
                df_cds_cla = df_cds[df_cds['class'] == cla]
                df_cs_cla = df_cs[df_cs['class'] == cla]

                if conj_set:
                    corr_words = conj_set
                else:
                    corr_words = set(df_cds_cla[df_cds_cla["freq"] >= MIN_FREQ]["word"]) \
                        & set(df_cs_cla[df_cs_cla["freq"] >= MIN_FREQ]["word"]) \
                        & set(df_cds_cla[~np.isnan(df_cds_cla['LOR'])]['word']) \
                        & set(df_cs_cla[~np.isnan(df_cs_cla['LOR'])]['word']) \
                        & vocab_small

                if not include_names:
                    corr_words = set([w for w in corr_words if w == w.lower() and w not in words_to_exclude])

                cds_corrs = get_corrs_fast(df_cds_cla, word_scores, corr_words)
                cs_corrs = get_corrs_fast(df_cs_cla, word_scores, corr_words)
                cds_corrs = [{"bootstrap_num": i, "speech_type": "CDS", "embeddings": embeddings, "class": cla, **corr} for corr in cds_corrs]
                cs_corrs = [{"bootstrap_num": i, "speech_type": "CS", "embeddings": embeddings, "class": cla, **corr} for corr in cs_corrs]
                all_corrs.extend(cds_corrs)
                all_corrs.extend(cs_corrs)

    # create and return a DataFrame
    df_corrs = pd.DataFrame(all_corrs)
    return df_corrs


def get_yearly_correlations_bootstrap(all_dfs_cds, all_dfs_cs, embedding_types, age_range, conj_words=None, include_names=False):
    """
    This function computes the correlations for each child age with lists of dataframes containing bootstrapped statistics
    """
    all_corrs = []

    # compute a list of all the words
    all_words = compute_all_words(all_dfs_cds, all_dfs_cs)

    for embeddings in embedding_types:
        # load the relevant embeddings
        wv, vocab = get_embeddings(embeddings)

        # filter out the words that are not in the vocabulary
        embedding_words = [w for w in all_words if w in vocab]

        # compute WEAT and PROJ scores for every word
        word_scores = compute_word_scores(embedding_words, wv, embeddings)

        # create a smaller version of the word embedding vocabulary that's easier to check
        vocab_small = set(embedding_words)

        print("running bootstrapping")
        for i in tqdm(range(len(all_dfs_cds))):
            df_cds = all_dfs_cds[i]
            df_cs = all_dfs_cs[i]

            for age in age_range:
                # restrict the dataframe to the specified age
                df_cs_age = df_cs[df_cs['age'] == age]
                df_cds_age = df_cds[df_cds['age'] == age]

                # either use the conjunctive set of words or compute a new set
                if conj_words == None:
                    yearly_corr_words = set(df_cds_age[df_cds_age["freq"] >= MIN_FREQ]["word"]) \
                        & set(df_cs_age[df_cs_age["freq"] >= MIN_FREQ]["word"]) \
                        & set(df_cds_age[~np.isnan(df_cds_age['LOR'])]['word']) \
                        & set(df_cs_age[~np.isnan(df_cs_age['LOR'])]['word']) \
                        & vocab_small
                else:
                    yearly_corr_words = conj_words

                # drop all names if specified
                if not include_names:
                    yearly_corr_words = set([w for w in yearly_corr_words if w == w.lower() and w not in words_to_exclude])

                cds_corrs = get_corrs_fast(df_cds_age, word_scores, yearly_corr_words)
                cs_corrs = get_corrs_fast(df_cs_age, word_scores, yearly_corr_words)
                cds_corrs = [{"bootstrap_num": i, "speech_type": "CDS", "embeddings": embeddings, "age": age, **corr} for corr in cds_corrs]
                cs_corrs = [{"bootstrap_num": i, "speech_type": "CS", "embeddings": embeddings, "age": age, **corr} for corr in cs_corrs]
                all_corrs.extend(cds_corrs)
                all_corrs.extend(cs_corrs)

    df_corrs = pd.DataFrame(all_corrs)
    return df_corrs


def get_correlations_by_decade_boostrap(all_dfs_cds, all_dfs_cs, decades, embedding_types, include_names=False, hist_embeddings=False):
    """
    Compute the correlation between word embedding associations and gender associations in speech for each decade based on bootstrapped
    dataframes
    """
    all_corrs = []

    # compute a list of all the words that we will need to compute association metrics for
    all_words = compute_all_words(all_dfs_cds, all_dfs_cs)

    for embeddings in embedding_types:
        # load the word embeddings
        if not hist_embeddings:
            wv, vocab = get_embeddings(embeddings)

            # filter out the words that are not in the vocabulary
            embedding_words = [w for w in all_words if w in vocab]

            # compute WEAT and PROJ scores for every word
            word_scores = compute_word_scores(embedding_words, wv, embeddings)

            # create a smaller version of the word embedding vocabulary that's easier to check
            vocab_small = set(embedding_words)

        # iterate over decades and test types
        for decade in decades:

            if hist_embeddings:
                # read the HistWords embeddings
                with open(f"/hal9000/datasets/wordembeddings/HistWords/19{decade[:-1]}-vocab.pkl", "rb") as fp:
                    vocab = pickle.load(fp)

                vectors = np.load(f"/hal9000/datasets/wordembeddings/HistWords/19{decade[:-1]}-w.npy")
                wv = dict(zip(vocab, vectors))

                # filter out the words with zero norm
                vocab = [w for w in vocab if np.linalg.norm(wv[w]) != 0]

                # filter out the words that are not in the vocabulary
                embedding_words = [w for w in all_words if w in vocab]

                # compute WEAT and PROJ scores for every word
                word_scores = compute_word_scores(embedding_words, wv, embeddings)

                # create a smaller version of the word embedding vocabulary that's easier to check
                vocab_small = set(embedding_words)

            print("running bootstrapping")
            for i in tqdm(range(len(all_dfs_cds))):
                df_cds = all_dfs_cds[i]
                df_cs = all_dfs_cs[i]

                # compute the dataframe for this decade
                df_cds_dec = df_cds[df_cds['decade'] == decade]
                df_cs_dec = df_cs[df_cs['decade'] == decade]

                corr_words = set(df_cds[df_cds["freq"] >= MIN_FREQ]["word"]) \
                    & set(df_cs[df_cs["freq"] >= MIN_FREQ]["word"]) \
                    & set(df_cds[~np.isnan(df_cds['LOR'])]['word']) \
                    & set(df_cs[~np.isnan(df_cs['LOR'])]['word']) \
                    & vocab_small

                # drop all names if specified
                if not include_names:
                    corr_words = set([w for w in corr_words if w == w.lower() and w not in words_to_exclude])

                # compute the correlations here
                cds_corrs = get_corrs_fast(df_cds_dec, word_scores, corr_words)
                cs_corrs = get_corrs_fast(df_cs_dec, word_scores, corr_words)
                cds_corrs = [{"bootstrap_num": i, "speech_type": "CDS", "embeddings": embeddings, "decade": decade, **corr} for corr in cds_corrs]
                cs_corrs = [{"bootstrap_num": i, "speech_type": "CS", "embeddings": embeddings, "decade": decade, **corr} for corr in cs_corrs]
                all_corrs.extend(cds_corrs)
                all_corrs.extend(cs_corrs)

        if hist_embeddings:
            break

    # create and save the dataframe
    df_corrs = pd.DataFrame(all_corrs)
    return df_corrs


def compute_mean_associations_by_pair(df_fathers, df_mothers, wv, vocab, association_type, include_names):
    """
    Compute the mean gender associations in word embeddings by parent-child gender pair
    """
    # filter out the words that should not be included, either because they are names or they are not in the word
    # embeddings' vocabulary
    if not include_names:
        df_fathers['keep'] = df_fathers['word'].apply(lambda w: w.isalpha() and w == w.lower() and w in vocab)
        df_mothers['keep'] = df_mothers['word'].apply(lambda w: w.isalpha() and w == w.lower() and w in vocab)
    else:
        df_fathers['keep'] = df_fathers['word'].apply(lambda w: w in vocab)
        df_mothers['keep'] = df_mothers['word'].apply(lambda w: w in vocab)
    df_fathers = df_fathers[df_fathers['keep']]
    df_mothers = df_mothers[df_mothers['keep']]

    # compute the gender associations based on the association type
    if association_type == 'WEAT':
        df_fathers['assoc_score'] = df_fathers['word'].apply(lambda w: word_weat(w, wv))
        df_mothers['assoc_score'] = df_mothers['word'].apply(lambda w: word_weat(w, wv))
    else:
        female_vectors = [wv[word] for word in f_words_proj]
        male_vectors = [wv[word] for word in m_words_proj]
        gendered_pairs = [(f_words_proj[i], m_words_proj[i]) for i in range(10)]
        pca = doPCA(gendered_pairs, wv)
        gender_basis = pca.components_[0]

        if np.mean([projection(wv[w], gender_basis) for w in f_words_proj]) < np.mean([projection(wv[w], gender_basis) for w in m_words_proj]):
            gender_basis = - gender_basis

        df_fathers['assoc_score'] = df_fathers['word'].apply(lambda w: projection(wv[w], gender_basis))
        df_mothers['assoc_score'] = df_mothers['word'].apply(lambda w: projection(wv[w], gender_basis))

    # compute the average gender gender association for each parent-child gender pair
    df_father_son = df_fathers[df_fathers['child_sex'] == 'male']
    df_father_daughter = df_fathers[df_fathers['child_sex'] == 'female']
    df_mother_son = df_mothers[df_mothers['child_sex'] == 'male']
    df_mother_daughter = df_mothers[df_mothers['child_sex'] == 'female']

    father_son_avg = np.average(df_father_son['assoc_score'], weights=df_father_son['freq'])
    father_daughter_avg = np.average(df_father_daughter['assoc_score'], weights=df_father_daughter['freq'])
    mother_son_avg = np.average(df_mother_son['assoc_score'], weights=df_mother_son['freq'])
    mother_daughter_avg = np.average(df_mother_daughter['assoc_score'], weights=df_mother_daughter['freq'])

    return {'father_son': father_son_avg, 'father_daughter': father_daughter_avg, 'mother_son': mother_son_avg, 'mother_daughter': mother_daughter_avg}


def compute_mean_associations_by_pair_fast(df_fathers, df_mothers, vocab, word_scores, include_names):
    """
    Compute the mean gender associations in word embeddings by parent-child gender pair
    """
    # filter out the words that should not be included, either because they are names or they are not in the word
    # embeddings' vocabulary
    if not include_names:
        df_fathers['keep'] = df_fathers['word'].apply(lambda w: type(w) == str and w.isalpha() and w == w.lower() and w not in words_to_exclude and w in vocab)
        df_mothers['keep'] = df_mothers['word'].apply(lambda w: type(w) == str and w.isalpha() and w == w.lower() and w not in words_to_exclude and w in vocab)
    else:
        df_fathers['keep'] = df_fathers['word'].apply(lambda w: w in vocab)
        df_mothers['keep'] = df_mothers['word'].apply(lambda w: w in vocab)
    df_fathers = df_fathers[df_fathers['keep']]
    df_mothers = df_mothers[df_mothers['keep']]

    all_results = {}
    for assoc_type in ("WEAT", "PROJ"):
        df_fathers['assoc_score'] = df_fathers['word'].apply(lambda w: word_scores[assoc_type][w])
        df_mothers['assoc_score'] = df_mothers['word'].apply(lambda w: word_scores[assoc_type][w])

        # compute the average gender gender association for each parent-child gender pair
        df_father_son = df_fathers[df_fathers['child_sex'] == 'male']
        df_father_daughter = df_fathers[df_fathers['child_sex'] == 'female']
        df_mother_son = df_mothers[df_mothers['child_sex'] == 'male']
        df_mother_daughter = df_mothers[df_mothers['child_sex'] == 'female']

        father_son_avg = np.average(df_father_son['assoc_score'], weights=df_father_son['freq'])
        father_daughter_avg = np.average(df_father_daughter['assoc_score'], weights=df_father_daughter['freq'])
        mother_son_avg = np.average(df_mother_son['assoc_score'], weights=df_mother_son['freq'])
        mother_daughter_avg = np.average(df_mother_daughter['assoc_score'], weights=df_mother_daughter['freq'])

        all_results[assoc_type] = {'father_son': father_son_avg, 'father_daughter': father_daughter_avg, 'mother_son': mother_son_avg, 'mother_daughter': mother_daughter_avg}

    return all_results


def get_parent_child_results_bootstrap(all_dfs_fathers, all_dfs_mothers, embedding_types, include_names=False):
    """
    Compute the mean gender associations for each parent-child pair across bootstrapped dataframes
    """
    # compute a list of all the words that we will need to compute association metrics for
    all_words = compute_all_words(all_dfs_fathers, all_dfs_mothers, no_min_freq=True)

    all_corrs = []
    for embeddings in embedding_types:
        # load the gender associations
        wv, vocab = get_embeddings(embeddings)

        # filter out the words that are not in the vocabulary
        embedding_words = [w for w in all_words if w in vocab]

        # compute WEAT and PROJ scores for every word
        word_scores = compute_word_scores(embedding_words, wv, embeddings)

        # create a smaller version of the word embedding vocabulary that's easier to check
        vocab_small = set(embedding_words)

        print("doing bootstrapping")
        for i in tqdm(range(len(all_dfs_mothers))):
            df_mothers = all_dfs_mothers[i]
            df_fathers = all_dfs_fathers[i]

            # compute the mean associations and add rows to the dataframe
            mean_assocs = compute_mean_associations_by_pair_fast(df_fathers, df_mothers, vocab_small, word_scores, include_names)
            for assoc_type in mean_assocs:
                for speech_type in ('father_son', 'father_daughter', 'mother_son', 'mother_daughter'):
                    all_corrs.append({"bootstrap_num": i, "speech_type": speech_type, "embeddings": embeddings, "assoc": assoc_type, "corr": mean_assocs[assoc_type][speech_type]})

    return pd.DataFrame(all_corrs)


def compute_reduced_sim_correlation(words_to_track, all_vecs, full_sims):
    # apply t-SNE to reduce the dimensionality
    tsne = TSNE(n_components=2)
    reduced_vecs = tsne.fit_transform(all_vecs)

    # get the cosine similarities of the reduced embeddings
    reduced_sims = np.triu(cosine_similarity(reduced_vecs), k = 1) - 10 * np.tril(np.ones((60, 60)))
    reduced_sims_lst = [x.item() for x in np.nditer(reduced_sims) if x > -9]

    # return the Pearson correlation between the full and reduced similarities
    return pearsonr(full_sims, reduced_sims_lst)


def compute_similarity_correlations(words_to_track, embedding_types, n_runs=10000):

    all_rows = []
    for embeddings in embedding_types:

        # get the word embeddings
        wv, _ = get_embeddings(embeddings)

        # make a list of the full-dimensional embeddings
        all_vecs = [wv[w] for w in words_to_track]

        # compute the cosine similarities between the embeddings
        all_words_matrix = np.array([wv[w] for w in words_to_track])
        full_sims = np.triu(cosine_similarity(all_words_matrix), k = 1) - 10 * np.tril(np.ones((60, 60)))
        full_sims_lst = full_sims_mat = [x.item() for x in np.nditer(full_sims) if x > -9]

        # compute all the correlations
        print("computing correlations")
        for _ in tqdm(range(n_runs)):
            r, p = compute_reduced_sim_correlation(words_to_track, all_vecs, full_sims_lst)
            all_rows.append({"embeddings": embeddings, "r": r, "p": p})

    # create and return a dataframe
    df_sims = pd.DataFrame(all_rows)
    return df_sims
