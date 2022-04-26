import pandas as pd
import numpy as np
from collections import defaultdict

def make_yearly_df(m_words, f_words):
    """
    Make a dataframe with yearly data on words' frequencies and gender associations in speech
    """
    df = pd.DataFrame(columns=['age', 'word', 'pfgw', "OR", "LOR", 'freq'])
    
    # create a list of all words for each gender
    all_m_words = []
    all_f_words = []
    all_words = set([])
    for age in range(1,6):
        age = str(age)
        all_m_words.extend(m_words[str(age)])
        all_f_words.extend(f_words[str(age)])
    # list of all words said overall
    all_words = all_words | set(all_m_words) | set(all_f_words)
    all_words = list(all_words)    
    
    for age in range(1,6):
        # initialize dictionaries
        pfgw = {}
        f_freq = {}
        m_freq = {}
        odds_ratio = {}
        log_odds_ratio = {}
        for word in all_words:
            f_freq[word] = 0
            m_freq[word] = 0
            pfgw[word] = 0.5

        # count the number of times each word is said to/by male/female speakers
        for word in m_words[str(age)]:
            if word.isalpha() and len(word) > 1:
                m_freq[word] += 1
        for word in f_words[str(age)]:
            if word.isalpha() and len(word) > 1:
                f_freq[word] += 1

        # get the total number of times any word was said to male/female speakers
        f_total = sum([f_freq[word] for word in  f_freq])
        m_total = sum([m_freq[word] for word in m_freq])
        for word in all_words:
            if m_freq[word] + f_freq[word] > 0:
                pfgw[word] = (f_freq[word]/f_total) / ((f_freq[word]/f_total) + (m_freq[word]/m_total))
            if f_freq[word] > 0 and m_freq[word] > 0:
                odds_ratio[word] = ((f_freq[word]) / (f_total - f_freq[word])) / (m_freq[word] / (m_total - m_freq[word]))
                log_odds_ratio[word] = np.log(odds_ratio[word])
            else:
                odds_ratio[word] = 1
                log_odds_ratio[word] = 0

        # create lists from each statistic calculated
        ages = [age for word in all_words]
        words = [word for word in all_words]
        pfgws = [pfgw[word] for word in all_words]
        freqs = [f_freq[word] + m_freq[word] for word in all_words]
        ORs = [odds_ratio[word] for word in all_words]
        LORs = [log_odds_ratio[word] for word in all_words]

        # extend the current dataframe with data from this age
        data = {'age': ages, 'word': words, 'pfgw': pfgws, 'freq': freqs, "OR": ORs, "LOR": LORs}
        df_ext = pd.DataFrame(data)
        df = df.append(df_ext, ignore_index=True, sort=False)
        
    return df


def make_aggregate_df(m_words, f_words):
    """
    Create a dataframe with aggregate statistics about words' frequencies and gender associations
    for children of all ages
    """
    # count the words said to/by male/female children     
    all_m_words = []
    all_f_words = []
    if type(m_words) != list and type(m_words) != np.ndarray:
        for age in range(1,6):
            age = str(age)
            all_m_words.extend(m_words[age])
            all_f_words.extend(f_words[age])
    else:
        all_m_words = m_words
        all_f_words = f_words
    # make a list of all the words said to anyone in the corpus
    all_words = set(all_m_words) | set(all_f_words)
    all_words = list(all_words)

    # initialize dictionaries
    pfgw = {}
    f_freq = {}
    m_freq = {}
    odds_ratio = {}
    log_odds_ratio = {}
    for word in all_words:
        f_freq[word] = 0
        m_freq[word] = 0
        pfgw[word] = 0.5

    # count the number of times each word was said to/by a male/female child
    for w in all_m_words:
        m_freq[w] += 1
    for w in all_f_words:
        f_freq[w] += 1

    # compute the number of times any word was said to/by a male/female child
    m_total = sum([m_freq[w] for w in all_words])
    f_total = sum([f_freq[w] for w in all_words])

    # compute gender association metrics for each word
    for word in all_words:
        if m_freq[word] + f_freq[word] > 0:
            pfgw[word] = (f_freq[word]/f_total) / ((f_freq[word]/f_total) + (m_freq[word]/m_total))
            if f_freq[word] > 0 and m_freq[word] > 0:
                odds_ratio[word] = ((f_freq[word]) / (f_total - f_freq[word])) / (m_freq[word] / (m_total - m_freq[word]))
                log_odds_ratio[word] = np.log(odds_ratio[word])
            else:
                odds_ratio[word] = 1
                log_odds_ratio[word] = 0

    # turn the gender associations into lists
    pfgws = [pfgw[word] for word in all_words]
    freqs = [f_freq[word] + m_freq[word] for word in all_words]
    ORs = [odds_ratio[word] for word in all_words]
    LORs = [log_odds_ratio[word] for word in all_words]

    # make a dataframe
    cols = {'word': all_words,  'pfgw': pfgws, 'freq': freqs, "OR": ORs, "LOR": LORs}
    df = pd.DataFrame(cols)

    return df


def make_parent_df(m_words, f_words):
    """
    Here, m_words and f_words are nested dictionaries of form words[class][age]
    """
    # count the occurences of all the words said to male and female children
    all_m_words = []
    all_f_words = []
    for age in range(1,6):
        age = str(age)
        all_m_words.extend(m_words[age])
        all_f_words.extend(f_words[age])
    # compute all of the words said in the corpus
    all_words = set(all_m_words) | set(all_f_words)
    all_words = list(all_words)

    # initialize dictionaries
    f_freq = defaultdict(int)
    m_freq = defaultdict(int)

    # count the frequencies of each word
    for w in all_m_words:
        m_freq[w] += 1
    for w in all_f_words:
        f_freq[w] += 1

    # turn the frequencies into dataframes and merge them
    cols = {'child_sex': ['male' for _ in all_words], 'word': all_words,'freq': [m_freq[w] for w in all_words]}
    df_male = pd.DataFrame(cols)
    cols = {'child_sex': ['female' for _ in all_words], 'word': all_words,'freq': [f_freq[w] for w in all_words]}
    df_female = pd.DataFrame(cols)
    df = df_male.append(df_female, ignore_index=True)

    return df

def make_df_by_decade(m_words, f_words):
    """
    Make a dataframe with gender associations in speech for each decade
    """
    # make a list of all the words in the corpus
    all_words = []
    for decade in ('70s', '80s', '90s'):
        all_words.extend(m_words[decade])
        all_words.extend(f_words[decade])
    all_words = [word for word in set(all_words) if word.isalpha() and len(word) > 1]
    
    # initialize a dataframe
    df_by_decade = pd.DataFrame(columns=['decade', 'word', 'pfgw', 'freq', 'OR', 'LOR'])

    for decade in ('70s', '80s', '90s'):
        # initialize dictionaries to store statistics
        pfgw = {}
        f_freq = {}
        m_freq = {}
        odds_ratio = {}
        log_odds_ratio = {}
        for word in all_words:
            f_freq[word] = 0
            m_freq[word] = 0
            pfgw[word] = 0.5

        # compute the frequencies with which words are said to/by male and female children
        for word in m_words[decade]:
            if word.isalpha() and len(word) > 1:
                m_freq[word] += 1
        for word in f_words[decade]:
            if word.isalpha() and len(word) > 1:
                f_freq[word] += 1

        # get the total number of any words said to male and female children
        f_total = sum([f_freq[word] for word in  f_freq])
        m_total = sum([m_freq[word] for word in m_freq])

        # compute the gender association statistics for boys and girls
        for word in all_words:
            if m_freq[word] + f_freq[word] > 0:
                pfgw[word] = (f_freq[word]/f_total) / ((f_freq[word]/f_total) + (m_freq[word]/m_total))
            if f_freq[word] > 0 and m_freq[word] > 0:
                odds_ratio[word] = ((f_freq[word]) / (f_total - f_freq[word])) / (m_freq[word] / (m_total - m_freq[word]))
                log_odds_ratio[word] = np.log(odds_ratio[word])
            else:
                odds_ratio[word] = 1
                log_odds_ratio[word] = 0

        # make lists for conversion to dataframe
        decades = [decade for word in all_words]
        words = [word for word in all_words]
        pfgws = [pfgw[word] for word in all_words]
        freqs = [f_freq[word] + m_freq[word] for word in all_words]
        ORs = [odds_ratio[word] for word in all_words]
        LORs = [log_odds_ratio[word] for word in all_words]

        # extend the dataframe
        data = {'decade': decades, 'word': words, 'pfgw': pfgws, 'freq': freqs, 'OR': ORs, 'LOR': LORs}
        df_ext = pd.DataFrame(data)
        df_by_decade = df_by_decade.append(df_ext, ignore_index=True, sort=False)
    return df_by_decade

def make_class_df(m_words, f_words):
    """
    Compute a dataframe with speech statistics by class
    Here, m_words and f_words are nested dictionaries of form words[class][age]
    """
    classes = ['Black,WC', 'Black,UC', 'White,WC', 'White,UC']

    # count all the words that are said to/by male and female children    
    all_m_words = defaultdict(list)
    all_f_words = defaultdict(list)
    all_words = set([])
    for cla in classes:
        all_m_words[cla].extend(m_words[cla])
        all_f_words[cla].extend(f_words[cla])
        all_words = all_words | set(all_m_words[cla]) | set(all_f_words[cla])
    
    # make a list of all the words in the corpus
    all_words = list(all_words)

    df_all = pd.DataFrame()

    # iterate over classes
    for cla in classes:

        # Initialize dictionaries
        pfgw = {}
        f_freq = {}
        m_freq = {}
        odds_ratio = {}
        log_odds_ratio = {}
        for word in all_words:
            f_freq[word] = 0
            m_freq[word] = 0
            pfgw[word] = 0.5

        # count the occurrences of each word said to/by boys/girls
        for w in all_m_words[cla]:
            m_freq[w] += 1
        for w in all_f_words[cla]:
            f_freq[w] += 1

        # get the total number of times any word was said to/by boys/girls
        m_total = sum([m_freq[w] for w in all_words])
        f_total = sum([f_freq[w] for w in all_words])

        # compute gender association statistics for each word
        for word in all_words:
            if m_freq[word] + f_freq[word] > 0:
                pfgw[word] = (f_freq[word]/f_total) / ((f_freq[word]/f_total) + (m_freq[word]/m_total))
            if f_freq[word] > 0 and m_freq[word] > 0 and f_total - f_freq[word] != 0 and m_total - m_freq[word] != 0:
                odds_ratio[word] = ((f_freq[word]) / (f_total - f_freq[word])) / (m_freq[word] / (m_total - m_freq[word]))
                log_odds_ratio[word] = np.log(odds_ratio[word])
            else:
                odds_ratio[word] = 1
                log_odds_ratio[word] = 0

        # create a dataframe with the computed statistics
        pfgws = [pfgw[word] for word in all_words]
        freqs = [f_freq[word] + m_freq[word] for word in all_words]
        ORs = [odds_ratio[word] for word in all_words]
        LORs = [log_odds_ratio[word] for word in all_words]
        cols = {'word': all_words,  'pfgw': pfgws, 'freq': freqs, "OR": ORs, "LOR": LORs, 'class': [cla for _ in range(len(all_words))]}
        df = pd.DataFrame(cols)
        df_all = df_all.append(df)
            
    return df_all
