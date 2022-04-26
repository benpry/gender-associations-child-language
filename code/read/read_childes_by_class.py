import pylangacq as pla
from os.path import abspath
from collections import defaultdict
import pandas as pd
import numpy as np
import pickle


# relative path to CHILDES
corpora_path = '../../data/ChildesCorpus/'
output_path = "../../data/raw_data/"
# corpora with SES information, to include
corpora_with_ses = ["Hall"]
# min and max age for children in the corpus
min_age = 1
max_age = 5

# statistics to compute about the size of the corpus
total_boys = 0
total_girls = 0
total_tokens = 0
boys_tokens_by_age = defaultdict(int)
girls_tokens_by_age = defaultdict(int)
all_ages = []

if __name__ == "__main__":
    # read all the paths to chat files from all_coprora_paths
    dir_endings = []
    with open('../../data/all_corpora_paths.txt', 'r') as fp:
        for line in fp:
            if line.split("/")[1] in corpora_with_ses:
                dir_endings.append(line[1:-1])

    # initialize dictionaries to store words
    m_words_cds = defaultdict(list)
    m_words_cs = defaultdict(list)
    f_words_cds = defaultdict(list)
    f_words_cs = defaultdict(list)

    # variables to keep track of stats on the corpus
    total_tokens = 0
    num_boys = 0
    num_girls = 0
    boys_yearly = [0 for i in range(6)]
    girls_yearly = [0 for i in range(6)]

    for line in dir_endings:
        # read the file, except a parsing error
        try:
            f = corpora_path + line
            chat = pla.read_chat(f)
            ap = abspath(f)
        except ValueError:
            print(f, 'led to a parsing error.')
            continue

        # if there is no child, skip this chat
        if ap not in chat.participants() or 'CHI' not in chat.participants()[ap].keys():
            continue

        curr_age = chat.participants()[ap]['CHI']['age'].split(';')[0]
        all_ages.append(curr_age)
        # verify that the child is in the right age range
        if len(curr_age) == 0 or int(curr_age) < min_age or int(curr_age) > max_age:
            continue

        total_tokens += len(chat.words())

        # get the SES and current age of the child
        ses = chat.participants()[ap]['CHI']['SES']
        curr_age = chat.participants()[ap]['CHI']['age'].split(';')[0]

        # extend the relevant word lists
        if chat.participants()[ap]['CHI']['sex'] == 'male':
            total_boys += 1
            boys_tokens_by_age[curr_age] += len(chat.words())
            boys_yearly[int(curr_age)] += 1
            m_words_cs[ses].extend(chat.words(participant='CHI'))
            for p in chat.participants()[ap]:
                if p != 'CHI':
                    m_words_cds[ses].extend(chat.words(participant=p))
        elif chat.participants()[ap]['CHI']['sex'] == 'female':
            total_girls += 1
            girls_tokens_by_age[curr_age] += len(chat.words())
            f_words_cs[ses].extend(chat.words(participant='CHI'))
            for p in chat.participants()[ap]:
                if p != 'CHI':
                    f_words_cds[ses].extend(chat.words(participant=p))

    # save the word lists to a file
    with open(output_path + 'm_words_cds_class.p', 'wb') as fp:
        pickle.dump(m_words_cds, fp)

    with open(output_path + 'm_words_cs_class.p', 'wb') as fp:
        pickle.dump(m_words_cs, fp)

    with open(output_path + 'f_words_cds_class.p', 'wb') as fp:
        pickle.dump(f_words_cds, fp)

    with open(output_path + 'f_words_cs_class.p', 'wb') as fp:
        pickle.dump(f_words_cs, fp)

    print(f"total tokens: {total_tokens}")
    print(f"total boys: {total_boys}")
    print(f"total girls: {total_girls}")
    print(f"total children: {total_boys + total_girls}")
    print("boys by age")
    print(boys_tokens_by_age)
    print("girls by age")
    print(girls_tokens_by_age)
    percentage_girls_tokens_by_age = {}
    for age in boys_tokens_by_age:
        percentage_girls_tokens_by_age[age] = girls_tokens_by_age[age] / (girls_tokens_by_age[age] + boys_tokens_by_age[age])
    print("percentage girls tokens by age")
    print(percentage_girls_tokens_by_age)
    total_tokens_by_age = {}
    for age in boys_tokens_by_age:
        total_tokens_by_age[age] = boys_tokens_by_age[age] + girls_tokens_by_age[age]
    print("total tokens by age")
    print(total_tokens_by_age)
    print("all ages")
    print(list(set(all_ages)))
