import pylangacq as pla
from os.path import abspath
import pickle
from collections import defaultdict
import csv

# relative path to CHILDES
corpora_path = '../../data/ChildesCorpus/'
output_path = "../../data/raw_data/"
# corpora with SES information, to exclude
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
            dir_endings.append(line[1:-1])

    # initialize dictionaries to store words
    m_words_cds = defaultdict(list)
    m_words_cs = defaultdict(list)
    f_words_cds = defaultdict(list)
    f_words_cs = defaultdict(list)

    # iterate over each chat file
    for line in dir_endings:

        # skip this line if it is in the list of corpora with SES information
        if line.split('/')[1] in corpora_with_ses:
            continue

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

        # extend the current list of words
        if chat.participants()[ap]['CHI']['sex'] == 'male':
            total_boys += 1
            boys_tokens_by_age[curr_age] += len(chat.words())
            for p in chat.participants()[ap]:
                # add the words to the cs dictionary if they are from the child and the
                # cs dictionary if they are from a non-child
                if p == 'CHI':
                    m_words_cs[curr_age].extend(chat.words(participant=p))
                else:
                    m_words_cds[curr_age].extend(chat.words(participant=p))
        elif chat.participants()[ap]['CHI']['sex'] == 'female':
            total_girls += 1
            girls_tokens_by_age[curr_age] += len(chat.words())
            for p in chat.participants()[ap]:
                if p == 'CHI':
                    f_words_cs[curr_age].extend(chat.words(participant=p))
                else:
                    f_words_cds[curr_age].extend(chat.words(participant=p))

    # save the dictionaries to pickle files
    with open(output_path + 'm_words_cs.p', 'wb') as fp:
        pickle.dump(m_words_cs, fp)

    with open(output_path + 'f_words_cs.p', 'wb') as fp:
        pickle.dump(f_words_cs, fp)

    with open(output_path + 'm_words_cds.p', 'wb') as fp:
        pickle.dump(m_words_cds, fp)

    with open(output_path + 'f_words_cds.p', 'wb') as fp:
        pickle.dump(f_words_cds, fp)

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
