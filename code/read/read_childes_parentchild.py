import pylangacq as pla
from os.path import abspath
from collections import defaultdict
import pickle

# relative path to CHILDES
corpora_path = '../../data/ChildesCorpus/'
output_path = "../../data/raw_data/"
# corpora with SES information, to exclude
corpora_with_ses = ["Hall"]
# min and max age for children in the corpus
min_age = 1
max_age = 5



if __name__ == "__main__":
    # read all the paths to chat files from all_coprora_paths
    dir_endings = []
    with open('../../data/all_corpora_paths.txt', 'r') as fp:
        for line in fp:
            dir_endings.append(line[1:-1])

    # initialize dictionaries to store words
    m_m_cds = defaultdict(list)
    m_f_cds = defaultdict(list)
    f_m_cds = defaultdict(list)
    f_f_cds = defaultdict(list)

    # initialize variables to store stats on the corpus
    total_tokens = 0
    num_boys = 0
    num_girls = 0
    boys_yearly = [0 for i in range(6)]
    girls_yearly = [0 for i in range(6)]

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
        # verify that the child is in the right age range
        if len(curr_age) == 0 or int(curr_age) < min_age or int(curr_age) > max_age:
            continue

        total_tokens += len(chat.words())

        # extend the relevant lists of words
        if chat.participants()[ap]['CHI']['sex'] == 'male':
            # increment the relevant counts
            num_boys += 1
            boys_yearly[int(curr_age)] += 1
            for p in chat.participants()[ap]:
                if p != 'CHI':
                    if chat.participants()[ap][p]['sex'] == 'male':
                        m_m_cds[curr_age].extend(chat.words(participant=p))
                    elif chat.participants()[ap][p]['sex'] == 'female':
                        f_m_cds[curr_age].extend(chat.words(participant=p))
        elif chat.participants()[ap]['CHI']['sex'] == 'female':
            num_girls += 1
            girls_yearly[int(curr_age)] += 1
            for p in chat.participants()[ap]:
                if p != 'CHI':
                    if chat.participants()[ap][p]['sex'] == 'male':
                        m_f_cds[curr_age].extend(chat.words(participant=p))
                    elif chat.participants()[ap][p]['sex'] == 'female':
                        f_f_cds[curr_age].extend(chat.words(participant=p))

    # save the word lists to dictionarise
    with open(output_path + 'm_m_cds.p', 'wb') as fp:
        pickle.dump(m_m_cds, fp)

    with open(output_path + 'f_m_cds.p', 'wb') as fp:
        pickle.dump(f_m_cds, fp)

    with open(output_path + 'm_f_cds.p', 'wb') as fp:
        pickle.dump(m_f_cds, fp)

    with open(output_path + 'f_f_cds.p', 'wb') as fp:
        pickle.dump(f_f_cds, fp)
