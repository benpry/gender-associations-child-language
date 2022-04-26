import pylangacq as pla
from os.path import abspath
import pickle
from collections import defaultdict


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

        # if there is no date of recording, skip this chat
        if chat.dates_of_recording()[ap] is None:
            continue

        # get the child's age in years
        curr_age = chat.participants()[ap]['CHI']['age'].split(';')[0]

        # verify that the child is in the right age range
        if len(curr_age) == 0 or int(curr_age) < min_age or int(curr_age) > max_age:
            continue

        # get the decade of the recording
        year = chat.dates_of_recording()[ap][0][0]
        if year >= 1970 and year <= 1979:
            decade = '70s'
        elif year >= 1980 and year <= 1989:
            decade = '80s'
        elif year >= 1990 and year <= 1999:
            decade = '90s'
        else:
            continue

        # extend the relevant list of words based on gender and decade
        if chat.participants()[ap]['CHI']['sex'] == 'male':
            for p in chat.participants()[ap]:
                if p != 'CHI':
                    m_words_cds[decade].extend(chat.words(participant=p))
                else:
                    m_words_cs[decade].extend(chat.words(participant=p))

        elif chat.participants()[ap]['CHI']['sex'] == 'female':
            for p in chat.participants()[ap]:
                if p != 'CHI':
                    f_words_cds[decade].extend(chat.words(participant=p))
                else:
                    f_words_cs[decade].extend(chat.words(participant=p))

    # save the dictionaries to pickle files
    with open(output_path + 'm_words_cds_bydecade.p', 'wb') as fp:
        pickle.dump(m_words_cds, fp)

    with open(output_path + 'f_words_cds_bydecade.p', 'wb') as fp:
        pickle.dump(f_words_cds, fp)

    with open(output_path + 'm_words_cs_bydecade.p', 'wb') as fp:
        pickle.dump(m_words_cs, fp)

    with open(output_path + 'f_words_cs_bydecade.p', 'wb') as fp:
        pickle.dump(f_words_cs, fp)
