import pylangacq as pla
from os.path import abspath
import pickle
from tqdm import tqdm
import pandas as pd
from processing_utils import make_aggregate_df
import numpy as np
import re

corpus_path = '../../data/SantaBarbaraCorpus/'
output_folder = '../../data/santa_barbara_data/'
names_folder = "../../data/name_genders/"

# the number of bootstrapped instances to create
n_iter = 10000

if __name__ == "__main__":

    # random seed for reproducibility
    np.random.seed(5678)

    # First, read the corpus

    # create a list of all the filenames in the corpus
    dir_endings = []
    for i in range(1,10):
        dir_endings.append('SBC00{}.cha'.format(i))
        
    for i in range(10,61):
        dir_endings.append('SBC0{}.cha'.format(i))

    # initialize variables
    total_tokens = 0
    chat_dict = {}

    # read each chat file
    for line in dir_endings:
        try:
            f = corpus_path + line
            chat = pla.read_chat(f)
            ap = abspath(f)
        except ValueError:
            print(f, 'caused an error.')
            continue
        
        # index by the chat number
        chat_dict[line[4:6]] = {}
            
        # add each participant's data
        for p in chat.participants()[ap]:
            p_name = chat.participants()[ap][p]['participant_name']
            
            chat_dict[line[4:6]][p_name] = chat.words(participant=p)

    # categorize the words by gender based on the name of the speaker
    male_names = [line[:-1] for line in open(names_folder + 'male.txt')][6:]
    female_names = [line[:-1] for line in open(names_folder + 'female.txt')][6:]
    male_words = []
    female_words = []
    for k in chat_dict:
        for name in chat_dict[k]:
            if name.title() in male_names:
                # remove non-alphanumeric characters and hyphens
                male_words.extend([re.sub(r'⌈|⌋|⌊|⌉|1|2|3|4|5|6|7|8|9', '', w.lower()).strip('-') for w in chat_dict[k][name]])
            elif name.title() in female_names: 
                female_words.extend([re.sub(r'⌈|⌋|⌊|⌉|1|2|3|4|5|6|7|8|9', '', w.lower()).strip('-') for w in chat_dict[k][name]])
     
    male_words = [w for w in male_words if w != '']
    female_words = [w for w in female_words if w != '']

    # Save the male and female words to pickle files
    with open(output_folder + 'adult_male_words.p', 'wb') as fp:
        pickle.dump(male_words, fp)
    with open(output_folder +  'adult_female_words.p', 'wb') as fp:
        pickle.dump(female_words, fp)

    # Next, create bootstrapped word tables
    sample_n = int((len(male_words) + len(female_words)) / 2)
    for iteration in tqdm(range(n_iter)):
        # sample randomly with replacement
        m_bs = np.random.choice(male_words, size=sample_n, replace=True)
        f_bs = np.random.choice(female_words, size=sample_n, replace=True)

        # make a dataframe and save it to csv
        df_bs = make_aggregate_df(m_bs, f_bs)
        df_bs.to_csv(output_folder + f"/bootstrapped/df_santabarbara_bs{iteration}.csv")
