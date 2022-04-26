"""
Read the switchboard corpus and save its data in a format suitable for subsequent bootstrapping and analysis
"""
import pickle
from ast import literal_eval
from nltk import word_tokenize
from convokit import Corpus, download
import pandas as pd

if __name__ == "__main__":

    corpus = Corpus(filename=download("switchboard-corpus"))

    all_m_words = []
    all_f_words = []
    for sid in corpus.get_speaker_ids():
        speaker = corpus.get_speaker(sid)
        speaker_gender = speaker.meta["sex"]
        for uid in speaker.get_utterance_ids():
            utterance = speaker.get_utterance(uid)
            speaker_words = [w.lower() for w in word_tokenize(utterance.text) if w.isalpha()]
            if speaker_gender == "MALE":
                all_m_words.extend(speaker_words)
            elif speaker_gender == "FEMALE":
                all_f_words.extend(speaker_words)

    #with open("../../data/switchboard_data/m_words.p", "wb") as fp:
    #    pickle.dump(all_m_words, fp)

    #with open("../../data/switchboard_data/f_words.p", "wb") as fp:
    #    pickle.dump(all_f_words, fp)

    print(f"Total word count: {len(all_m_words) + len(all_f_words)}")
