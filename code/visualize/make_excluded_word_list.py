"""
This is a simple script that formats the list of excluded words for LaTeX
"""

if __name__ == "__main__":

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

    all_words = sorted(list(set(f_words_proj + m_words_proj + m_words_weat + f_words_weat + gendered_words)))

    all_words_str = ""
    for word in all_words:
        all_words_str += f"``{word}'', "

    print(all_words_str)
