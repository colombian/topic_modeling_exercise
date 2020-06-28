from typing import List

from gensim.corpora import Dictionary


def create_dictionary(documents: List[List[str]]) -> gensim.corpora:
    return Dictionary(documents)
