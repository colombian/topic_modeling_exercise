from typing import List

from gensim.corpora import Dictionary


def create_dictionary(documents: List[List[str]]):
    return Dictionary(documents)


def term_document_matrix(documents>List[List[str]], dictionary: Dictionary)->List[Tuple[int,int]:
    return [dictionary.doc2bow(text) for text in documents]