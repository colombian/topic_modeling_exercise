from typing import List

from src.features import nlp
from src.features.utils import sent_to_words, remove_stopwords, build_bigrams, lemmatization

def tokenize(documents: List[str]) -> List[List[str]]:

    document_words = list(sentences_to_words(documents))
    document_words = remove_stopwords(document_words)
    document_words = build_bigrams(document_words)
    document_words = lemmatization(nlp, document_words)

    return document_words