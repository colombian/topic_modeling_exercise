import pickle

import numpy as np

from src.data.split import split_data
from src.features.dictionary import create_dictionary


def train():
    with open(r"../data/interim/positive_words.pkl", "rb") as input_file:
        positive_docs = pickle.load(input_file)

    with open(r"../data/interim/negative_words.pkl", "rb") as input_file:
        negative_docs = pickle.load(input_file)

    negative_words = [item for sublist in negative_docs for item in sublist]
    positive_words = [item for sublist in positive_docs for item in sublist]

    dictionary = create_dictionary([negative_words, positive_words])

    negative_split = split_data(negative_words)
    positive_split = split_data(positive_words)

    negative_bow = dictionary.doc2bow(negative_split['train'])
    positive_bow = dictionary.doc2bow(positive_split['train'])

    total_negative = len(negative_split['train']) + len(negative_bow)
    total_positive = len(positive_split['train']) + len(positive_bow)

    negative_word_probs = {}
    for id, count in negative_bow:
        negative_word_probs[dictionary[id]] = {
            'id': id,
            'logprob': np.log((count + 1) / total_negative),
        }

    positive_word_probs = {}
    for id, count in positive_bow:
        positive_word_probs[dictionary[id]] = {
            'id': id,
            'logprob': np.log((count + 1) / total_positive),
        }