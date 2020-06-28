from typing import Generator, List

from gensim.models import Phrases
from gensim.models.phrases import Phraser
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
from spacy.lang.en import English

def clean_up_text(df:pd.DataFrame) -> List[str]:
    data = df.content.values.tolist() # convertir a lista
    data = [re.sub(r'\S*@\S*\s?', '', sent) for sent in data] # Quitar e-mail
    data = [re.sub(r'\s+', ' ', sent) for sent in data] # quitar enters (new line)
    data = [re.sub(r"\'", "", sent) for sent in data] # quitar comillas
    return(data)
    

def sent_to_words(sentences: List[str]) -> List[List[str]]:
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))
        
def remove_stopwords(texts: List[List[str]]) -> List[List[str]]:
    stop_words = stopwords.words('english')
    stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
        
def core_bigram(data_words: List[List[str]], min_count:int=5,threshold:int=100):
    bigram = gensim.models.Phrases(data_words, min_count=min_count, threshold=threshold)
    return(bigram)

def build_bigrams(data_words: List[List[str]], min_count:int=5,threshold:int=100) -> List[List[str]]:
    bigram = core_bigram(data_words,min_count,threshold)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    return [bigram_mod[doc] for doc in data_words]
    
def build_trigrams(data_words:List[List[str]], threshold:int=100) -> List[List[str]]:
    bigram = core_bigram(data_words,min_count,threshold)
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    return[trigram_mod[bigram_mod[doc]] for doc in data_words]
    
def lemmatization(texts:List[List[str]], allowed_postags:List[str]=['NOUN', 'ADJ', 'VERB', 'ADV']) -> List[List[str]]:
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out
