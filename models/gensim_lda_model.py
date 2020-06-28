import pandas as pd
import gensim 

from src.features import nlp
from src.features.tokenize import tokenize
from src.features.dictionary import create_dictionary
from src.data.prepare_data import read_sample
from src.features.utils import clean_up_text
from gensim.models import CoherenceModel


def load_doc() -> pd.DataFrame:
    return read_sample()

def lda_model(raw_file: pd.DataFrame) -> List[gensim.LdaModel,gensim.CoherenceModel,float]:
    doc       = clean_up_text(text)
    lemma     = tokenize(doc)
    id2word   = create_dictionary(lemma)
    corpus    = [id2word.doc2bow(text) for text in lemma]
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,id2word=id2word,num_topics=20, random_state=100,
                                    update_every=1,chunksize=100,passes=10,alpha='auto',per_word_topics=True)
    coherence = coherence_model(lda_model,lemma,id2word)
    perplexity = perplexity(lda_model,corpus)
    modelo = [lda_model,coherence, perplexity]
    return(modelo)
                                    
def coherence_model(model:gensim.LdaModel, lemma:List[List[str]], id2word:gensim.corpora) -> gensim.CoherenceModel:
    coherence_model_lda = CoherenceModel(model=lda_model, texts=lemma, dictionary=id2word, coherence='c_v')
    return coherence_model_lda.get_coherence()

def perplexity(model:gensim.LdaModel, corpus:List[List[str]]) -> float:
    return lda_model.log_perplexity(corpus)
    
 ## como usarlo
 ## doc = clean_up_text(load_doc())
 ## lda_model=lda_model(doc)
 ## pprint(lda_model[0].print_topics())
 ## pprint('coherence ', lda_model[1])
 ## pprint('perplexity ', lda_model[2])