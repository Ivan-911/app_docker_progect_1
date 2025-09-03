import pandas as pd
import re
from natasha import Segmenter, NewsEmbedding, NewsMorphTagger, MorphVocab, Doc
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download('stopwords')

segmenter = Segmenter()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
morph_vocab = MorphVocab()

def lemmatize_natasha(text):
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    for token in doc.tokens:
        token.lemmatize(morph_vocab)
    lemmas = [token.lemma for token in doc.tokens if token.lemma not in stopwords.words('russian')]    
    return ' '.join(lemmas)

def preprocess_text(text):
    # Отделяем эмодзи пробелами
    text = re.sub(r'([\U0001F600-\U0001F64F])', r' \1 ', text)
    # Удаляем повторяющиеся знаки препинания
    text = re.sub(r'([!?.])\1+', r'\1', text)
    # Приводим к нижнему регистру
    text = text.lower()
    return text

def lemmatize_natasha_batch(texts):
    return [lemmatize_natasha(text) for text in texts]
    
def preprocess_text_batch(texts):
    return [preprocess_text(text) for text in texts]
