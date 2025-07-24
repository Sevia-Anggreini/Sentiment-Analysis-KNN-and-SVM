import re
import string
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Inisialisasi stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def clean_text(text):
    # Semua fungsi cleaning digabung
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.strip().lower()
    return text

def normalize(text, normalization_dict):
    words = text.split()
    return ' '.join([normalization_dict.get(w, w) for w in words])

def tokenize(text):
    return re.findall(r'\b\w+\b', text)

def remove_stopwords(tokens, stopwords):
    return [w for w in tokens if w not in stopwords]

def stem(words):
    return ' '.join([stemmer.stem(w) for w in words])

def preprocess_dataframe(df, normalization_dict, stopwords):
    df['cleaned'] = df['Context'].astype(str).apply(clean_text)
    df['normalized'] = df['cleaned'].apply(lambda x: normalize(x, normalization_dict))
    df['tokens'] = df['normalized'].apply(tokenize)
    df['filtered'] = df['tokens'].apply(lambda x: remove_stopwords(x, stopwords))
    df['result_preprocessing'] = df['filtered'].apply(stem)
    return df
