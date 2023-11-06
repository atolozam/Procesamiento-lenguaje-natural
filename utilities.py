import nltk
from nltk import word_tokenize
from nltk.book import *
from nltk.corpus import stopwords

from string import punctuation
import re

from urllib import request
from bs4 import BeautifulSoup

import numpy as np

# Importar en caso que sea un archivo txt

def import_txt(path):
    with open(path, 'r') as f:
        text = f.read()
    return text

# Importar en caso que sea un archivo html

def import_html(url):
    req = request.urlopen(url)
    html = req.read().decode('utf8')
    soup = BeautifulSoup(html.text, 'html.parser')
    text = soup.get_text()
    return text


# Hace un ligera limpieza al texto

def clean_text(text):
    text=re.sub(r'<.*?>', ' ', text)
    text=re.sub(r'\s+', ' ', text)
    return text

# Tokeniza el texto

def tokenize(text):

    pattern =   r'''(?x)              # Flag para iniciar el modo verbose
                (?:[A-Z]\.)+          # Hace match con abreviaciones como U.S.A.
                | \w+(?:-\w+)*        # Hace match con palabras que pueden tener un guión interno
                | \$?\d+(?:\.\d+)?%?  # Hace match con dinero o porcentajes como $15.50 o 100%
                | \.\.\.              # Hace match con puntos suspensivos
                | [][.,;"'?():-_`]    # Hace match con signos de puntuación
    '''

    tokens = nltk.nltk.regexp_tokenize(text, pattern)
    return tokens
    

# Limpia los tokens para tener un vocabulario más limpio (sin stopwords, puntuación, etc.)

def clean_tokens(lenguaje, tokens):

    stopwd = stopwords.words(str(lenguaje))

    tokens = [token for token in tokens if token not in punctuation]
    tokens = [token for token in tokens if token not in stopwd]
    tokens = [token.lower() for token in tokens]
    return tokens

# Crea el vocabulario

def create_vocab(tokens):
    vocab = sorted(set(tokens))
    return vocab

# Crea el diccionario de frecuencias

def create_freq(tokens):
    freq = nltk.FreqDist(tokens)
    freq_organize = np.arange(len(freq))

    return freq_organize