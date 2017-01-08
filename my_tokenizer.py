from string import punctuation
from preprocess_twitter import tokenize as tokenizer_g
from gensim.parsing.preprocessing import STOPWORDS


def glove_tokenize(text):
    text = tokenizer_g(text)
    text = ''.join([c for c in text if c not in punctuation])
    words = text.split()
    words = [word for word in words if word not in STOPWORDS]
    return words

