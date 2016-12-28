import gensim
import numpy as np
import nltk
from nltk import tokenize
from data_handler import get_data

GLOVE_MODEL_FILE="/home/grim/DATASETS/glove-twitter/GENSIM.glove.twitter.27B.25d.txt"

MyTokenizer = tokenize.casual.TweetTokenizer(strip_handles=True, reduce_len=True)

def mean_glove():
    # Processing
    tweets = get_data()
    X, Y = [], []

    y_map = {
            'none': 0,
            'racism': 1,
            'sexism': 2
        }

    # Embeeding Model
    model = gensim.models.Word2Vec.load_word2vec_format(GLOVE_MODEL_FILE)
    for tweet in tweets:
        _emb = []
        words = tokenize(tweet['text'])
        for w in words:
            try:
                word_emb = model[w]
                _emb.append(word_emb)
            except:
                print 'Skipping a word %s' %(w)
        _emb = np.mean(np.array(_emb), axis=0)
        X.append(_emb)
        Y.append(y_map[tweet['label']])
    X = np.asarray(tweet_embeedings)
    Y = np.asarray(Y)
    return X, Y
    
    

def tokenize(tweet):
    return MyTokenizer.tokenize(tweet)

if __name__=="__main__":
    mean_glove()
