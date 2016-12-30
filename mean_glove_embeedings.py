import gensim
import numpy as np
import nltk
from nltk import tokenize
from data_handler import get_data
from preprocess_twitter import tokenize as tokenize_g
import pdb

#GLOVE_MODEL_FILE="/home/shashank/data/embeddings/GloVe/glove-twitter200-w2v"
GLOVE_MODEL_FILE="/home/shashank/data/embeddings/GloVe/glove-twitter25-w2v"
#GLOVE_MODEL_FILE="/home/pinkesh/DATASETS/glove-twitter/GENSIM.glove.twitter.27B.25d.txt"


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
                pass
                #print 'Skipping a word %s' %(w)
        if not len(_emb):
            #print 'BLANK mean, skipping this tweet'
            continue
            #pdb.set_trace()
        _emb = np.mean(np.asarray(_emb), axis=0)
        _emb = _emb.reshape((1, _emb.shape[0]))     # Convert to row vector
        X.append(_emb)
        Y.append(y_map[tweet['label']])

    #pdb.set_trace()
    X = np.concatenate(X, axis=0)
    print X.shape
    Y = np.asarray(Y)
    return X, Y
    
    

def tokenize(tweet):
    return MyTokenizer.tokenize(tweet)
    #return tokenize_g(tweet)

if __name__=="__main__":
    mean_glove()
