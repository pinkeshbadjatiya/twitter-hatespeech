from data_handler import get_data
import sys
import numpy as np
from preprocess_twitter import tokenize as tokenizer_g
import pdb
from nltk import tokenize
from sklearn.metrics import make_scorer, f1_score, accuracy_score, recall_score, precision_score, classification_report, precision_recall_fscore_support
from sklearn.ensemble  import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
import pdb
from sklearn.metrics import make_scorer, f1_score, accuracy_score, recall_score, precision_score, classification_report, precision_recall_fscore_support
from sklearn.utils import shuffle
from sklearn.ensemble  import GradientBoostingClassifier, RandomForestClassifier
from gensim.parsing.preprocessing import STOPWORDS
from sklearn.model_selection import KFold
from sklearn import linear_model
from sklearn.utils import shuffle
import codecs
import operator
import gensim, sklearn
from collections import defaultdict
from batch_gen import batch_gen


### Preparing the text data
texts = []  # list of text samples
labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids
label_map = {
        'none': 0,
        'racism': 1,
        'sexism': 2
    }
tweet_data = get_data()
for tweet in tweet_data:
    texts.append(tweet['text'])
    labels.append(label_map[tweet['label']])
print('Found %s texts. (samples)' % len(texts))


# Load the orginal glove file
# SHASHANK files
#GLOVE_MODEL_FILE="/home/shashank/data/embeddings/GloVe/glove-twitter25-w2v"


# PINKESH files
EMBEDDING_DIM = sys.argv[1]
GLOVE_MODEL_FILE="/home/pinkesh/DATASETS/glove-twitter/GENSIM.glove.twitter.27B." + EMBEDDING_DIM+"d.txt"


SEED=42
MAX_NB_WORDS = None
#MAX_SEQUENCE_LENGTH = 20
VALIDATION_SPLIT = 0.2
word2vec_model = gensim.models.Word2Vec.load_word2vec_format(GLOVE_MODEL_FILE)
word_embed_size = word2vec_model['the'].shape[0]

# vocab generation
MyTokenizer = tokenize.casual.TweetTokenizer(strip_handles=True, reduce_len=True)
vocab, reverse_vocab = {}, {}
freq = defaultdict(int)
tweets = {}
#
#
#def get_embedding(word):
#    try:
#        return word2vec_model[word]
#    except Exception, e:
#        print 'Encoding not found: %s' %(word)
#        return np.zeros(EMBEDDING_DIM) 
#
#
#def get_embedding_weights():
#    embedding = []
#    embedding.append([0]*EMBEDDING_DIM)     # Create a NULL vector entry for the 1st index
#    for (w_index, word) in sorted(reverse_vocab.iteritems()):
#        embedding.append(get_embedding(word))
#    pdb.set_trace()
#    return np.array(embedding)
#

def select_tweets():
    # selects the tweets as in mean_glove_embedding method
    # Processing
    tweets = get_data()
    X, Y = [], []
    tweet_return = []
    for tweet in tweets:
        _emb = 0
        words = Tokenize(tweet['text']).split()
        for w in words:
            if w in word2vec_model:  # Check if embeeding there in GLove model
                _emb+=1
        if _emb:   # Not a blank tweet
            tweet_return.append(tweet)
    print 'Tweets selected:', len(tweet_return)
    #pdb.set_trace()
    return tweet_return


def gen_data():
    y_map = {
            'none': 0,
            'racism': 1,
            'sexism': 2
            }

    X, y = [], []
    for tweet in tweets:
        words = Tokenize(tweet['text']).split()
        words = [word for word in words if word not in STOPWORDS]
        emb = np.zeros(word_embed_size)
        for word in words:
            try:
                emb += word2vec_model[word]
            except:
                pass
        emb /= len(words)
        X.append(emb)
        y.append(y_map[tweet['label']])
    return X, y

    
def Tokenize(tweet):
    #return MyTokenizer.tokenize(tweet)
    #pdb.set_trace()
    return tokenizer_g(tweet)


def clasfication_model(X, y):
    X, Y = gen_data()
    #pdb.set_trace()
    NO_OF_FOLDS=10
    #logreg = linear_model.LogisticRegression()
    #logreg = RandomForestClassifier()
    logreg = GradientBoostingClassifier()
    X, Y = shuffle(X, Y, random_state=SEED)
    scores1 = cross_val_score(logreg, X, Y, cv=NO_OF_FOLDS, scoring='precision_weighted')
    predictions = cross_val_predict(logreg, X, Y, cv=NO_OF_FOLDS)
    print scores1
    print "Precision(avg): %0.3f (+/- %0.3f)" % (scores1.mean(), scores1.std() * 2)

    recall = make_scorer(recall_score, average='recall_weighted')
    logreg = linear_model.LogisticRegression()
    scores2 = cross_val_score(logreg, X, Y, cv=NO_OF_FOLDS, scoring='recall_weighted')
    print "Recall(avg): %0.3f (+/- %0.3f)" % (scores2.mean(), scores2.std() * 2)
    
    f1 = make_scorer(f1_score, average='f1_weighted')
    logreg = linear_model.LogisticRegression()
    scores3 = cross_val_score(logreg, X, Y, cv=NO_OF_FOLDS, scoring='f1_weighted')
    print "F1-score(avg): %0.3f (+/- %0.3f)" % (scores3.mean(), scores3.std() * 2)

    pdb.set_trace()



if __name__ == "__main__":

    Tweets = select_tweets()
    tweets = Tweets
    #filter_vocab(20000)
    X, y = gen_data()    
    clasfication_model(X, y)
    
    pdb.set_trace()


