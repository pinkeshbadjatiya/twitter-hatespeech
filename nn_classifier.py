from data_handler import get_data
import sys
import numpy as np
import pdb, json
from sklearn.metrics import make_scorer, f1_score, accuracy_score, recall_score, precision_score, classification_report, precision_recall_fscore_support
from sklearn.ensemble  import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.feature_extraction.text import TfidfVectorizer
import pdb
from sklearn.metrics import make_scorer, f1_score, accuracy_score, recall_score, precision_score, classification_report, precision_recall_fscore_support
from sklearn.utils import shuffle
from sklearn.ensemble  import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
import codecs
import operator
import gensim, sklearn
from collections import defaultdict
from batch_gen import batch_gen
from my_tokenizer import glove_tokenize
import xgboost as xgb

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
    texts.append(tweet['text'].lower())
    labels.append(label_map[tweet['label']])
print('Found %s texts. (samples)' % len(texts))


# logistic, gradient_boosting, random_forest, svm, tfidf_svm_linear, tfidf_svm_rbf
model_count = 2
word_embed_size = 200
GLOVE_MODEL_FILE = str(sys.argv[1])
EMBEDDING_DIM = int(sys.argv[2])
MODEL_TYPE=sys.argv[3]
print 'Embedding Dimension: %d' %(EMBEDDING_DIM)
print 'GloVe Embedding: %s' %(GLOVE_MODEL_FILE)

word2vec_model1 = np.load('fast_text.npy')
word2vec_model1 = word2vec_model1.reshape((word2vec_model1.shape[1], word2vec_model1.shape[2]))
f_vocab = open('vocab_fast_text', 'r')
vocab = json.load(f_vocab)
word2vec_model = {}
for k,v in vocab.iteritems():
word2vec_model[k] = word2vec_model1[int(v)]
del word2vec_model1


SEED=42
MAX_NB_WORDS = None
VALIDATION_SPLIT = 0.2


# vocab generation
vocab, reverse_vocab = {}, {}
freq = defaultdict(int)
tweets = {}


def select_tweets_whose_embedding_exists():
    # selects the tweets as in mean_glove_embedding method
    # Processing
    tweets = get_data()
    X, Y = [], []
    tweet_return = []
    for tweet in tweets:
        _emb = 0
        words = glove_tokenize(tweet['text'])
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
        words = glove_tokenize(tweet['text'])
        emb = np.zeros(word_embed_size)
        for word in words:
            try:
                emb += word2vec_model[word]
            except:
                pass
        emb /= len(words)
        X.append(emb)
        y.append(y_map[tweet['label']])
    X = np.array(X)
    y = np.array(y)
    return X, y

    
def get_model(m_type=None):
    if not m_type:
        print 'ERROR: Please provide a valid method name'
        return None

    if m_type == 'logistic':
        logreg = LogisticRegression()
    elif m_type == "gradient_boosting":
        #logreg = GradientBoostingClassifier(n_estimators=10)
        logreg = xgb.XGBClassifier(nthread=-1)
    elif m_type == "random_forest":
        logreg = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    elif m_type == "svm_rbf":
        logreg = SVC(class_weight="balanced", kernel='rbf')
    elif m_type == "svm_linear":
        logreg = LinearSVC(class_weight="balanced")
    else:
        print "ERROR: Please specify a correst model"
        return None

    return logreg


def classification_model(X, Y, model_type="logistic"):
    NO_OF_FOLDS=10
    X, Y = shuffle(X, Y, random_state=SEED)
    print "Model Type:", model_type

    #predictions = cross_val_predict(logreg, X, Y, cv=NO_OF_FOLDS)
    scores1 = cross_val_score(get_model(model_type), X, Y, cv=NO_OF_FOLDS, scoring='precision_weighted')
    print "Precision(avg): %0.3f (+/- %0.3f)" % (scores1.mean(), scores1.std() * 2)

    scores2 = cross_val_score(get_model(model_type), X, Y, cv=NO_OF_FOLDS, scoring='recall_weighted')
    print "Recall(avg): %0.3f (+/- %0.3f)" % (scores2.mean(), scores2.std() * 2)
    
    scores3 = cross_val_score(get_model(model_type), X, Y, cv=NO_OF_FOLDS, scoring='f1_weighted')
    print "F1-score(avg): %0.3f (+/- %0.3f)" % (scores3.mean(), scores3.std() * 2)

    pdb.set_trace()



if __name__ == "__main__":

    #filter_vocab(20000)

    tweets = select_tweets_whose_embedding_exists()
    X, Y = gen_data()

    classification_model(X, Y, MODEL_TYPE)
    pdb.set_trace()


