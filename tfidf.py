from data_handler import get_data
import sys
import numpy as np
import pdb
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


# tfidf_logistic, tfidf_gradient_boosting, tfidf_random_forest, tfidf_svm_linear, tfidf_svm_rbf
MODEL_TYPE=sys.argv[1]
MAX_NGRAM_LENGTH=int(sys.argv[2])
print 'Max-ngram-length: %d' %(MAX_NGRAM_LENGTH)


SEED=42
MAX_NB_WORDS = None
VALIDATION_SPLIT = 0.2


# vocab generation
vocab, reverse_vocab = {}, {}
freq = defaultdict(int)
tweets = {}


#def select_tweets_whose_embedding_exists():
#    # selects the tweets as in mean_glove_embedding method
#    # Processing
#    tweets = get_data()
#    X, Y = [], []
#    tweet_return = []
#    for tweet in tweets:
#        _emb = 0
#        words = glove_tokenize(tweet['text'])
#        for w in words:
#            if w in word2vec_model:  # Check if embeeding there in GLove model
#                _emb+=1
#        if _emb:   # Not a blank tweet
#            tweet_return.append(tweet)
#    print 'Tweets selected:', len(tweet_return)
#    #pdb.set_trace()
#    return tweet_return
#
#
#def gen_data():
#    y_map = {
#            'none': 0,
#            'racism': 1,
#            'sexism': 2
#            }
#
#    X, y = [], []
#    for tweet in tweets:
#        words = glove_tokenize(tweet['text'])
#        emb = np.zeros(word_embed_size)
#        for word in words:
#            try:
#                emb += word2vec_model[word]
#            except:
#                pass
#        emb /= len(words)
#        X.append(emb)
#        y.append(y_map[tweet['label']])
#    return X, y

    
def get_model(m_type=None):
    if not m_type:
        print 'Please specify a model type'
        return None

    if m_type == "tfidf_svm_rbf":
        #logreg = SVC(class_weight="balanced", kernel='rbf')
        logreg = SVC(kernel='rbf')
    elif m_type == "tfidf_svm_linear":
        logreg = LinearSVC(C=0.01, loss='hinge', class_weight="balanced")
    elif m_type == 'tfidf_logistic':
        logreg = LogisticRegression()
    elif m_type == "tfidf_gradient_boosting":
        logreg = GradientBoostingClassifier()
    elif m_type == "tfidf_random_forest":
        logreg = RandomForestClassifier()
        print "ERROR: Please specify a correct model"
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

    # For TFIDF-SVC or any other varient
    # We do not need to run the above code for TFIDF
    # It does not use the filtered data using gen_data()
    tfidf_transformer = TfidfVectorizer(use_idf=True, analyzer="word", tokenizer=glove_tokenize, ngram_range=(1, MAX_NGRAM_LENGTH))
    #tfidf_transformer = TfidfVectorizer(use_idf=True, ngram_range=(1, MAX_NGRAM_LENGTH))
    X_train_tfidf = tfidf_transformer.fit_transform(texts)
    X = X_train_tfidf
    Y = labels

    classification_model(X, Y, MODEL_TYPE)
    pdb.set_trace()


