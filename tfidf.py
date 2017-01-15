from data_handler import get_data
import argparse
import sys
import numpy as np
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
from nltk.tokenize import TweetTokenizer


### Preparing the text data
texts = []  # list of text samples
labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids

# vocab generation
vocab, reverse_vocab = {}, {}
freq = defaultdict(int)
tweets = {}


# tfidf_logistic, tfidf_gradient_boosting, tfidf_random_forest, tfidf_svm_linear, tfidf_svm_rbf
MODEL_TYPE=None
MAX_NGRAM_LENGTH=None
NO_OF_FOLDS=10
CLASS_WEIGHT = None
N_ESTIMATORS = None
LOSS_FUN = None
KERNEL = None
MAX_NGRAM_LENGTH = None
SEED=42
TOKENIZER=None


def gen_data():
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



def get_model(m_type=None):
    if not m_type:
        print 'Please specify a model type'
        return None

    if m_type == "tfidf_svm":
        logreg = SVC(class_weight=CLASS_WEIGHT, kernel=KERNEL)
    elif m_type == "tfidf_svm_linear":
        logreg = LinearSVC(C=0.01, loss=LOSS_FUN, class_weight=CLASS_WEIGHT)
    elif m_type == 'tfidf_logistic':
        logreg = LogisticRegression()
    elif m_type == "tfidf_gradient_boosting":
        logreg = GradientBoostingClassifier(loss=LOSS_FUN, n_estimators=N_ESTIMATORS)
    elif m_type == "tfidf_random_forest":
        logreg = RandomForestClassifier(class_weight=CLASS_WEIGHT, n_estimators=N_ESTIMATORS)
        print "ERROR: Please specify a correct model"
        return None

    return logreg


def classification_model(X, Y, model_type=None):
    X, Y = shuffle(X, Y, random_state=SEED)
    print "Model Type:", model_type

    #predictions = cross_val_predict(logreg, X, Y, cv=NO_OF_FOLDS)
    scores1 = cross_val_score(get_model(model_type), X, Y, cv=NO_OF_FOLDS, scoring='precision_weighted')
    print "Precision(avg): %0.3f (+/- %0.3f)" % (scores1.mean(), scores1.std() * 2)

    scores2 = cross_val_score(get_model(model_type), X, Y, cv=NO_OF_FOLDS, scoring='recall_weighted')
    print "Recall(avg): %0.3f (+/- %0.3f)" % (scores2.mean(), scores2.std() * 2)
    
    scores3 = cross_val_score(get_model(model_type), X, Y, cv=NO_OF_FOLDS, scoring='f1_weighted')
    print "F1-score(avg): %0.3f (+/- %0.3f)" % (scores3.mean(), scores3.std() * 2)



if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='TF-IDF model for twitter Hate speech detection')
    parser.add_argument('-m', '--model', choices=['tfidf_svm', 'tfidf_svm_linear', 'tfidf_logistic', 'tfidf_gradient_boosting', 'tfidf_random_forest'], required=True)
    parser.add_argument('--max_ngram', required=True)
    parser.add_argument('--tokenizer', choices=['glove', 'nltk'], required=True)
    parser.add_argument('-s', '--seed', default=SEED)
    parser.add_argument('--folds', default=NO_OF_FOLDS)
    parser.add_argument('--estimators', default=N_ESTIMATORS)
    parser.add_argument('--loss', default=LOSS_FUN)
    parser.add_argument('--kernel', default=KERNEL)
    parser.add_argument('--class_weight')
    parser.add_argument('--use-inverse-doc-freq', action='store_true')

    args = parser.parse_args()

    MODEL_TYPE = args.model
    SEED = int(args.seed)
    NO_OF_FOLDS = int(args.folds)
    CLASS_WEIGHT = args.class_weight
    N_ESTIMATORS = int(args.estimators) if args.estimators else args.estimators
    LOSS_FUN = args.loss
    KERNEL = args.kernel
    MAX_NGRAM_LENGTH = int(args.max_ngram)
    USE_IDF = args.use_inverse_doc_freq

    if args.tokenizer == "glove":
        TOKENIZER = glove_tokenize
    elif args.tokenizer == "nltk":
        TOKENIZER = TweetTokenizer().tokenize

    print 'Max-ngram-length: %d' %(MAX_NGRAM_LENGTH)
    #filter_vocab(20000)

    # For TFIDF-SVC or any other varient
    # We do not need to run the above code for TFIDF
    # It does not use the filtered data using gen_data()
    gen_data()
    tfidf_transformer = TfidfVectorizer(use_idf=USE_IDF, analyzer="word", tokenizer=TOKENIZER, ngram_range=(1, MAX_NGRAM_LENGTH))
    #tfidf_transformer = TfidfVectorizer(use_idf=True, ngram_range=(1, MAX_NGRAM_LENGTH))
    X_train_tfidf = tfidf_transformer.fit_transform(texts)
    X = X_train_tfidf
    Y = labels

    classification_model(X, Y, MODEL_TYPE)
