from data_handler import get_data
import sys
import numpy as np
from preprocess_twitter import tokenize as tokenizer_glove
import pdb
from nltk import tokenize
from sklearn.metrics import make_scorer, f1_score, accuracy_score, recall_score, precision_score, classification_report, precision_recall_fscore_support
from sklearn.ensemble  import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
import pdb
from sklearn.metrics import make_scorer, f1_score, accuracy_score, recall_score, precision_score, classification_report, precision_recall_fscore_support
from sklearn.utils import shuffle
from sklearn.ensemble  import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from gensim.parsing.preprocessing import STOPWORDS
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
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
    texts.append(tweet['text'].lower())
    labels.append(label_map[tweet['label']])
print('Found %s texts. (samples)' % len(texts))


# logistic, gradient_boosting, random_forest, svm
MODEL_TYPE=sys.argv[1]
EMBEDDING_DIM = int(sys.argv[2])


## Load the orginal glove file
## SHASHANK files
#GLOVE_MODEL_FILE="/home/shashank/data/embeddings/GloVe/glove-twitter" + str(EMBEDDING_DIM)+ "-w2v"
## PINKESH files
GLOVE_MODEL_FILE="/home/pinkesh/DATASETS/glove-twitter/GENSIM.glove.twitter.27B." + str(EMBEDDING_DIM) + "d.txt"


SEED=42
MAX_NB_WORDS = None
VALIDATION_SPLIT = 0.2
word2vec_model = gensim.models.Word2Vec.load_word2vec_format(GLOVE_MODEL_FILE)
word_embed_size = word2vec_model['the'].shape[0]


# Tokenizer to use
#MyTokenizer = tokenize.casual.TweetTokenizer(strip_handles=True, reduce_len=True).tokenize
MyTokenizer = tokenizer_glove

# vocab generation
vocab, reverse_vocab = {}, {}
freq = defaultdict(int)
tweets = {}


def select_tweets_whose_embedding_exis_whose_embedding_exist():
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
    return MyTokenizer(tweet)

def get_model(m_type="logistic"):
    if m_type == 'logistic':
        logreg = LogisticRegression()
    elif m_type == "gradient_boosting":
        logreg = GradientBoostingClassifier()
    elif m_type == "random_forest":
        logreg = RandomForestClassifier()
    elif m_type == "svm":
        logreg = SVC(class_weight="balanced", kernel='rbf')
    else:
        print "ERROR: Please specify a correst model"
        return None

    return logreg


def clasfication_model(X, y, model_type="logistic_regression"):
    NO_OF_FOLDS=10
    X, Y = gen_data()
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

    tweets = select_tweets_whose_embedding_exist()
    #filter_vocab(20000)
    X, y = gen_data()    
    clasfication_model(X, y)

    # For TFIDF-SVC
    # We do not need to run the above code for TFIDF
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(texts)
    tfidf_transformer = TfidfTransformer(use_idf=True)
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    X = X_train_tfidf
    print 'Training'
    classification_model(X, labels)

    
    pdb.set_trace()


