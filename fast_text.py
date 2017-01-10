from data_handler import get_data
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Input, LSTM
from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Input, Merge, Convolution1D, MaxPooling1D, GlobalMaxPooling1D, GlobalAveragePooling1D
import numpy as np
from preprocess_twitter import tokenize as tokenizer_g
import pdb
from nltk import tokenize
from sklearn.metrics import make_scorer, f1_score, accuracy_score, recall_score, precision_score, classification_report, precision_recall_fscore_support
from sklearn.ensemble  import GradientBoostingClassifier, RandomForestClassifier
from gensim.parsing.preprocessing import STOPWORDS
from sklearn.model_selection import KFold
from keras.utils import np_utils
import codecs
import operator
import gensim, sklearn
from collections import defaultdict
from batch_gen import batch_gen
from string import punctuation
from get_similar_words import get_similar_words
import sys

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

EMBEDDING_DIM = int(sys.argv[1])
np.random.seed(42)
# Load the orginal glove file
# SHASHANK files
#GLOVE_MODEL_FILE="/home/shashank/DL_NLP/glove-twitter" + str(EMBEDDING_DIM) + "-w2v"


# PINKESH files
GLOVE_MODEL_FILE="/home/pinkesh/DATASETS/glove-twitter/GENSIM.glove.twitter.27B." + str(EMBEDDING_DIM) + "d.txt"
NO_OF_CLASSES=3

MAX_NB_WORDS = None
VALIDATION_SPLIT = 0.2
word2vec_model = gensim.models.Word2Vec.load_word2vec_format(GLOVE_MODEL_FILE)


# vocab generation
MyTokenizer = tokenize.casual.TweetTokenizer(strip_handles=True, reduce_len=True)
vocab, reverse_vocab = {}, {}
freq = defaultdict(int)
tweets = {}


def get_embedding(word):
    #return
    try:
        return word2vec_model[word]
    except Exception, e:
        print 'Encoding not found: %s' %(word)
        return np.zeros(EMBEDDING_DIM)

def get_embedding_weights():
    embedding = np.zeros((len(vocab) + 1, EMBEDDING_DIM))
    n = 0
    for k, v in vocab.iteritems():
    	try:
    		embedding[v] = word2vec_model[k]
    	except:
    		n += 1
    		pass
    print "%d embedding missed"%n
    #pdb.set_trace()
    return embedding


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


def gen_vocab():
    # Processing
    vocab_index = 1
    for tweet in tweets:
        text = Tokenize(tweet['text'])
        text = ''.join([c for c in text if c not in punctuation])
        words = text.split()
        words = [word for word in words if word not in STOPWORDS]

        for word in words:
            if word not in vocab:
                vocab[word] = vocab_index
                reverse_vocab[vocab_index] = word       # generate reverse vocab as well
                vocab_index += 1
            freq[word] += 1
    vocab['UNK'] = len(vocab) + 1
    reverse_vocab[len(vocab)] = 'UNK'
    #pdb.set_trace()


def filter_vocab(k):
    global freq, vocab
    #pdb.set_trace()
    freq_sorted = sorted(freq.items(), key=operator.itemgetter(1))
    tokens = freq_sorted[:k]
    vocab = dict(zip(tokens, range(1, len(tokens) + 1)))
    vocab['UNK'] = len(vocab) + 1


def gen_sequence():
    y_map = {
            'none': 0,
            'racism': 1,
            'sexism': 2
            }

    X, y = [], []
    for tweet in tweets:
        text = Tokenize(tweet['text'])
        text = ''.join([c for c in text if c not in punctuation])
        words = text.split()
        words = [word for word in words if word not in STOPWORDS]
        seq, _emb = [], []
        for word in words:
            seq.append(vocab.get(word, vocab['UNK']))
        X.append(seq)
        y.append(y_map[tweet['label']])
    return X, y


def Tokenize(tweet):
    #return MyTokenizer.tokenize(tweet)
    #pdb.set_trace()
    return tokenizer_g(tweet)


def shuffle_weights(model):
    weights = model.get_weights()
    weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
    model.set_weights(weights)


def fast_text_model(sequence_length):
    model = Sequential()
    model.add(Embedding(len(vocab)+1, EMBEDDING_DIM, input_length=sequence_length))
    #model.add(Embedding(len(vocab)+1, EMBEDDING_DIM, input_length=sequence_length, trainable=False))
    model.add(Dropout(0.5))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    print model.summary()
    return model

def train_fasttext(X, y, model, inp_dim,embedding_weights, epochs=10, batch_size=128):
    cv_object = KFold(n_splits=10, shuffle=True, random_state=42)
    print cv_object
    p, r, f1 = 0., 0., 0.
    p1, r1, f11 = 0., 0., 0.
    sentence_len = X.shape[1]
    lookup_table = np.zeros_like(model.layers[0].get_weights()[0])
    for train_index, test_index in cv_object.split(X):
        shuffle_weights(model)
        #pdb.set_trace()
        #model.layers[0].set_weights([embedding_weights])
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]
        y_train = y_train.reshape((len(y_train), 1))
        X_temp = np.hstack((X_train, y_train))
        for epoch in xrange(epochs):
            for X_batch in batch_gen(X_temp, batch_size):
                x = X_batch[:, :sentence_len]
                y_temp = X_batch[:, sentence_len]
		class_weights = {}
		class_weights[0] = np.where(y_temp == 0)[0].shape[0]/float(len(y_temp))
		class_weights[1] = np.where(y_temp == 1)[0].shape[0]/float(len(y_temp))
		class_weights[2] = np.where(y_temp == 2)[0].shape[0]/float(len(y_temp))
                try:
                    y_temp = np_utils.to_categorical(y_temp, nb_classes=3)
                except Exception as e:
                    print e
                #print x.shape, y.shape
                loss, acc = model.train_on_batch(x, y_temp)#, class_weight=class_weights)
                print loss, acc
        #pdb.set_trace()
        lookup_table += model.layers[0].get_weights()[0]
        y_pred = model.predict_on_batch(X_test)
        y_pred = np.argmax(y_pred, axis=1)
        print classification_report(y_test, y_pred)
        print precision_recall_fscore_support(y_test, y_pred)
        print y_pred
        p += precision_score(y_test, y_pred, average='weighted')
        p1 += precision_score(y_test, y_pred, average='micro')
        r += recall_score(y_test, y_pred, average='weighted')
        r1 += recall_score(y_test, y_pred, average='micro')
        f1 += f1_score(y_test, y_pred, average='weighted')
        f11 += f1_score(y_test, y_pred, average='micro')

    print "macro results are"
    print "average precision is %f" %(p/10)
    print "average recall is %f" %(r/10)
    print "average f1 is %f" %(f1/10)

    print "micro results are"
    print "average precision is %f" %(p1/10)
    print "average recall is %f" %(r1/10)
    print "average f1 is %f" %(f11/10)
    return lookup_table/float(10)


def check_semantic_sim(embedding_table, word):
    reverse_vocab = {v:k for k,v in vocab.iteritems()}
    sim_word_idx = get_similar_words(embedding_table, embedding_table[vocab[word]], 25)
    sim_words = map(lambda x:reverse_vocab[x[1]], sim_word_idx)
    print sim_words

def tryWord(embedding_table):
    while True:
        print "enter word"
        word = raw_input()
        if word == "pdb":
            pdb.set_trace()
        elif word == 'exit':
            return
        else:
            check_semantic_sim(embedding_table, word)


if __name__ == "__main__":

    Tweets = select_tweets()
    tweets = Tweets
    gen_vocab()
    X, y = gen_sequence()
    MAX_SEQUENCE_LENGTH = max(map(lambda x:len(x), X))
    print "max seq length is %d"%(MAX_SEQUENCE_LENGTH)
    data = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
    y = np.array(y)
    W = get_embedding_weights()
    data, y = sklearn.utils.shuffle(data, y)
    model = fast_text_model(data.shape[1])
    _ = train_fasttext(data, y, model, EMBEDDING_DIM, W)
    table = model.layers[0].get_weights()[0]
    #check_semantic_sim(table)
    tryWord(table)
    pdb.set_trace()


