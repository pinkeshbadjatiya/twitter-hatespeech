from data_handler import get_data
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Input
from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Input, Merge, Convolution1D, MaxPooling1D
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

# Load the orginal glove file
# SHASHANK files
#GLOVE_MODEL_FILE="/home/shashank/data/embeddings/GloVe/glove-twitter25-w2v"


# PINKESH files
GLOVE_MODEL_FILE="/home/pinkesh/DATASETS/glove-twitter/GENSIM.glove.twitter.27B." + str(EMBEDDING_DIM) + "d.txt"


MAX_NB_WORDS = None
VALIDATION_SPLIT = 0.2
word2vec_model = gensim.models.Word2Vec.load_word2vec_format(GLOVE_MODEL_FILE)


# vocab generation
MyTokenizer = tokenize.casual.TweetTokenizer(strip_handles=True, reduce_len=True)
vocab, reverse_vocab = {}, {}
freq = defaultdict(int)
tweets = {}


def get_embedding(word):
    try:
        return word2vec_model[word]
    except Exception, e:
        print 'Encoding not found: %s' %(word)
        return np.zeros(EMBEDDING_DIM) 

def get_embedding_weights():
    embedding = []
    embedding.append([0]*EMBEDDING_DIM)     # Create a NULL vector entry for the 1st index
    for (w_index, word) in sorted(reverse_vocab.iteritems()):
        embedding.append(get_embedding(word))
    #pdb.set_trace()
    return np.array(embedding)


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
        words = Tokenize(tweet['text']).split()
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
    pdb.set_trace()
    freq_sorted = sorted(freq.items(), key=operator.itemgetter(1))
    tokens = freq_sorted[:k]
    vocab = dict(zip(tokens, range(1, len(tokens) + 1)))
    vocab['UNK'] = len(vocab) + 1


#def gen_embeeding_matrix():
#    emb_matrix = []
#    for k, v in vocab.iteritems():
#        emb_matrix.append(

def gen_sequence():
    y_map = {
            'none': 0,
            'racism': 1,
            'sexism': 2
            }

    X, y = [], []
    for tweet in tweets:
        words = Tokenize(tweet['text']).split()
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


def cnn_model(sequence_length, embedding_dim, embedding_weights):
    model_variation = 'CNN-rand'  #  CNN-rand | CNN-non-static | CNN-static
    print('Model variation is %s' % model_variation)

    # Model Hyperparameters
    n_classes = 3
    embedding_dim = EMBEDDING_DIM
    filter_sizes = (3, 4)
    num_filters = 150
    dropout_prob = (0.25, 0.5)
    hidden_dims = 150

    # Training parameters
    batch_size = 32
    num_epochs = 100
    val_split = 0.1

    # Word2Vec parameters, see train_word2vec
    min_word_count = 1  # Minimum word count                        
    context = 10        # Context window size    

    graph_in = Input(shape=(sequence_length, embedding_dim))
    convs = []
    for fsz in filter_sizes:
        conv = Convolution1D(nb_filter=num_filters,
                             filter_length=fsz,
                             border_mode='valid',
                             activation='relu',
                             subsample_length=1)(graph_in)
        pool = MaxPooling1D(pool_length=2)(conv)
        flatten = Flatten()(pool)
        convs.append(flatten)
        
    if len(filter_sizes)>1:
        out = Merge(mode='concat')(convs)
    else:
        out = convs[0]

    graph = Model(input=graph_in, output=out)

    # main sequential model
    model = Sequential()
    #if not model_variation=='CNN-rand':
    model.add(Embedding(len(vocab)+1, embedding_dim, input_length=sequence_length,
                                weights=[embedding_weights], trainable=False))
    #model.add(Embedding(len(vocab), embedding_dim, input_length=sequence_length))
    model.add(Dropout(dropout_prob[0], input_shape=(sequence_length, embedding_dim)))
    model.add(graph)
    model.add(Dense(hidden_dims))
    model.add(Dropout(dropout_prob[1]))
    model.add(Activation('relu'))
    model.add(Dense(n_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

def train_CNN(X, y, model, inp_dim, epochs=10, batch_size=128):
    cv_object = KFold(n_splits=10, shuffle=True, random_state=42)
    print cv_object
    p, r, f1 = 0., 0., 0.
    p1, r1, f11 = 0., 0., 0.
    sentence_len = X.shape[1]
    for train_index, test_index in cv_object.split(X):
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]
        #pdb.set_trace()
        y_train = y_train.reshape((len(y_train), 1))
        X_temp = np.hstack((X_train, y_train))
        for epoch in xrange(epochs):
            for X_batch in batch_gen(X_temp, batch_size):
                x = X_batch[:, :sentence_len]
                y_temp = X_batch[:, sentence_len]
                try:
                    y_temp = np_utils.to_categorical(y_temp, nb_classes=3)
                except Exception as e:
                    print e
                    print y_temp
                print x.shape, y.shape
                loss, acc = model.train_on_batch(x, y_temp)
                print loss, acc
        #clf.fit(X_train, y_train)
        y_pred = model.predict_on_batch(X_test)
        y_pred = np.argmax(y_pred, axis=1)
        #print y_pred
        #pdb.set_trace()
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


if __name__ == "__main__":

    Tweets = select_tweets()
    tweets = Tweets
    gen_vocab()
    #filter_vocab(20000)
    X, y = gen_sequence()    
    #Y = y.reshape((len(y), 1))
    MAX_SEQUENCE_LENGTH = max(map(lambda x:len(x), X))
    data = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
    y = np.array(y)
    data, y = sklearn.utils.shuffle(data, y)
    model = cnn_model(data.shape[1], 25, get_embedding_weights())
    train_CNN(data, y, model, 25)
    
    pdb.set_trace()


