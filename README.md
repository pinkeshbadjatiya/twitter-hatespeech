## Hate Speech Detection on Twitter

Repository to train a Machine learning model to detect hate speech in a tweet. Contains code for different models.

### Dataset

Dataset can be downloaded from [https://github.com/zeerakw/hatespeech](https://github.com/zeerakw/hatespeech). Contains tweet id's and corresponding annotations. 

Tweets are labelled as either Racist, Sexist or Neither Racist or Sexist. 

Use your favourite tweet crawler and download the data and place the tweets in the folder 'tweet_data'.


### Requirements
* Keras 
* Tensorflow or Theano (we experimented with theano)
* Gensim
* xgboost

### Instructions to run

To run a model ([MODEL]) for training, use 


For BoWV.py
usage: BoWV.py [-h] -m
               {logistic,gradient_boosting,random_forest,svm,svm_linear} -f
               EMBEDDINGFILE -d DIMENSION --tokenizer {glove,nltk} [-s SEED]
               [--folds FOLDS] [--estimators ESTIMATORS] [--loss LOSS]
               [--kernel KERNEL] [--class_weight CLASS_WEIGHT]

BagOfWords model for twitter Hate speech detection

optional arguments:
  -h, --help            show this help message and exit
  -m {logistic,gradient_boosting,random_forest,svm,svm_linear}, --model {logistic,gradient_boosting,random_forest,svm,svm_linear}
  -f EMBEDDINGFILE, --embeddingfile EMBEDDINGFILE
  -d DIMENSION, --dimension DIMENSION
  --tokenizer {glove,nltk}
  -s SEED, --seed SEED
  --folds FOLDS
  --estimators ESTIMATORS
  --loss LOSS
  --kernel KERNEL
  --class_weight CLASS_WEIGHT


For tfidf.py

usage: tfidf.py [-h] -m
                {tfidf_svm,tfidf_svm_linear,tfidf_logistic,tfidf_gradient_boosting,tfidf_random_forest}
                --max_ngram MAX_NGRAM --tokenizer {glove,nltk} [-s SEED]
                [--folds FOLDS] [--estimators ESTIMATORS] [--loss LOSS]
                [--kernel KERNEL] [--class_weight CLASS_WEIGHT]
                [--use-inverse-doc-freq]

TF-IDF model for twitter Hate speech detection

optional arguments:
  -h, --help            show this help message and exit
  -m {tfidf_svm,tfidf_svm_linear,tfidf_logistic,tfidf_gradient_boosting,tfidf_random_forest}, --model {tfidf_svm,tfidf_svm_linear,tfidf_logistic,tfidf_gradient_boosting,tfidf_random_forest}
  --max_ngram MAX_NGRAM
  --tokenizer {glove,nltk}
  -s SEED, --seed SEED
  --folds FOLDS
  --estimators ESTIMATORS
  --loss LOSS
  --kernel KERNEL
  --class_weight CLASS_WEIGHT
  --use-inverse-doc-freq


For lstm.py
usage: lstm.py [-h] -f EMBEDDINGFILE -d DIMENSION --tokenizer {glove,nltk}
               --loss LOSS --optimizer OPTIMIZER [-s SEED] [--folds FOLDS]
               [--kernel KERNEL] [--class_weight CLASS_WEIGHT]
               --initialize-weights {random,glove} [--learn-embeddings]

LSTM based models for twitter Hate speech detection

optional arguments:
  -h, --help            show this help message and exit
  -f EMBEDDINGFILE, --embeddingfile EMBEDDINGFILE
  -d DIMENSION, --dimension DIMENSION
  --tokenizer {glove,nltk}
  --loss LOSS
  --optimizer OPTIMIZER
  -s SEED, --seed SEED
  --folds FOLDS
  --kernel KERNEL
  --class_weight CLASS_WEIGHT
  --initialize-weights {random,glove}
  --learn-embeddings




'python [MODEL] [WORD_EMBED_DIM']

### TO-DOs

Some model parameters hard-coded, add a nice command line argument module


