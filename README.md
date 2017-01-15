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
usage: BoWV.py [-h] -m MODEL -f EMBEDDINGFILE -d DIMENSION [-s SEED]
               [--folds FOLDS] [--estimators ESTIMATORS] [--loss LOSS]
               [--kernel KERNEL] [--class_weight CLASS_WEIGHT]

BagOfWords model for twitter Hate speech detection

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
  -f EMBEDDINGFILE, --embeddingfile EMBEDDINGFILE
  -d DIMENSION, --dimension DIMENSION
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


'python [MODEL] [WORD_EMBED_DIM']

### TO-DOs

Some model parameters hard-coded, add a nice command line argument module


