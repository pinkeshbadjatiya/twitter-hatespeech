# Hate Speech Detection on Twitter

Repository to train a Machine learning model to detect hate speech in a tweet. Contains code for different models.

## Dataset

Dataset can be downloaded from [https://github.com/zeerakw/hatespeech](https://github.com/zeerakw/hatespeech). Contains tweet id's and corresponding annotations. 

Tweets are labelled as either Racist, Sexist or Neither Racist or Sexist. 

Use your favourite tweet crawler and download the data and place the tweets in the folder 'tweet_data'.


## Requirements
* Keras 
* Tensorflow or Theano (we experimented with theano)
* Gensim
* xgboost
* NLTK
* Sklearn
* Numpy

## Instructions to run

To run a model ([MODEL]) for training, use the following instructions mentioned below. Use appropriate parameter settings to test the variations of the model.

- BagOfWords models - **BoWV.py**
```
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
```

- TF-IDF based models - **tfidf.py**
```
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
```

- LSTM(RNN) based methods - **lstm.py**
```
usage: lstm.py [-h] -f EMBEDDINGFILE -d DIMENSION --tokenizer {glove,nltk}
               --loss LOSS --optimizer OPTIMIZER --epochs EPOCHS --batch-size
               BATCH_SIZE [-s SEED] [--folds FOLDS] [--kernel KERNEL]
               [--class_weight CLASS_WEIGHT] --initialize-weights
               {random,glove} [--learn-embeddings] [--scale-loss-function]

LSTM based models for twitter Hate speech detection

optional arguments:
  -h, --help            show this help message and exit
  -f EMBEDDINGFILE, --embeddingfile EMBEDDINGFILE
  -d DIMENSION, --dimension DIMENSION
  --tokenizer {glove,nltk}
  --loss LOSS
  --optimizer OPTIMIZER
  --epochs EPOCHS
  --batch-size BATCH_SIZE
  -s SEED, --seed SEED
  --folds FOLDS
  --kernel KERNEL
  --class_weight CLASS_WEIGHT
  --initialize-weights {random,glove}
  --learn-embeddings
  --scale-loss-function
```

- CNN based models - **cnn.py**
```
usage: cnn.py [-h] -f EMBEDDINGFILE -d DIMENSION --tokenizer {glove,nltk}
              --loss LOSS --optimizer OPTIMIZER --epochs EPOCHS --batch-size
              BATCH_SIZE [-s SEED] [--folds FOLDS]
              [--class_weight CLASS_WEIGHT] --initialize-weights
              {random,glove} [--learn-embeddings] [--scale-loss-function]

CNN based models for twitter Hate speech detection

optional arguments:
  -h, --help            show this help message and exit
  -f EMBEDDINGFILE, --embeddingfile EMBEDDINGFILE
  -d DIMENSION, --dimension DIMENSION
  --tokenizer {glove,nltk}
  --loss LOSS
  --optimizer OPTIMIZER
  --epochs EPOCHS
  --batch-size BATCH_SIZE
  -s SEED, --seed SEED
  --folds FOLDS
  --class_weight CLASS_WEIGHT
  --initialize-weights {random,glove}
  --learn-embeddings
  --scale-loss-function
```



### Examples:
```
python BoWV.py --model logistic --seed 42 -f glove.twitter.27b.25d.txt -d 25 --seed 42 --folds 10
python tfidf.py -m tfidf_svm_linear --max_ngram 3 --tokenizer glove --loss squared_hinge
python lstm.py -f ~/DATASETS/glove-twitter/GENSIM.glove.twitter.27B.25d.txt -d 25 --tokenizer glove --loss categorical_crossentropy --optimizer adam --initialize-weights random --learn-embeddings --epochs 10 --batch-size 512
python cnn.py -f ~/DATASETS/glove-twitter/GENSIM.glove.twitter.27B.25d.txt -d 25 --tokenizer nltk --loss categorical_crossentropy --optimizer adam --epochs 10 --batch-size 128 --initialize-weights random --scale-loss-function

```

### TO-DOs

Some model parameters hard-coded, add a nice command line argument module


