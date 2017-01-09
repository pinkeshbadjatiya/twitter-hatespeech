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

To run a model (<MODEL>) for training, use 

'python <MODEL> <WORD_EMBED_DIM>'

### TO-DOs

Some model parameters hard-coded, add a nice command line argument module


