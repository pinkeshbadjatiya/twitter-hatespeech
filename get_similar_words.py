from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pdb

def get_similar_words(X, vec, K=1):
    # X: (n_samples, n_features)
    # vec: (1, n_features)
    # returns: K top most similar words with score values and their indexes
    scores = cosine_similarity(X, vec)
    scores = sorted([(val, index) for index, val in enumerate(scores.reshape((1,scores.shape[0]))[0])], reverse=True)    
    scores = scores[1:K]
    return scores

