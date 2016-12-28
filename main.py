from sklearn import linear_model
import numpy as np
from mean_glove_embeedings import mean_glove
from sklearn.cross_validation import cross_val_score
import pdb

def main():
    X, Y = mean_glove()

    logreg = linear_model.LogisticRegression(C=1e5)
    #logreg.fit(X, Y)
    
    scores = cross_val_score(logreg, X, Y, cv=10, scoring='precision')
    print "Precision: %0.2f " % (scores.mean())
    pdb.set_trace()


if __name__=="__main__":
    main()
