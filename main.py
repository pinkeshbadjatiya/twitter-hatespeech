from sklearn import linear_model
import numpy as np
from mean_glove_embeedings import mean_glove
from sklearn.model_selection import cross_val_score
import pdb
from sklearn.metrics import make_scorer, f1_score, accuracy_score, recall_score

def main():
    X, Y = mean_glove()
    NO_OF_FOLDS=10

    #pdb.set_trace()
    precision = make_scorer(accuracy_score)
    logreg = linear_model.LogisticRegression(C=1e5)
    scores1 = cross_val_score(logreg, X, Y, cv=NO_OF_FOLDS, scoring=precision)
    print "Precision(avg): %0.3f (+/- %0.3f)" % (scores1.mean(), scores1.std() * 2)

    recall = make_scorer(recall_score, average='micro')
    logreg = linear_model.LogisticRegression(C=1e5)
    scores2 = cross_val_score(logreg, X, Y, cv=NO_OF_FOLDS, scoring=recall)
    print "Recall(avg): %0.3f (+/- %0.3f)" % (scores2.mean(), scores2.std() * 2)
    
    f1 = make_scorer(f1_score, average='micro')
    logreg = linear_model.LogisticRegression(C=1e5)
    scores3 = cross_val_score(logreg, X, Y, cv=NO_OF_FOLDS, scoring=f1)
    print "F1-score(avg): %0.3f (+/- %0.3f)" % (scores3.mean(), scores3.std() * 2)

    pdb.set_trace()


if __name__=="__main__":
    main()
