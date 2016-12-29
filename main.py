from sklearn import linear_model
import numpy as np
from mean_glove_embeedings import mean_glove
from sklearn.model_selection import cross_val_score, cross_val_predict
import pdb
from sklearn.metrics import make_scorer, f1_score, accuracy_score, recall_score, precision_score, precision_recall_fscore_support
from sklearn.utils import shuffle
from sklearn.model_selection import KFold

def main():
    X, Y = mean_glove()
    NO_OF_FOLDS=10
    X, Y = shuffle(X, Y)
    #pdb.set_trace()
    precision = make_scorer(accuracy_score)
    logreg = linear_model.LogisticRegression()
    cv_object = KFold(10)
    p, r, f1 = 0., 0., 0.
    p1, r1, f11 = 0., 0., 0.
    for train_index, test_index in cv_object.split(X):
        X_train, y_train = X[train_index], Y[train_index]
        X_test, y_test = X[test_index], Y[test_index]
        logreg.fit(X_train, y_train)
        y_pred = logreg.predict(X_test)
        print y_pred
        p += precision_score(y_test, y_pred, average='macro')
        p1 += precision_score(y_test, y_pred, average='micro')
        r += recall_score(y_test, y_pred, average='macro')
        r1 += recall_score(y_test, y_pred, average='micro')
        f1 += f1_score(y_test, y_pred, average='macro')
        f11 += f1_score(y_test, y_pred, average='micro')

    print "macro results are"
    print "average precision is %f" %(p/10)
    print "average recall is %f" %(r/10)
    print "average f1 is %f" %(f1/10)

    print "micro results are"
    print "average precision is %f" %(p1/10)
    print "average recall is %f" %(r1/10)
    print "average f1 is %f" %(f11/10)


    # scores1 = cross_val_score(logreg, X, Y, cv=NO_OF_FOLDS, scoring=precision)
    # predictions = cross_val_predict(logreg, X, Y, cv=NO_OF_FOLDS)
    # print scores1
    # print "Precision(avg): %0.3f (+/- %0.3f)" % (scores1.mean(), scores1.std() * 2)

    # recall = make_scorer(recall_score, average='micro')
    # logreg = linear_model.LogisticRegression()
    # scores2 = cross_val_score(logreg, X, Y, cv=NO_OF_FOLDS, scoring=recall)
    # print "Recall(avg): %0.3f (+/- %0.3f)" % (scores2.mean(), scores2.std() * 2)
    
    # f1 = make_scorer(f1_score, average='micro')
    # logreg = linear_model.LogisticRegression()
    # scores3 = cross_val_score(logreg, X, Y, cv=NO_OF_FOLDS, scoring=f1)
    # print "F1-score(avg): %0.3f (+/- %0.3f)" % (scores3.mean(), scores3.std() * 2)

    pdb.set_trace()


if __name__=="__main__":
    main()
