# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from sklearn import svm, linear_model
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from skll.metrics import kappa
import DataTasks as dt
import MovieTasks as mt
from scipy import stats


class SVM:

    class_type = "Keywords"
    class_names = None
    vector_path = None
    #class_names = ["absurd", "abstract", "abrupt"]
    #vector_path = "newdata\spaces\All-Layer-50-L.mds"
    class_by_class = True
    input_size = 200
    training_data = 10000
    amount_of_scores = 5

    def __init__(self, class_type="Keywords", class_names=None, vector_path=None, class_by_class=True, input_size=200,
                 training_data=10000, amount_of_scores=5, rankSVM=True):

        movie_names, movie_vectors, movie_labels = mt.getMovieData(class_type=class_type, input_size=input_size, class_names=class_names,
                                                                   class_by_class=class_by_class, vector_path=vector_path)

        file_names = dt.getAllFileNames("filmdata\classes" + class_type)

        n_train, x_train, y_train, n_test, x_test, y_test = self.getSampledData(movie_names, movie_vectors, movie_labels, training_data)
        
        top_keywords, overall_kappa, overall_accuracy, overall_f1 = self.runAllSVMs(len(y_train), amount_of_scores, y_test, y_train, x_train, x_test, class_type, input_size, file_names, rankSVM)
        row = [overall_kappa, overall_accuracy, overall_f1]

        dt.write1dArray(top_keywords, "filmdata/top_"+str(amount_of_scores)+"_" +class_type+"_" + str(input_size) + ".txt")
        dt.writeToSheet(row, "filmdata/experiments/experimenttest.csv")

        print "Kappa:", overall_kappa, "Accuracy:", overall_accuracy, "F1", overall_f1

    def getSampledData(self, movie_names, movie_vectors, movie_labels, training_data):

        n_train = movie_names[:training_data]
        x_train = []
        for x in range(len(x_train)):
            x_train.append(movie_vectors[:training_data])

        n_test = movie_names[training_data:]
        x_test = movie_vectors[training_data:]

        y_train = []
        y_test = []
        for ml in movie_labels:
            y_train.append(ml[:training_data])
            y_test.append(ml[training_data:])

        for yt in range(len(y_train)):
            y1 = 0
            y0 = 0
            for y in range(len(y_train[yt])):
                if y_train[yt][y] == 1:
                    y1 = y1 + 1
                if y_train[yt][y] == 0:
                    y0 = y0 + 1

            deleted_count= 0
            if y0 > (y1*2):
                for y in range(len(y_train[yt])):
                    try:
                        if y_train[yt][y] == 0:
                            if (y0 - deleted_count) > y1*2:
                                del x_train[yt][y]
                                del y_train[yt][y]
                                deleted_count = deleted_count + 1
                    except:
                        break

        return n_train, x_train, y_train, n_test, x_test, y_test


    """
    We have used the LIBSVM 27 implementation. Because default values of the parameters yielded very poor results,
    we have used a grid search procedure to find the optimal value of the C parameter for every class. To this end, the
    training data for each class was split into 2/3 training and 1/3 validation. Moreover, to address class imbalance we
    under-sampled negative training examples, such that the ratio between positive and negative training examples
    was at least 1/2.
    """

    def runRankSVM(self, y_test, y_train, x_train, x_test, class_type, input_size, file_names, keyword):
        clf = svm.LinearSVC()
        print len(x_train), len(y_train)
        clf.fit(x_train[keyword], y_train[keyword])
        #clf.decision_function(x_test)
        direction = clf.coef_.ravel() / linalg.norm(clf.coef_)

        tau_score = stats.kendalltau(linear_model.ridge.predict(X_test[b_test == i]), y_test[b_test == i])

        tau_score = stats.kendalltau(np.dot(x_test, direction), y_test)

        return tau_score, direction

    def runSVM(self, y_test, y_train, x_train, x_test, class_type, input_size, file_names, keyword):
        clf = svm.SVC(kernel='linear', C=.1)
        clf.fit(x_train[keyword], y_train[keyword])
        #clf.decision_function(x_test)
        direction = clf.coef_
        y_pred = clf.predict(x_test)
        y_pred = y_pred.tolist()
        kappa_score = kappa(y_test[keyword], y_pred)
        accuracy = accuracy_score(y_test[keyword], y_pred)
        f1 = f1_score(y_test[keyword], y_pred, average='macro')
        return kappa_score, accuracy, f1, direction

    def runAllSVMs(self, keyword_amount, amount_of_scores, y_test, y_train, x_train, x_test, class_type, input_size, file_names, rankSVM):
        totals = np.zeros(3)
        kappa_scores = []
        accuracy_scores = []
        f1_scores = []

        directions = []
        for x in range(keyword_amount-1):
            if rankSVM:
                kappa, accuracy, f1, direction = self.runSVM(y_test, y_train, x_train, x_test, class_type, input_size, file_names, x)
            else:
                kappa, accuracy, f1, direction = self.runSVM(y_test, y_train, x_train, x_test, class_type, input_size, file_names, x)
                kappa_scores.append(kappa)
                accuracy_scores.append(accuracy)
                f1_scores.append(f1)
                directions.append(direction)

        for val in kappa_scores:
            totals[0] += val

        for val in accuracy_scores:
            totals[1] += val

        for val in f1_scores:
            totals[2] += val

        kappa_scores = np.asarray(kappa_scores)
        top_scores = np.argpartition(kappa_scores, -amount_of_scores)[-amount_of_scores:]
        top_keywords = []
        for i in top_scores:
            top_keywords.append(file_names[i])

        overall_kappa = totals[0] / keyword_amount
        overall_accuracy = totals[1] / keyword_amount
        overall_f1 = totals[2] / keyword_amount

        return top_keywords, overall_kappa, overall_accuracy, overall_f1



def main():
    newSVM = SVM()


if  __name__ =='__main__':main()
