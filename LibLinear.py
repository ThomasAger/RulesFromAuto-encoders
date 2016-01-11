# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import numpy as np
from sklearn.metrics import f1_score
from skll.metrics import kappa
from sklearn import svm
import DataTasks as dt
import MovieTasks as mt
from sklearn.metrics import accuracy_score
np.set_printoptions(threshold=np.nan)

class_type = "Keywords"
class_names = None
vector_path = None
#class_names = ["absurd", "abstract", "abrupt"]
vector_path = "newdata\spaces\Genres-standard200400.mds"
class_by_class = True
input_size = 200
training_data = 10000
amount_of_scores = 15

movie_names, movie_vectors, movie_labels = mt.getMovieData(class_type=class_type, input_size=input_size, class_names=class_names,
                                                           class_by_class=class_by_class, vector_path=vector_path)

file_names = dt.getAllFileNames("filmdata\classes" + class_type)
print file_names
n_train = movie_names[:training_data]
x_train = []
for x in range(15000):
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

"""
for x in x_train:
    x = np.asarray(x)

for y in y_train:
    y = np.asarray(y)

print "converted to numpy"
"""
"""
We have used the LIBSVM 27 implementation. Because default values of the parameters yielded very poor results,
we have used a grid search procedure to find the optimal value of the C parameter for every class. To this end, the
training data for each class was split into 2/3 training and 1/3 validation. Moreover, to address class imbalance we
under-sampled negative training examples, such that the ratio between positive and negative training examples
was at least 1/2.
"""
total_accuracy = 0.0
amount = len(y_test)
total_f1_score = 0
total_kappa = 0
directions = []
kappa_scores = []
# Training an SVM for every keyword
for x in range(len(y_train)):
    clf = svm.LinearSVC()
    clf.fit(x_train[x], y_train[x])
    clf.decision_function(x_test)
    directions.append(clf.coef_)
    y_pred = clf.predict(x_test)
    y_pred = y_pred.tolist()
    kappa_score = kappa(y_test[x], y_pred)
    kappa_scores.append(kappa_score)
    total_kappa = total_kappa + kappa_score
    accuracy = accuracy_score(y_test[x], y_pred)
    total_accuracy = total_accuracy + accuracy
    f1 = f1_score(y_test[x], y_pred, average='macro')
    total_f1_score = total_f1_score + f1
    print "kappa:", kappa_score, "accuracy:", accuracy, "f1_score:", f1
    """if vector_path is None:
        dt.write1dArray(directions[x], "newdata/directions/keywords/"+ str(input_size)+"/"+file_names[x]+ str(input_size) + "MDS" + str(kappa_score))
    else:
        dt.write1dArray(directions[x], "newdata/directions/keywords/"+ str(400)+"/"+file_names[x]+ str(400) + "NN" + str(kappa_score))"""
kappa_scores = np.asarray(kappa_scores)
top_scores = np.argpartition(kappa_scores, -amount_of_scores)[-amount_of_scores:]
top_keywords = []
top_keywordz = []
for i in top_scores:
    print i
    top_keywordz.append(file_names[i])
print "Top keywords:", top_keywords
print "Top keywordz:", top_keywordz
overall_accuracy = total_accuracy / amount
overall_f1 = total_f1_score / amount
overall_kappa = total_kappa / amount

print "overall accuracy:", overall_accuracy, "overall F1", overall_f1, "Overall Kappa", overall_kappa
