import numpy as np
import DataTasks as dt
from sklearn import tree
from sklearn.metrics import f1_score, accuracy_score
from StringIO import StringIO
from inspect import getmembers
import pandas
class Rules:
    clf = None
    def __init__(self, cluster_vectors_fn, cluster_labels_fn, movie_names_fn, label_names_fn, cluster_names_fn, filename, training_data):

        vectors = dt.importVectors(cluster_vectors_fn)
        labels = dt.importLabels(cluster_labels_fn)
        cluster_names = dt.importString(cluster_names_fn)
        vector_names = dt.importString(movie_names_fn)
        label_names = dt.importString(label_names_fn)

        x_train = np.asarray(vectors[:training_data])
        x_test = np.asarray(vectors[training_data:])
        y_train = np.asarray(labels[:training_data])
        y_test = np.asarray(labels[training_data:])

        var = pandas.qcut(vectors[0], 1, labels=["10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%", "90%", "100%"])
        print var
        #RIPPER GOES HERE

        y_pred = self.clf.predict(x_test)
        f1 = f1_score(y_test, y_pred, average='macro')
        accuracy = accuracy_score(y_test, y_pred)
        print "f1", f1, "accuracy", accuracy



def main():

    cluster_to_classify = 0
    low_threshold = 0.1
    high_threshold = 0.4
    amt_of_labels = ""
    filename = "AUTOENCODERN0.5R0.0tanhtanhmse1tanh2004SDA1"
    cluster_vectors_fn = "Rankings/" + filename + ".space"
    cluster_names_fn = "Clusters/" + filename + "Cut2000"+  "LeastSimilarHIGH"+str(high_threshold)+","+str(low_threshold)+".names"
    low_threshold = 0.08
    high_threshold = 0.37
    labels_fn = "AUTOENCODERN0.6R0.0tanhtanhmse1tanh1004SDA"
    label_names_fn = "Clusters/" + labels_fn + "1Cut2001"+  "LeastSimilarHIGH"+str(high_threshold)+","+str(low_threshold)+".names"
    cluster_labels_fn = "Rankings/" + labels_fn + amt_of_labels + ".labels"
    movie_names_fn = "filmdata/filmNames.txt"
    max_depth = 200
    tree_fn = "0.5to0.6AEL1L2" + str(max_depth)
    clf = Rules(cluster_vectors_fn, cluster_labels_fn, movie_names_fn, label_names_fn, cluster_names_fn,tree_fn,  10000)

    low_threshold = 0.08
    high_threshold = 0.37
    amt_of_labels = ""
    filename = "AUTOENCODERN0.6R0.0tanhtanhmse1tanh1004SDA"
    cluster_vectors_fn = "Rankings/" + filename + ".space"
    cluster_names_fn = "Clusters/" + filename + "1Cut2001"+  "LeastSimilarHIGH"+str(high_threshold)+","+str(low_threshold)+".names"
    low_threshold = 0.06
    high_threshold = 0.28
    labels_fn = "AUTOENCODERN0.7R0.0tanhtanhmse1tanh504SDA"
    label_names_fn = "Clusters/" + labels_fn + "1Cut2002"+  "LeastSimilarHIGH"+str(high_threshold)+","+str(low_threshold)+".names"
    cluster_labels_fn = "Rankings/" + labels_fn + amt_of_labels + ".labels"
    movie_names_fn = "filmdata/filmNames.txt"
    max_depth = 200
    tree_fn = "0.6to0.7AEL2L3" + str(max_depth)
    clf = Rules(cluster_vectors_fn, cluster_labels_fn, movie_names_fn, label_names_fn, cluster_names_fn,tree_fn,  10000)

    low_threshold = 0.06
    high_threshold = 0.28
    amt_of_labels = ""
    filename = "AUTOENCODERN0.7R0.0tanhtanhmse1tanh504SDA"
    cluster_vectors_fn = "Rankings/" + filename + ".space"
    cluster_names_fn = "Clusters/" + filename + "1Cut2002"+  "LeastSimilarHIGH"+str(high_threshold)+","+str(low_threshold)+".names"
    low_threshold = 0.04
    high_threshold = 0.21
    labels_fn = "AUTOENCODERN0.8R0.0tanhtanhmse1tanh254SDA"
    label_names_fn = "Clusters/" + labels_fn + "1Cut2003"+  "LeastSimilarHIGH"+str(high_threshold)+","+str(low_threshold)+".names"
    cluster_labels_fn = "Rankings/" + labels_fn + amt_of_labels + ".labels"
    movie_names_fn = "filmdata/filmNames.txt"
    max_depth = 200
    tree_fn = "0.7to0.8AEL3L4" + str(max_depth)
    clf = Rules(cluster_vectors_fn, cluster_labels_fn, movie_names_fn, label_names_fn, cluster_names_fn,tree_fn,  10000)

    low_threshold = 0.04
    high_threshold = 0.21
    amt_of_labels = ""
    filename = "AUTOENCODERN0.8R0.0tanhtanhmse1tanh254SDA"
    cluster_vectors_fn = "Rankings/" + filename + ".space"
    cluster_names_fn = "Clusters/" + filename + "1Cut2003"+  "LeastSimilarHIGH"+str(high_threshold)+","+str(low_threshold)+".names"
    labels_fn = "AUTOENCODERN0.8R0.0tanhtanhmse1tanh254SDA"
    label_names_fn = "Clusters/" + labels_fn + "1Cut2003"+  "LeastSimilarHIGH"+str(high_threshold)+","+str(low_threshold)+".names"
    cluster_labels_fn = "Rankings/" + labels_fn + amt_of_labels + ".labels"
    movie_names_fn = "filmdata/filmNames.txt"
    max_depth = 200
    tree_fn = "0.8to0.8AEL1L1" + str(max_depth)
    clf = Rules(cluster_vectors_fn, cluster_labels_fn, movie_names_fn, label_names_fn, cluster_names_fn,tree_fn,  10000)






if  __name__ =='__main__':main()