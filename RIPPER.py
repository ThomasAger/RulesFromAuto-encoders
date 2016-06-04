import numpy as np
import DataTasks as dt
from sklearn import tree
from sklearn.metrics import f1_score, accuracy_score
from StringIO import StringIO
from inspect import getmembers
import pandas
import subprocess
class RIPPER:
    clf = None
    def __init__(self, cluster_vectors_fn, cluster_labels_fn, movie_names_fn, label_names_fn, cluster_names_fn, filename, training_data):

        vectors = dt.importDiscreteVectors(cluster_vectors_fn)
        labels = dt.importDiscreteVectors(cluster_labels_fn)
        cluster_names = dt.importString(cluster_names_fn)
        vector_names = dt.importString(movie_names_fn)
        label_names = dt.importString(label_names_fn)

        x_train = np.asarray(vectors[:training_data])
        x_test = np.asarray(vectors[training_data:])
        y_train = np.asarray(labels[:training_data])
        y_test = np.asarray(labels[training_data:])

        # Create attribute file

        for l in range(len(labels[0])):
            print label_names[l]
            attribute_file = []
            training_data_file = []
            test_data_file = []
            new_labels = [0] * 15000
            for x in range(len(labels)):
                new_labels[x] = labels[x][l]
            attribute_file.append("label_"+label_names[l] + " 10% 20% 30% 40% 50% 60% 70% 80% 90% 100%")
            for v in range(len(cluster_names)):
                attribute_file.append(cluster_names[v] + " 10% 20% 30% 40% 50% 60% 70% 80% 90% 100%")

            for v in range(len(vectors)):
                float_string = str(new_labels[v])
                for f in vectors[v]:
                    float_string = float_string + " " + f
                if v < training_data:
                    training_data_file.append(float_string)
                else:
                    test_data_file.append(float_string)
            dt.write1dArray(attribute_file, "RIPPERk/data/"+filename + label_names[l]+"-attr.txt")
            dt.write1dArray(training_data_file, "RIPPERk/data/"+filename + label_names[l]+"-train.txt")
            dt.write1dArray(test_data_file, "RIPPERk/data/"+filename + label_names[l]+"-test.txt")
            call = "python ripperk.py -e learn -a RIPPERk/data/"+filename + label_names[l]+"-attr.txt -c label_"+label_names[l]+\
                   " -t RIPPERk/data/"+filename + label_names[l]+"-train.txt -m RIPPERk/results/"+filename+ label_names[l]+\
                   ".dat -o RIPPERk/results/"+ filename+ label_names[l]+".txt -k 2 -p 1"
            subprocess.call(call)



        """
        -e: the execution method (learn|classify)
        -a: the attribute file location.
        -c: the defining attribute in the attribute file (a.k.a what we are trying to predict).
        -t: the training/testing file location.
        -m: the model file (machine readable results).
        -o: the output file (human readable results).
        e.g.
        python ripperk.py -e learn -a "../data/restaurant-attr.txt" -c WillWait -t "../data/restaurant-train.txt"
        -m "../results/restaurant-model.dat" -o "../results/restaurant-model.txt" -k 2 -p 1
        """


        #RIPPER GOES HERE
        """
        y_pred = self.clf.predict(x_test)
        f1 = f1_score(y_test, y_pred, average='macro')
        accuracy = accuracy_score(y_test, y_pred)
        print "f1", f1, "accuracy", accuracy
        """


def main():

    amt_of_labels = "P0.1"


    low_threshold = 0.06
    high_threshold = 0.28
    filename = "AUTOENCODERN0.8R0.0tanhtanhmse2tanh504SDA3"
    cluster_vectors_fn = "Rankings/" + filename + amt_of_labels + ".discrete"
    cluster_names_fn = "Clusters/" + filename + "Cut2002"+  "LeastSimilarHIGH"+str(high_threshold)+","+str(low_threshold)+".names"
    low_threshold = 0.04
    high_threshold = 0.21
    labels_fn = "AUTOENCODERN1.0R0.0tanhtanhmse2tanh254SDA4"
    label_names_fn = "Clusters/" + labels_fn + "Cut2003"+  "LeastSimilarHIGH"+str(high_threshold)+","+str(low_threshold)+".names"
    cluster_labels_fn = "Rankings/" + labels_fn + amt_of_labels + ".discrete"
    movie_names_fn = "filmdata/filmNames.txt"
    tree_fn = "0.8to1.0AEL3L4"
    clf = RIPPER(cluster_vectors_fn, cluster_labels_fn, movie_names_fn, label_names_fn, cluster_names_fn,tree_fn,  10000)

    low_threshold = 0.1
    high_threshold = 0.44
    filename = "films200"
    cluster_vectors_fn = "Rankings/" + filename + amt_of_labels + ".discrete"
    cluster_names_fn = "Clusters/" + filename + "Cut2000"+  "LeastSimilarHIGH"+str(high_threshold)+","+str(low_threshold)+".names"
    low_threshold = 0.1
    high_threshold = 0.4
    labels_fn = "AUTOENCODERN0.4R0.0tanhtanhmse2tanh2004SDA1"
    label_names_fn = "Clusters/" + labels_fn + "Cut2000"+  "LeastSimilarHIGH"+str(high_threshold)+","+str(low_threshold)+".names"
    cluster_labels_fn = "Rankings/" + labels_fn + amt_of_labels + ".discrete"
    movie_names_fn = "filmdata/filmNames.txt"
    tree_fn = "MDSto0.4AEL0L1"
    clf = RIPPER(cluster_vectors_fn, cluster_labels_fn, movie_names_fn, label_names_fn, cluster_names_fn,tree_fn,  10000)


    print "FIRST AE TO SECOND AE"
    low_threshold = 0.1
    high_threshold = 0.4
    filename = "AUTOENCODERN0.4R0.0tanhtanhmse2tanh2004SDA1"
    cluster_vectors_fn = "Rankings/" + filename + amt_of_labels + ".discrete"
    cluster_names_fn = "Clusters/" + filename + "Cut2000"+  "LeastSimilarHIGH"+str(high_threshold)+","+str(low_threshold)+".names"
    low_threshold = 0.08
    high_threshold = 0.37
    labels_fn = "AUTOENCODERN0.6R0.0tanhtanhmse2tanh1004SDA2"
    label_names_fn = "Clusters/" + labels_fn + "Cut2001"+  "LeastSimilarHIGH"+str(high_threshold)+","+str(low_threshold)+".names"
    cluster_labels_fn = "Rankings/" + labels_fn + amt_of_labels + ".discrete"
    movie_names_fn = "filmdata/filmNames.txt"
    tree_fn = "0.4to0.6AEL1L2"
    clf = RIPPER(cluster_vectors_fn, cluster_labels_fn, movie_names_fn, label_names_fn, cluster_names_fn,tree_fn,  10000)

    low_threshold = 0.08
    high_threshold = 0.37
    filename = "AUTOENCODERN0.6R0.0tanhtanhmse2tanh1004SDA2"
    cluster_vectors_fn = "Rankings/" + filename + amt_of_labels + ".discrete"
    cluster_names_fn = "Clusters/" + filename + "Cut2001"+  "LeastSimilarHIGH"+str(high_threshold)+","+str(low_threshold)+".names"
    low_threshold = 0.06
    high_threshold = 0.28
    labels_fn = "AUTOENCODERN0.8R0.0tanhtanhmse2tanh504SDA3"
    label_names_fn = "Clusters/" + labels_fn + "Cut2002"+  "LeastSimilarHIGH"+str(high_threshold)+","+str(low_threshold)+".names"
    cluster_labels_fn = "Rankings/" + labels_fn + amt_of_labels + ".discrete"
    movie_names_fn = "filmdata/filmNames.txt"
    tree_fn = "0.6to0.8AEL2L3"
    clf = RIPPER(cluster_vectors_fn, cluster_labels_fn, movie_names_fn, label_names_fn, cluster_names_fn,tree_fn,  10000)






if  __name__ =='__main__':main()