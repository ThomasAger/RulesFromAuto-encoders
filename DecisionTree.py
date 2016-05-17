import numpy as np
import DataTasks as dt
import pydot
from sklearn import tree
from sklearn.metrics import f1_score, accuracy_score
from StringIO import StringIO
from inspect import getmembers
import jsbeautifier
class DecisionTree:
    clf = None
    def __init__(self, cluster_vectors_fn, cluster_labels_fn, movie_names_fn, label_names_fn, cluster_names_fn, filename, training_data, cluster_to_classify, max_depth):

        vectors = dt.importVectors(cluster_vectors_fn)
        labels = dt.importLabels(cluster_labels_fn)
        cluster_names = dt.importString(cluster_names_fn)
        vector_names = dt.importString(movie_names_fn)
        label_names = dt.importString(label_names_fn)
        scores_array = []
        for l in range(len(labels[0])):
            new_labels = [0] * 15000
            for x in range(len(labels)):
                new_labels[x] = labels[x][l]
            x_train = np.asarray(vectors[:training_data])
            x_test = np.asarray(vectors[training_data:])
            y_train = np.asarray(new_labels[:training_data])
            y_test = np.asarray(new_labels[training_data:])


            self.clf = tree.DecisionTreeClassifier( max_depth=max_depth)
            self.clf = self.clf.fit(x_train, y_train)

            y_pred = self.clf.predict(x_test)
            f1 = f1_score(y_test, y_pred, average='binary')
            accuracy = accuracy_score(y_test, y_pred)
            scores = [[label_names[l], "f1", f1, "accuracy", accuracy]]
            print scores[0]
            scores_array.append(scores)

            class_names = [ label_names[l], "NOT "+label_names[l]]
            tree.export_graphviz(self.clf, feature_names=cluster_names, class_names=class_names, out_file='Rules/'+label_names[l]+filename+'.dot', max_depth=10)
            """
            rewrite_dot_file = dt.importString('Rules/'+filename+label_names[l]+'.dot')
            new_dot_file = []
            for s in rewrite_dot_file:
                new_string = s
                if "->" not in s and "digraph" not in s and "node" not in s and "(...)" not in s and "}" not in s:
                    index = s.index("value")
                    new_string = s[:index] + '"] ;'
                new_dot_file.append(new_string)
            dt.write1dArray(new_dot_file, 'Rules/Cleaned'+filename+label_names[l]+'.dot')
            """
            graph = pydot.graph_from_dot_file('Rules/'+label_names[l]+filename+'.dot')
            graph.write_png('Rules/Images/'+label_names[l]+filename+".png")
            self.get_code(self.clf, cluster_names, class_names, label_names[l]+filename)
        dt.write1dArray(scores_array, 'Rules/Scores/'+filename+'.scores')

    def get_clf(self):
        return self.clf

    def get_code(self, tree, feature_names, class_names, filename):
        left      = tree.tree_.children_left
        right     = tree.tree_.children_right
        threshold = tree.tree_.threshold
        value = tree.tree_.value

        #print tree.tree_.feature, len(tree.tree_.feature
        # )
        features = []
        for i in tree.tree_.feature:
            if i != -2 or i <= 200:
                features.append(feature_names[i])
        rules_array = []
        def recurse(left, right, threshold, features,  node):
                if (threshold[node] != -2):
                        line = "IF ( " + features[node] + " <= " + str(threshold[node]) + " ) {"
                        rules_array.append(line)
                        if left[node] != -1:
                                recurse (left, right, threshold, features,left[node])
                        line = "} ELSE {"
                        rules_array.append(line)
                        if right[node] != -1:
                                recurse (left, right, threshold, features,right[node])
                        line = "}"
                        rules_array.append(line)
                else:
                        if value[node][0][0] >= value[node][0][1]:
                            line = "return", class_names[0]
                            rules_array.append(line)
                        else:
                            line = "return", class_names[1]
                            rules_array.append(line)
        recurse(left, right, threshold, features, 0)
        dt.write1dArray(rules_array, "Rules/Statements/"+filename+".rules")
        cleaned = jsbeautifier.beautify_file("Rules/Statements/"+filename+".rules")
        file = open("Rules/Statements/"+filename+".rules", "w")
        file.write(cleaned)
        file.close()


def main():
    """
    low_threshold = 0.1
    high_threshold = 0.44
    filename = "films200"
    cluster_vectors_fn = "filmdata/films200.mds/" + filename + ".mds"
    cluster_names_fn = "Clusters/" + filename + "Cut2000"+  "LeastSimilarHIGH"+str(high_threshold)+","+str(low_threshold)+".names"
    label_names_fn = "filmdata/classesGenres/class-Action"
    cluster_labels_fn = "filmdata/classesGenres/class-Action"
    movie_names_fn = "filmdata/filmNames.txt"
    tree_fn = "films200Genres"
    max_depth = 9999999
    clf = Rules(cluster_vectors_fn, cluster_labels_fn, movie_names_fn, label_names_fn, cluster_names_fn,tree_fn,  10000, 0, max_depth)
    """
    """
    amt_of_labels = "P0.1"
    max_depth = 4
    low_threshold = 0.1
    high_threshold = 0.4
    filename = "AUTOENCODERN0.5R0.0tanhtanhmse1tanh2004SDA1"
    cluster_vectors_fn = "Rankings/" + filename + ".space"
    cluster_names_fn = "Clusters/" + filename + "Cut2000"+  "LeastSimilarHIGH"+str(high_threshold)+","+str(low_threshold)+".names"
    labels_fn = filename
    label_names_fn = "Clusters/" + labels_fn + "Cut2000"+  "LeastSimilarHIGH"+str(high_threshold)+","+str(low_threshold)+".names"
    cluster_labels_fn = "Rankings/" + labels_fn + amt_of_labels + ".labels"
    movie_names_fn = "filmdata/filmNames.txt"
    tree_fn = "0.5to0.5AEL1L1" + str(max_depth)
    clf = DecisionTree(cluster_vectors_fn, cluster_labels_fn, movie_names_fn, label_names_fn, cluster_names_fn,tree_fn,  10000, 0, max_depth)
    """

    cluster_to_classify = -1
    amt_of_labels = "P0.1"
    max_depth = 3

    """
    tree_fn = "0.5to0.8AEL1L4D" + str(max_depth)
    low_threshold = 0.1
    high_threshold = 0.4
    filename = "AUTOENCODERN0.5R0.0tanhtanhmse1tanh2004SDA1"
    cluster_vectors_fn = "Rankings/" + filename + ".space"
    cluster_names_fn = "Clusters/" + filename + "Cut2000"+  "LeastSimilarHIGH"+str(high_threshold)+","+str(low_threshold)+".names"
    low_threshold = 0.04
    high_threshold = 0.21
    labels_fn = "AUTOENCODERN0.8R0.0tanhtanhmse1tanh254SDA"
    label_names_fn = "Clusters/" + labels_fn + "1Cut2003"+  "LeastSimilarHIGH"+str(high_threshold)+","+str(low_threshold)+".names"
    cluster_labels_fn = "Rankings/" + labels_fn + amt_of_labels + ".labels"
    movie_names_fn = "filmdata/filmNames.txt"
    clf = DecisionTree(cluster_vectors_fn, cluster_labels_fn, movie_names_fn, label_names_fn, cluster_names_fn,tree_fn,  10000, cluster_to_classify, max_depth)
    """
    low_threshold = 0.1
    high_threshold = 0.4
    filename = "AUTOENCODERN0.4R0.0tanhtanhmse2tanh2004SDA1"
    cluster_vectors_fn = "Rankings/" + filename + amt_of_labels + ".space"
    cluster_names_fn = "Clusters/" + filename + "Cut2000"+  "LeastSimilarHIGH"+str(high_threshold)+","+str(low_threshold)+".names"
    low_threshold = 0.08
    high_threshold = 0.37
    labels_fn = "AUTOENCODERN0.6R0.0tanhtanhmse2tanh1004SDA2"
    label_names_fn = "Clusters/" + labels_fn + "Cut2001"+  "LeastSimilarHIGH"+str(high_threshold)+","+str(low_threshold)+".names"
    cluster_labels_fn = "Rankings/" + labels_fn + amt_of_labels + ".labels"
    movie_names_fn = "filmdata/filmNames.txt"
    tree_fn = "0.4to0.6AEL1L2" + str(max_depth)
    clf = DecisionTree(cluster_vectors_fn, cluster_labels_fn, movie_names_fn, label_names_fn, cluster_names_fn,tree_fn,  10000, cluster_to_classify, max_depth)

    low_threshold = 0.08
    high_threshold = 0.37
    filename = "AUTOENCODERN0.6R0.0tanhtanhmse2tanh1004SDA2"
    cluster_vectors_fn = "Rankings/" + filename + amt_of_labels + ".space"
    cluster_names_fn = "Clusters/" + filename + "Cut2001"+  "LeastSimilarHIGH"+str(high_threshold)+","+str(low_threshold)+".names"
    low_threshold = 0.06
    high_threshold = 0.28
    labels_fn = "AUTOENCODERN0.8R0.0tanhtanhmse2tanh504SDA3"
    label_names_fn = "Clusters/" + labels_fn + "Cut2002"+  "LeastSimilarHIGH"+str(high_threshold)+","+str(low_threshold)+".names"
    cluster_labels_fn = "Rankings/" + labels_fn + amt_of_labels + ".labels"
    movie_names_fn = "filmdata/filmNames.txt"
    tree_fn = "0.6to0.8AEL2L3" + str(max_depth)
    clf = DecisionTree(cluster_vectors_fn, cluster_labels_fn, movie_names_fn, label_names_fn, cluster_names_fn,tree_fn,  10000, cluster_to_classify, max_depth)

    low_threshold = 0.06
    high_threshold = 0.28
    filename = "AUTOENCODERN0.8R0.0tanhtanhmse2tanh504SDA3"
    cluster_vectors_fn = "Rankings/" + filename + amt_of_labels + ".space"
    cluster_names_fn = "Clusters/" + filename + "Cut2002"+  "LeastSimilarHIGH"+str(high_threshold)+","+str(low_threshold)+".names"
    low_threshold = 0.04
    high_threshold = 0.21
    labels_fn = "AUTOENCODERN1.0R0.0tanhtanhmse2tanh254SDA4"
    label_names_fn = "Clusters/" + labels_fn + "Cut2003"+  "LeastSimilarHIGH"+str(high_threshold)+","+str(low_threshold)+".names"
    cluster_labels_fn = "Rankings/" + labels_fn + amt_of_labels + ".labels"
    movie_names_fn = "filmdata/filmNames.txt"
    tree_fn = "0.8to1.0AEL3L4" + str(max_depth)
    clf = DecisionTree(cluster_vectors_fn, cluster_labels_fn, movie_names_fn, label_names_fn, cluster_names_fn,tree_fn,  10000, cluster_to_classify, max_depth)

    """
    low_threshold = 0.04
    high_threshold = 0.21
    filename = "AUTOENCODERN0.8R0.0tanhtanhmse1tanh254SDA"
    cluster_vectors_fn = "Rankings/" + filename + ".space"
    cluster_names_fn = "Clusters/" + filename + "1Cut2003"+  "LeastSimilarHIGH"+str(high_threshold)+","+str(low_threshold)+".names"
    labels_fn = "AUTOENCODERN0.8R0.0tanhtanhmse1tanh254SDA"
    label_names_fn = "Clusters/" + labels_fn + "1Cut2003"+  "LeastSimilarHIGH"+str(high_threshold)+","+str(low_threshold)+".names"
    cluster_labels_fn = "Rankings/" + labels_fn + amt_of_labels + ".labels"
    movie_names_fn = "filmdata/filmNames.txt"
    max_depth = 200
    tree_fn = "0.8to0.8AEL1L1" + str(max_depth)
    clf = DecisionTree(cluster_vectors_fn, cluster_labels_fn, movie_names_fn, label_names_fn, cluster_names_fn,tree_fn,  10000, cluster_to_classify, max_depth)
    """






if  __name__ =='__main__':main()