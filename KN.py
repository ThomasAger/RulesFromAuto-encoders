import DataTasks as dt
import MovieTasks as mt
from scipy import spatial
from sklearn import neighbors
from sklearn.metrics import f1_score, accuracy_score
import numpy as np

def getKNearestMovies(data, x, k):
    movie_names = dt.importString("filmdata/filmNames.txt")
    kd_tree = spatial.KDTree(data)
    kd_query = kd_tree.query(x=x, k=k)
    nearest_distances = kd_query[0][1:]
    k_nearest = kd_query[1][1:]
    nearest_movies = []
    for k in k_nearest:
        nearest_movies.append(movie_names[k].strip())
    print nearest_movies
    return nearest_movies, nearest_distances

def getKNeighbors(vector_path="filmdata/films200.mds/films200.mds", class_path="filmdata/classesGenres/class-All",
                  n_neighbors=1, algorithm="kd_tree", leaf_size=30,
                  training_data=10000, name="normal200"):
    movie_vectors = np.asarray(dt.importVectors(vector_path))
    movie_labels = np.asarray(dt.importLabels(class_path))

    x_train, y_train, x_test, y_test = dt.splitData(training_data, movie_vectors, movie_labels)

    classifier = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=algorithm, leaf_size=leaf_size)
    classifier.fit(x_train, y_train.ravel())
    y_pred = classifier.predict(x_test)

    f1 = f1_score(y_test, y_pred, average='macro')
    accuracy = accuracy_score(y_test, y_pred)
    dt.write1dArray([f1, accuracy], "KNNScores/" + name + ".score")
    print "F1 " + str(f1), "Accuracy", accuracy


def main():
    file_path = "newdata/spaces/"
    filenames = [ "AUTOENCODERN0.4R0.0tanhtanhmse1tanh2004SDA1", "AUTOENCODERN0.6R0.0tanhtanhmse1tanh1004SDA2"]
    for f in range(len(filenames)):
        vector_path=file_path+filenames[f] + ".mds"
        print vector_path[:-4]
        getKNeighbors(vector_path=vector_path, name=filenames[f], class_path="filmdata/classesGenres/class-Comedy")

if  __name__ =='__main__':main()

