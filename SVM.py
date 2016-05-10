
from sklearn import svm, linear_model
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from skll.metrics import kappa
import DataTasks as dt
import MovieTasks as mt
from scipy import stats, linalg, spatial
import SVMTasks


class SVM:


    def __init__(self, name_distinction="", class_names=None, vector_path=None, class_path=None, class_by_class=True, input_size=200,
                 training_data=10000, amount_of_scores=400,  low_kappa=0.1, high_kappa=0.5, rankSVM=False, amount_to_cut_at=100, largest_cut=21470000):
        print "getting movie data"

        movie_vectors = dt.importVectors(vector_path)
        movie_labels = dt.importLabels(class_path)
        print "getting file names"

        file_names = dt.getFns(class_path[:-10])

        print len(movie_labels), len(movie_labels[0])

        print "getting training and test data"

        x_train = np.asarray(movie_vectors[:training_data])
        x_test = np.asarray(movie_vectors[training_data:])

        movie_labels = zip(*movie_labels)
        file_names, movie_labels = getSampledData(file_names, movie_labels, amount_to_cut_at, largest_cut)
        movie_labels = zip(*movie_labels)

        y_train = movie_labels[:training_data]
        y_test = movie_labels[training_data:]
        y_train = np.asarray(zip(*y_train))
        y_test = np.asarray(zip(*y_test))



        print len(y_train), len(y_test), training_data

        print "getting kappa scores"

        kappa_scores, directions =   self.runAllSVMs(y_test, y_train, x_train, x_test, file_names)

        dt.write1dArray(kappa_scores, "SVMResults/ALL_SCORES_"+name_distinction+".txt")
        dt.write1dArray(file_names, "SVMResults/ALL_NAMES_"+name_distinction+".txt")

        dt.write2dArray(directions, "directions/"+name_distinction+".directions")



    """
    We have used the LIBSVM 27 implementation. Because default values of the parameters yielded very poor results,
    we have used a grid search procedure to find the optimal value of the C parameter for every class. To this end, the
    training data for each class was split into 2/3 training and 1/3 validation. Moreover, to address class imbalance we
    under-sampled negative training examples, such that the ratio between positive and negative training examples
    was at least 1/2.
    """

    def runRankSVM(self, y_test, y_train, x_train, x_test, class_type, input_size, file_names, keyword):
        clf = svm.SVC(kernel='linear', C=.1)
        clf.fit(x_train[keyword], y_train[keyword])
        #clf.decision_function(x_test)
        direction = clf.coef_
        return direction

    def runSVM(self, y_test, y_train, x_train, x_test):
        clf = svm.LinearSVC()
        clf.fit(x_train, y_train)
        direction = clf.coef_.tolist()[0]
        y_pred = clf.predict(x_test)
        y_pred = y_pred.tolist()
        kappa_score = kappa(y_test, y_pred)
        return kappa_score,  direction

    def runAllSVMs(self,  y_test, y_train, x_train, x_test, file_names):
        kappa_scores = []
        directions = []
        for y in range(len(y_train)):
            kappa, direction = self.runSVM(y_test[y], y_train[y], x_train, x_test)
            kappa_scores.append(kappa)
            directions.append(direction)
            print y, kappa, file_names[y]

        return kappa_scores,  directions

    def rankByDirections(self, movie_names, movie_vectors, file_names, directions):
        dict = {}
        for d in range(len(directions)):
            unsorted_ranks = []
            for v in range(len(movie_vectors)):
                unsorted_ranks.append(linalg.norm(directions[d] * movie_vectors[v]))
            unsorted_ranks = np.asarray(unsorted_ranks)
            sorted_ranks = np.argpartition(unsorted_ranks, -len(directions))[-len(directions):]
            top_ranked_movies = []
            for s in sorted_ranks:
                top_ranked_movies.append(movie_names[s])
            dict[file_names[d]] = top_ranked_movies
        return dict

    def getSampledData(self, file_names, movie_labels, amount_to_cut_at, largest_cut):
        print len(movie_labels)
        print len(movie_labels[0])

        for yt in range(len(movie_labels)):
            y1 = 0
            y0 = 0
            for y in range(len(movie_labels[yt])):
                if movie_labels[yt][y] == 1:
                    y1 += 1
                if movie_labels[yt][y] == 0:
                    y0 += 1

            if y1 < amount_to_cut_at or y1 > largest_cut:
                print yt, "len(0):", y0, "len(1):", y1, "DELETED", file_names[yt]
                movie_labels[yt] = None
                file_names[yt] = None
                continue
            print yt, "len(0):", y0, "len(1):", y1, file_names[yt]

        file_names = [x for x in file_names if x is not None]
        movie_labels = [x for x in movie_labels if x is not None]

        return file_names, movie_labels

def main():
    path="Clusters/"
    #path="filmdata/films200.mds/"
    #array = ["700", "400", "100"]
    filenames = [ "AUTOENCODER1.0tanhtanhmse2tanh2004SDACut2004LeastSimilarHIGH0.18,0.01"]

    """

                 "AUTOENCODER0.2tanhtanhmse15tanh[1000]4SDA1","AUTOENCODER0.2tanhtanhmse60tanh[200]4SDA2","AUTOENCODER0.2tanhtanhmse30tanh[1000]4SDA3",
                 "AUTOENCODER0.2tanhtanhmse60tanh[200]4SDA4"
    """
    cut = 200
    for f in range(1, 6):
        newSVM = SVM(vector_path=path+filenames[0]+".clusters", class_path="filmdata/classesPhrases/class-All", amount_to_cut_at=cut, training_data=10000, name_distinction=filenames[0]+"Cut"+str(cut)+str(f), largest_cut=9999999999)




if  __name__ =='__main__':main()
