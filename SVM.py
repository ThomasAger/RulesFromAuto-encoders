
from sklearn import svm, linear_model
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from skll.metrics import kappa
import DataTasks as dt
import MovieTasks as mt
from scipy import stats, linalg, spatial
import SVMTasks


class SVM:


    def __init__(self, class_type="Phrases", name_distinction="", class_names=None, vector_path=None, class_path=None, class_by_class=True, input_size=200,
                 training_data=10000, amount_of_scores=400,  low_kappa=0.1, high_kappa=0.5, rankSVM=False, amount_to_cut_at=100, largest_cut=21470000):
        print "getting movie data"

        missing_samples = None
        if class_type == "NewKeywords":
            missing_samples="filmdata/MISSING_KEYWORD_ITEMS.txt"

        movie_vectors, movie_labels = mt.getMovieData(class_type=class_type, input_size=input_size, class_path=class_path,
                                                                   class_names=class_names, class_by_class=class_by_class, vector_path=vector_path)

        print "getting file names"
        if class_path is None:
            file_names = dt.getAllFileNamesAndExtensions("filmdata\classes" + class_type)
        else:
            file_names = dt.getAllFileNamesAndExtensions(class_path[:-10])

        print len(movie_labels), len(movie_labels[0])

        print "getting training and test data"


        x_train = movie_vectors[:training_data]
        x_test = movie_vectors[training_data:]

        file_names, movie_labels = SVMTasks.getSampledData(file_names, movie_labels, amount_to_cut_at, largest_cut)
        movie_labels = zip(*movie_labels)
        y_train = movie_labels[:training_data]
        y_test = movie_labels[training_data:]
        print len(y_test), len(y_test[0])

        y_train = zip(*y_train)
        y_test = zip(*y_test)

        movie_names = None
        movie_vectors = None
        movie_labels = None

        kappa_scores, directions =   self.runAllSVMs(len(y_train), amount_of_scores, y_test, y_train, x_train, x_test, class_type, input_size, file_names, low_kappa, high_kappa, rankSVM)

        dt.write1dArray(kappa_scores, "SVMResults/ALL_SCORES_"+ class_type+"_"+name_distinction+".txt")
        dt.write1dArray(file_names, "SVMResults/ALL_NAMES_"+ class_type+"_"+name_distinction+".txt")



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

    def runSVM(self, y_test, y_train, x_train, x_test, class_type, input_size, file_names):
        clf = svm.LinearSVC()
        clf.fit(x_train, y_train)
        direction = clf.coef_
        y_pred = clf.predict(x_test)
        y_pred = y_pred.tolist()
        kappa_score = kappa(y_test, y_pred)
        return kappa_score,  direction

    def runAllSVMs(self, keyword_amount, amount_of_scores, y_test, y_train, x_train, x_test, class_type, input_size, file_names, low_kappa, high_kappa, rankSVM):
        kappa_scores = []
        directions = []
        for y in range(len(y_train)):
            kappa,  direction = self.runSVM(y_test[y], y_train[y], x_train, x_test, class_type, input_size, file_names)
            kappa_scores.append(kappa)
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


def main():
    print "starting"
    """
    fp = "D:\Dropbox\PhD\My Work\Code\MSDA\Python\Data\IMDB\Transformed"

    for x in range(1, 5):
        fn = "msda_representation_sowNL"+ str(x) + "N0.6D1000"+ str(x) + ".mm.txt"
        newSVM = SVM(vector_path=fp+"\\"+fn, class_path="filmdata\classesNewKeywords/class-all",
                 amount_to_cut_at=200, training_data=10000, name_distinction=fn, largest_cut=2500000)
    """
    path="newdata/spaces/"
    fp = "AUTOENCODER1sigmoidsoftmaxbinary_crossentropy2sigmoid[700]FINETUNED"
   # fp = "filmdata/films200.mds/films200"
    newSVM = SVM(vector_path=path+fp+".mds", class_type="Keywords", amount_to_cut_at=200, training_data=10000, name_distinction=fp, largest_cut=9999999999)





if  __name__ =='__main__':main()
