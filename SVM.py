
from sklearn import svm, linear_model
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from skll.metrics import kappa
import DataTasks as dt
import MovieTasks as mt
from scipy import stats, linalg


class SVM:


    def __init__(self, class_type="Phrases", name_distinction="", class_names=None, vector_path=None, class_by_class=True, input_size=200,
                 training_data=14000, amount_of_scores=400, low_kappa=0.1, high_kappa=0.4, rankSVM=False):

        movie_names, movie_vectors, movie_labels = mt.getMovieData(class_type=class_type, input_size=input_size, class_names=class_names,
                                                                   class_by_class=class_by_class, vector_path=vector_path)


        file_names = dt.getAllFileNames("filmdata\classes" + class_type)

        n_train, x_train, y_train, n_test, x_test, y_test = self.getSampledData(movie_names, movie_vectors, movie_labels, training_data)

        top_keywords, top_kappas, high_keywords, low_keywords, high_directions, low_directions, overall_kappa, overall_accuracy, overall_f1, directions = self.runAllSVMs(len(y_train), amount_of_scores,
                                                y_test, y_train, x_train, x_test, class_type, input_size, file_names, low_kappa, high_kappa, rankSVM)

        #direction_ranks_dict = self.rankByDirections(movie_names, movie_vectors, file_names, high_directions)

        #print direction_ranks_dict

        row = [overall_kappa, overall_accuracy, overall_f1]

        print row
        print high_keywords
        print low_keywords
        dt.write1dArray(top_keywords, "filmdata/allTOP_SCORES_" + str(amount_of_scores) + "_" + class_type+"_"+name_distinction+".txt")
        dt.write1dArray(top_kappas, "filmdata/allTOP_KAPPAS_" + str(amount_of_scores) + "_"+ class_type+"_"+name_distinction+".txt")

        dt.write1dArray(high_keywords, "filmdata/allhigh_keywords_"+ class_type+"_"+name_distinction+".txt")
        dt.write1dArray(low_keywords, "filmdata/alllow_keywords_"+ class_type+"_"+name_distinction+".txt")

        dt.write1dArray(high_directions, "filmdata/allhigh_"+str(amount_of_scores)+"_" +"LowPhrases"+"_" + str(input_size) + "_Round2.txt")
        dt.writeToSheet(row, "filmdata/experiments/experimenttest.csv")

        print "Kappa:", overall_kappa, "Accuracy:", overall_accuracy, "F1", overall_f1

    def getSampledData(self, movie_names, movie_vectors, movie_labels, training_data):

        n_train = movie_names[:training_data]
        x_train = []
        for x in range(len(movie_vectors)):
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
                    y1 += 1
                if y_train[yt][y] == 0:
                    y0 += 1
            y = 0
            while(y0 > int(y1*3)):
                if y_train[yt][y] == 0:
                    del x_train[yt][y]
                    del y_train[yt][y]
                    y0 -= 1
                else:
                    y += 1
            print "len(0):", y0, "len(1):", y1

        return n_train, x_train, y_train, n_test, x_test, y_test


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

    def runSVM(self, y_test, y_train, x_train, x_test, class_type, input_size, file_names, keyword):
        clf = svm.LinearSVC()
        clf.fit(x_train[keyword], y_train[keyword])
        #clf.decision_function(x_test)
        direction = clf.coef_
        y_pred = clf.predict(x_test)
        y_pred = y_pred.tolist()
        kappa_score = kappa(y_test[keyword], y_pred)
        accuracy = accuracy_score(y_test[keyword], y_pred)
        f1 = f1_score(y_test[keyword], y_pred, average='macro')
        return kappa_score, accuracy, f1, direction

    def runAllSVMs(self, keyword_amount, amount_of_scores, y_test, y_train, x_train, x_test, class_type, input_size, file_names, low_kappa, high_kappa, rankSVM):
        totals = np.zeros(3)
        kappa_scores = []
        accuracy_scores = []
        f1_scores = []

        directions = []
        for x in range(keyword_amount-1):
            if rankSVM:
                direction = self.runRankSVM(y_test, y_train, x_train, x_test, class_type, input_size, file_names, x)
                directions.append(direction)
            else:
                kappa, accuracy, f1, direction = self.runSVM(y_test, y_train, x_train, x_test, class_type, input_size, file_names, x)
                kappa_scores.append(kappa)
                print kappa, len(x_train[x]), len(y_train[x])
                accuracy_scores.append(accuracy)
                f1_scores.append(f1)
                directions.append(direction)

        top_scores = np.argpartition(kappa_scores, -amount_of_scores)[-amount_of_scores:]
        top_keywords = []
        top_kappas = []
        for i in top_scores:
            top_keywords.append(file_names[i])
            top_kappas.append(kappa_scores[i])


        high_kappas = np.where(np.diff(kappa_scores) > 0.5)[0] + 1
        low_kappas = np.where(np.diff(kappa_scores) > 0.1)[0] + 1



        for val in kappa_scores:
            totals[0] += val

        for val in accuracy_scores:
            totals[1] += val

        for val in f1_scores:
            totals[2] += val

        kappa_scores = np.asarray(kappa_scores)

        high_keywords = []
        low_keywords = []
        high_directions = []
        low_directions = []
        for val in high_kappas:
            high_keywords.append(file_names[val])
            high_directions.append(file_names[val])

        for val in low_kappas:
            low_keywords.append(file_names[val])
            low_directions.append(file_names[val])


        overall_kappa = totals[0] / keyword_amount
        overall_accuracy = totals[1] / keyword_amount
        overall_f1 = totals[2] / keyword_amount




        return top_keywords, top_kappas, high_keywords, low_keywords, high_directions, low_directions, overall_kappa, overall_accuracy, overall_f1, directions

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
    newSVM = SVM(vector_path="newdata/spaces/All-Layer-20-L.mds", name_distinction="All-20")
    newSVM = SVM(vector_path="newdata/spaces/All-Layer-100-L.mds", name_distinction="All-100")
    newSVM = SVM(vector_path="newdata/spaces/All-Layer-200-L.mds", name_distinction="All-200")
    newSVM = SVM(vector_path="newdata/spaces/Keywords-Layer-20-L.mds", name_distinction="Keywords-20")
    newSVM = SVM(vector_path="newdata/spaces/Keywords-Layer-100-L.mds", name_distinction="Keywords-100")
    newSVM = SVM(vector_path="newdata/spaces/Keywords-Layer-200-L.mds", name_distinction="Keywords-200")
    newSVM = SVM(vector_path="newdata/spaces/Genres-Layer-20-L.mds", name_distinction="Keywords-20")
    newSVM = SVM(vector_path="newdata/spaces/Genres-Layer-100-L.mds", name_distinction="Keywords-100")
    newSVM = SVM(vector_path="newdata/spaces/Genres-Layer-200-L.mds", name_distinction="Keywords-200")


if  __name__ =='__main__':main()
