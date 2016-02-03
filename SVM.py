
from sklearn import svm, linear_model
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from skll.metrics import kappa
import DataTasks as dt
import MovieTasks as mt
from scipy import stats, linalg, spatial
import SVMTasks


class SVM:


    def __init__(self, class_type="Phrases", name_distinction="", class_names=None, vector_path=None, class_by_class=True, input_size=200,
                 training_data=10000, amount_of_scores=400, low_kappa=0.1, high_kappa=0.5, rankSVM=False, missing_samples=""):

        movie_names, movie_vectors, movie_labels = mt.getMovieData(class_type=class_type, input_size=input_size,
                                                                   class_names=class_names, class_by_class=class_by_class, vector_path=vector_path)

        file_names = dt.getAllFileNamesAndExtensions("filmdata\classes" + class_type)
        missing_samples_set = range(15000)
        if missing_samples != "":
            missing_samples = dt.importString(missing_samples)
            for m in missing_samples:
                missing_samples_set[int(m)] = -1
                print movie_names[int(m)]
        else:
            missing_samples_set = None

        file_names, n_train, x_train, y_train, n_test, x_test, y_test = SVMTasks.getSampledData(file_names, movie_names, movie_vectors, movie_labels, training_data, missing_samples_set)

        movie_names = None
        movie_vectors = None
        movie_labels = None

        kappa_scores, top_keywords, top_kappas, top_directions, high_keywords, ultra_low_keywords, low_keywords, \
        high_directions, low_directions, overall_kappa, overall_accuracy, overall_f1, directions = \
        self.runAllSVMs(len(y_train), amount_of_scores, y_test, y_train, x_train, x_test, class_type, input_size, file_names, low_kappa, high_kappa, rankSVM)

        row = [overall_kappa, overall_accuracy, overall_f1]

        print row
        print high_keywords
        print low_keywords
        if class_type == "NewKeywords":
            dt.write1dArray(top_keywords, "filmdata/KeywordData/SVM/allTOP_SCORES_" + str(amount_of_scores) + "_" + class_type+"_"+name_distinction+".txt")
            dt.write1dArray(top_kappas, "filmdata/KeywordData/SVM/allTOP_KAPPAS_" + str(amount_of_scores) + "_"+ class_type+"_"+name_distinction+".txt")
            dt.write1dArray(ultra_low_keywords, "filmdata/KeywordData/SVM/allultra_low_keywords_" + str(amount_of_scores) + "_"+ class_type+"_"+name_distinction+".txt")
            dt.write1dArray(high_keywords, "filmdata/KeywordData/SVM/allhigh_keywords_"+ class_type+"_"+name_distinction+".txt")
            dt.write1dArray(low_keywords, "filmdata/KeywordData/SVM/alllow_keywords_"+ class_type+"_"+name_distinction+".txt")
            dt.write1dArray(kappa_scores, "filmdata/KeywordData/SVM/ALL_SCORES_"+ class_type+"_"+name_distinction+".txt")
            dt.write1dArray(file_names, "filmdata/KeywordData/SVM/ALL_NAMES_"+ class_type+"_"+name_distinction+".txt")
            for d in range(len(high_directions)):
                dt.write1dArray(high_directions[d], "filmdata/KeywordData/new_directions/high_directions/"+name_distinction+"/"+ high_keywords[d]+".txt")
            for d in range(len(low_directions)):
                dt.write1dArray(low_directions[d], "filmdata/KeywordData/new_directions/low_directions/"+name_distinction+"/"+ low_keywords[d]+".txt")
            for d in range(len(top_directions)):
                dt.write1dArray(top_directions[d], "filmdata/KeywordData/new_directions/top_directions/"+name_distinction+"/"+ top_keywords[d]+".txt")
        else:
            dt.write1dArray(top_keywords, "filmdata/SVM/allTOP_SCORES_" + str(amount_of_scores) + "_" + class_type+"_"+name_distinction+".txt")
            dt.write1dArray(top_kappas, "filmdata/SVM/allTOP_KAPPAS_" + str(amount_of_scores) + "_"+ class_type+"_"+name_distinction+".txt")
            dt.write1dArray(ultra_low_keywords, "filmdata/SVM/allultra_low_keywords_" + str(amount_of_scores) + "_"+ class_type+"_"+name_distinction+".txt")
            dt.write1dArray(high_keywords, "filmdata/SVM/allhigh_keywords_"+ class_type+"_"+name_distinction+".txt")
            dt.write1dArray(low_keywords, "filmdata/SVM/alllow_keywords_"+ class_type+"_"+name_distinction+".txt")
            dt.write1dArray(kappa_scores, "filmdata/SVM/ALL_SCORES_"+ class_type+"_"+name_distinction+".txt")
            dt.write1dArray(file_names, "filmdata/SVM/ALL_NAMES_"+ class_type+"_"+name_distinction+".txt")
            for d in range(len(high_directions)):
                dt.write1dArray(high_directions[d], "filmdata/new_directions/high_directions/"+name_distinction+"/"+ high_keywords[d]+".txt")
            for d in range(len(low_directions)):
                dt.write1dArray(low_directions[d], "filmdata/new_directions/low_directions/"+name_distinction+"/"+ low_keywords[d]+".txt")
            for d in range(len(top_directions)):
                dt.write1dArray(top_directions[d], "filmdata/new_directions/top_directions/"+name_distinction+"/"+ top_keywords[d]+".txt")


        dt.writeToSheet(row, "filmdata/experiments/experimenttest.csv")

        print "Kappa:", overall_kappa, "Accuracy:", overall_accuracy, "F1", overall_f1

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
        try:
            clf.fit(x_train[keyword], y_train[keyword])
        except ValueError:
            print keyword, "FAILED"
            return 0, 0, 0, None
        #clf.decision_function(x_test)
        direction = clf.coef_
        y_pred = clf.predict(x_test)
        y_pred = y_pred.tolist()
        kappa_score = kappa(y_test[keyword], y_pred)
        #accuracy = accuracy_score(y_test[keyword], y_pred)
        #f1 = f1_score(y_test[keyword], y_pred, average='macro')
        print keyword, "SUCCESS"

        return kappa_score, 1, 1, direction

    def runAllSVMs(self, keyword_amount, amount_of_scores, y_test, y_train, x_train, x_test, class_type, input_size, file_names, low_kappa, high_kappa, rankSVM):
        totals = np.zeros(3)
        kappa_scores = []
        accuracy_scores = []
        f1_scores = []

        directions = []

        # For every keyword
        for x in range(keyword_amount-1):
            if x_train[x] is not None:
                if rankSVM:
                    direction = self.runRankSVM(y_test, y_train, x_train, x_test, class_type, input_size, file_names, x)
                    directions.append(direction)
                else:
                    # Run an SVM for that keyword that matches each movie vector to its respective label
                    kappa, accuracy, f1, direction = self.runSVM(y_test, y_train, x_train, x_test, class_type, input_size, file_names, x)
                    kappa_scores.append(kappa)
                    print kappa, len(x_train[x]), len(y_train[x])
                    accuracy_scores.append(accuracy)
                    f1_scores.append(f1)
                    directions.append(direction)

        top_scores = np.argpartition(kappa_scores, amount_of_scores)[-amount_of_scores:]
        top_keywords = []
        top_kappas = []
        top_directions = []
        for i in top_scores:
            top_keywords.append(file_names[i])
            top_kappas.append(kappa_scores[i])
            top_directions.append(directions[i])

        high_kappas = np.where(np.diff(kappa_scores) > 0.5)[0] + 1
        low_kappas = np.where(np.diff(kappa_scores) > 0.1)[0] + 1
        ultra_low_kappas = np.where(np.diff(kappa_scores) < 0.1)[0] + 1

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
            high_directions.append(directions[val])

        for val in low_kappas:
            low_keywords.append(file_names[val])
            low_directions.append(directions[val])

        ultra_low_keywords = []
        for val in ultra_low_kappas:
            ultra_low_keywords.append(file_names[val])

        """
        overall_kappa = totals[0] / keyword_amount
        overall_accuracy = totals[1] / keyword_amount
        overall_f1 = totals[2] / keyword_amount
        """

        return kappa_scores, top_keywords, top_kappas, top_directions, high_keywords, ultra_low_keywords, low_keywords, high_directions, low_directions, 1, 1, 1, directions

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
    newSVM = SVM(input_size=20, name_distinction="Normal-20")
    newSVM = SVM(input_size=50,  name_distinction="Normal-50")
    newSVM = SVM(input_size=100, name_distinction="Normal-100")
    newSVM = SVM(vector_path="newdata/spaces/All-Layer-20-L.mds", name_distinction="All-20")
    newSVM = SVM(vector_path="newdata/spaces/All-Layer-50-L.mds",  name_distinction="All-50")
    newSVM = SVM(vector_path="newdata/spaces/All-Layer-100-L.mds",  name_distinction="All-100")
    newSVM = SVM(vector_path="newdata/spaces/All-Layer-20-20-L.mds",  name_distinction="All-20-20")
    newSVM = SVM(vector_path="newdata/spaces/All-Layer-50-50-L.mds",  name_distinction="All-50-50")
    newSVM = SVM(vector_path="newdata/spaces/All-Layer-100-100-L.mds",  name_distinction="All-100-100")

    """

    #newSVM = SVM(vector_path="newdata/spaces/Genres-Layer-200-400-L.mds",  name_distinction="Genres-200-400")
    #newSVM = SVM(vector_path="newdata/spaces/All-200-20, 200x2.mds",
    #             name_distinction="All-200-20, 200x2")

    newSVM = SVM(vector_path="newdata/spaces/All-20-200, 20x2.mds",
                 name_distinction="All-All-20-200, 20x2")

    newSVM = SVM(vector_path="newdata/spaces/All-100,20,200x3.mds",
                 name_distinction="All-100,20,200x3")



    """
    newSVM = SVM(vector_path="newdata/spaces/Keywords-Layer-200-400-L.mds",  name_distinction="Keywords-200-400")

    newSVM = SVM(vector_path="newdata/spaces/Keywords-Layer-20-L.mds",  name_distinction="Keywords-20")
    newSVM = SVM(vector_path="newdata/spaces/Keywords-Layer-50-L.mds",  name_distinction="Keywords-50")
    newSVM = SVM(vector_path="newdata/spaces/Keywords-Layer-100-L.mds",  name_distinction="Keywords-100")
    newSVM = SVM(vector_path="newdata/spaces/Keywords-Layer-20-20-L.mds",  name_distinction="Keywords-20-20")
    newSVM = SVM(vector_path="newdata/spaces/Keywords-Layer-50-50-L.mds",  name_distinction="Keywords-50-50")
    newSVM = SVM(vector_path="newdata/spaces/Keywords-Layer-100-100-L.mds",  name_distinction="Keywords-100-100")
    """
    """
    newSVM = SVM(vector_path="newdata/spaces/Genres-Layer-20-L.mds",  name_distinction="Genres-20")
    newSVM = SVM(vector_path="newdata/spaces/Genres-Layer-50-L.mds",  name_distinction="Genres-50")
    newSVM = SVM(vector_path="newdata/spaces/Genres-Layer-100-L.mds",  name_distinction="Genres-100")
    newSVM = SVM(vector_path="newdata/spaces/Genres-Layer-20-20-L.mds",  name_distinction="Genres-20-20")
    newSVM = SVM(vector_path="newdata/spaces/Genres-Layer-50-50-L.mds",  name_distinction="Genres-50-50")
    newSVM = SVM(vector_path="newdata/spaces/Genres-Layer-100-100-L.mds",  name_distinction="Genres-100-100")

     """


    """
    newSVM = SVM(input_size=20, class_type="NewKeywords", name_distinction="Normal-20", missing_samples="filmdata/MISSING_KEYWORD_ITEMS.txt")
    newSVM = SVM(input_size=50, class_type="NewKeywords", name_distinction="Normal-50", missing_samples="filmdata/MISSING_KEYWORD_ITEMS.txt")
    newSVM = SVM(input_size=100, class_type="NewKeywords", name_distinction="Normal-100", missing_samples="filmdata/MISSING_KEYWORD_ITEMS.txt")
    newSVM = SVM(input_size=200, class_type="NewKeywords", name_distinction="Normal-200", missing_samples="filmdata/MISSING_KEYWORD_ITEMS.txt")
    newSVM = SVM(vector_path="newdata/spaces/All-Layer-20-L.mds", class_type="NewKeywords", name_distinction="All-20", missing_samples="filmdata/MISSING_KEYWORD_ITEMS.txt")
    newSVM = SVM(vector_path="newdata/spaces/All-Layer-50-L.mds", class_type="NewKeywords", name_distinction="All-50", missing_samples="filmdata/MISSING_KEYWORD_ITEMS.txt")
    newSVM = SVM(vector_path="newdata/spaces/All-Layer-100-L.mds", class_type="NewKeywords", name_distinction="All-100", missing_samples="filmdata/MISSING_KEYWORD_ITEMS.txt")
    newSVM = SVM(vector_path="newdata/spaces/All-Layer-200-L.mds", class_type="NewKeywords", name_distinction="All-200", missing_samples="filmdata/MISSING_KEYWORD_ITEMS.txt")
    newSVM = SVM(vector_path="newdata/spaces/All-Layer-20-20-L.mds", class_type="NewKeywords", name_distinction="All-20-20", missing_samples="filmdata/MISSING_KEYWORD_ITEMS.txt")
    newSVM = SVM(vector_path="newdata/spaces/All-Layer-50-50-L.mds", class_type="NewKeywords", name_distinction="All-50-50", missing_samples="filmdata/MISSING_KEYWORD_ITEMS.txt")
    newSVM = SVM(vector_path="newdata/spaces/All-Layer-100-100-L.mds", class_type="NewKeywords", name_distinction="All-100-100", missing_samples="filmdata/MISSING_KEYWORD_ITEMS.txt")
    newSVM = SVM(vector_path="newdata/spaces/Keywords-Layer-20-L.mds", class_type="NewKeywords", name_distinction="Keywords-20", missing_samples="filmdata/MISSING_KEYWORD_ITEMS.txt")
    newSVM = SVM(vector_path="newdata/spaces/Keywords-Layer-50-L.mds", class_type="NewKeywords", name_distinction="Keywords-50", missing_samples="filmdata/MISSING_KEYWORD_ITEMS.txt")
    newSVM = SVM(vector_path="newdata/spaces/Keywords-Layer-100-L.mds", class_type="NewKeywords", name_distinction="Keywords-100", missing_samples="filmdata/MISSING_KEYWORD_ITEMS.txt")
    newSVM = SVM(vector_path="newdata/spaces/Keywords-Layer-200-L.mds", class_type="NewKeywords", name_distinction="Keywords-200", missing_samples="filmdata/MISSING_KEYWORD_ITEMS.txt")
    newSVM = SVM(vector_path="newdata/spaces/Keywords-Layer-20-20-L.mds", class_type="NewKeywords", name_distinction="Keywords-20-20", missing_samples="filmdata/MISSING_KEYWORD_ITEMS.txt")
    newSVM = SVM(vector_path="newdata/spaces/Keywords-Layer-50-50-L.mds", class_type="NewKeywords", name_distinction="Keywords-50-50", missing_samples="filmdata/MISSING_KEYWORD_ITEMS.txt")
    newSVM = SVM(vector_path="newdata/spaces/Keywords-Layer-100-100-L.mds", class_type="NewKeywords", name_distinction="Keywords-100-100", missing_samples="filmdata/MISSING_KEYWORD_ITEMS.txt")
    newSVM = SVM(vector_path="newdata/spaces/Genres-Layer-20-L.mds", class_type="NewKeywords", name_distinction="Genres-20", missing_samples="filmdata/MISSING_KEYWORD_ITEMS.txt")
    newSVM = SVM(vector_path="newdata/spaces/Genres-Layer-50-L.mds", class_type="NewKeywords", name_distinction="Genres-50", missing_samples="filmdata/MISSING_KEYWORD_ITEMS.txt")
    newSVM = SVM(vector_path="newdata/spaces/Genres-Layer-100-L.mds", class_type="NewKeywords", name_distinction="Genres-100", missing_samples="filmdata/MISSING_KEYWORD_ITEMS.txt")
    newSVM = SVM(vector_path="newdata/spaces/Genres-Layer-200-L.mds", class_type="NewKeywords", name_distinction="Genres-200", missing_samples="filmdata/MISSING_KEYWORD_ITEMS.txt")
    newSVM = SVM(vector_path="newdata/spaces/Genres-Layer-20-20-L.mds", class_type="NewKeywords", name_distinction="Genres-20-20", missing_samples="filmdata/MISSING_KEYWORD_ITEMS.txt")
    newSVM = SVM(vector_path="newdata/spaces/Genres-Layer-50-50-L.mds", class_type="NewKeywords", name_distinction="Genres-50-50", missing_samples="filmdata/MISSING_KEYWORD_ITEMS.txt")
    newSVM = SVM(vector_path="newdata/spaces/Genres-Layer-100-100-L.mds", class_type="NewKeywords", name_distinction="Genres-100-100", missing_samples="filmdata/MISSING_KEYWORD_ITEMS.txt")
    """
    """
    # RE-doing these to find the differences between them

    newSVM = SVM(input_size=200,  name_distinction="Normal-200")
    newSVM = SVM(vector_path="newdata/spaces/All-Layer-200-L.mds", name_distinction="All-200")

    newSVM = SVM(vector_path="newdata/spaces/Keywords-Layer-200-L.mds", name_distinction="Keywords-200")
    newSVM = SVM(vector_path="newdata/spaces/Genres-Layer-200-L.mds", name_distinction="Genres-200")

    """
    """
    newSVM = SVM(input_size=20, class_type="Phrases/nonbinary", name_distinction="NBNormal-20")
    newSVM = SVM(input_size=50, class_type="Phrases/nonbinary", name_distinction="NBNormal-50")
    newSVM = SVM(input_size=100, class_type="Phrases/nonbinary", name_distinction="NBNormal-100")
    newSVM = SVM(input_size=200, class_type="Phrases/nonbinary", name_distinction="NBNormal-200")
    newSVM = SVM(vector_path="newdata/spaces/All-Layer-20-L.mds", class_type="Phrases/nonbinary", name_distinction="NBAll-20")
    newSVM = SVM(vector_path="newdata/spaces/All-Layer-50-L.mds", class_type="Phrases/nonbinary", name_distinction="NBAll-50")
    newSVM = SVM(vector_path="newdata/spaces/All-Layer-100-L.mds", class_type="Phrases/nonbinary", name_distinction="NBAll-100")
    newSVM = SVM(vector_path="newdata/spaces/All-Layer-200-L.mds", class_type="Phrases/nonbinary", name_distinction="NBAll-200")
    newSVM = SVM(vector_path="newdata/spaces/Keywords-Layer-20-L.mds", class_type="Phrases/nonbinary", name_distinction="NBKeywords-20")
    newSVM = SVM(vector_path="newdata/spaces/Keywords-Layer-50-L.mds", class_type="Phrases/nonbinary", name_distinction="NBKeywords-50")
    newSVM = SVM(vector_path="newdata/spaces/Keywords-Layer-100-L.mds", class_type="Phrases/nonbinary", name_distinction="NBKeywords-100")
    newSVM = SVM(vector_path="newdata/spaces/Keywords-Layer-200-L.mds", class_type="Phrases/nonbinary", name_distinction="NBKeywords-200")
    newSVM = SVM(vector_path="newdata/spaces/Genres-Layer-20-L.mds", class_type="Phrases/nonbinary", name_distinction="NBGenres-20")
    newSVM = SVM(vector_path="newdata/spaces/Genres-Layer-50-L.mds", class_type="Phrases/nonbinary", name_distinction="NBGenres-50")
    newSVM = SVM(vector_path="newdata/spaces/Genres-Layer-100-L.mds", class_type="Phrases/nonbinary", name_distinction="NBGenres-100")
    newSVM = SVM(vector_path="newdata/spaces/Genres-Layer-200-L.mds", class_type="Phrases/nonbinary", name_distinction="NBGenres-200")
    """


if  __name__ =='__main__':main()
