
from sklearn import svm, linear_model
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from skll.metrics import kappa
import DataTasks as dt
import MovieTasks as mt
from scipy import stats, linalg, spatial


class SVM:


    def __init__(self, class_type="Phrases", name_distinction="", class_names=None, vector_path=None, class_by_class=True, input_size=200,
                 training_data=10000, amount_of_scores=400, low_kappa=0.1, high_kappa=0.5, rankSVM=False):

        movie_names, movie_vectors, movie_labels = mt.getMovieData(class_type=class_type, input_size=input_size, class_names=class_names,
                                                                   class_by_class=class_by_class, vector_path=vector_path)


        file_names = dt.getAllFileNames("filmdata\classes" + class_type)

        n_train, x_train, y_train, n_test, x_test, y_test = self.getSampledData(movie_names, movie_vectors, movie_labels, training_data)

        movie_names = None
        movie_vectors = None
        movie_labels = None

        top_keywords, top_kappas, high_keywords, ultra_low_keywords, low_keywords, high_directions, low_directions, overall_kappa, overall_accuracy, overall_f1, directions = self.runAllSVMs(len(y_train), amount_of_scores,
                                                y_test, y_train, x_train, x_test, class_type, input_size, file_names, low_kappa, high_kappa, rankSVM)

        #direction_ranks_dict = self.rankByDirections(movie_names, movie_vectors, file_names, high_directions)

        #print direction_ranks_dict

        row = [overall_kappa, overall_accuracy, overall_f1]

        print row
        print high_keywords
        print low_keywords
        dt.write1dArray(top_keywords, "filmdata/allTOP_SCORES_" + str(amount_of_scores) + "_" + class_type+"_"+name_distinction+".txt")
        dt.write1dArray(top_kappas, "filmdata/allTOP_KAPPAS_" + str(amount_of_scores) + "_"+ class_type+"_"+name_distinction+".txt")
        dt.write1dArray(ultra_low_keywords, "filmdata/allultra_low_keywords_" + str(amount_of_scores) + "_"+ class_type+"_"+name_distinction+".txt")

        dt.write1dArray(high_keywords, "filmdata/allhigh_keywords_"+ class_type+"_"+name_distinction+".txt")
        dt.write1dArray(low_keywords, "filmdata/alllow_keywords_"+ class_type+"_"+name_distinction+".txt")

        dt.write1dArray(high_directions, "filmdata/allhigh_"+str(amount_of_scores)+"_" +"LowPhrases"+"_" + str(input_size) + "_Round2.txt")
        dt.writeToSheet(row, "filmdata/experiments/experimenttest.csv")

        print "Kappa:", overall_kappa, "Accuracy:", overall_accuracy, "F1", overall_f1


    """

    Sample the data such that there is a unique amount of sampled movies for each
    phrase, according to how often that phrase occurs.

    """

    def getSampledData(self, movie_names, movie_vectors, movie_labels, training_data):
        print len(movie_vectors), len(movie_labels)
        n_train = movie_names[:training_data]
        x_train = []
        for x in range(len(movie_labels)):
            x_train.append(movie_vectors[:training_data])

        n_test = movie_names[training_data:]
        x_test = movie_vectors[training_data:]

        y_train = []
        y_test = []
        for ml in movie_labels:
            y_train.append(ml[:training_data])
            y_test.append(ml[training_data:])
        print len(x_train), len(x_train[0]), len(x_train[0][0])
        print len(y_train), len(y_train[0])
        for yt in range(len(y_train)):
            y1 = 0
            y0 = 0
            for y in range(len(y_train[yt])):
                if y_train[yt][y] == 1:
                    y1 += 1
                if y_train[yt][y] == 0:
                    y0 += 1
            y = 0
            while(y0 > int(y1*2)):
                if y_train[yt][y] == 0:
                    del x_train[yt][y]
                    del y_train[yt][y]
                    y0 -= 1
                else:
                    y += 1
            print yt, "len(0):", y0, "len(1):", y1

        return n_train, x_train, y_train, n_test, x_test, y_test


    def getSimilarity(self, vector1, vector2):
        return 1 - spatial.distance.cosine(vector1, vector2)

    def getMostSimilarTerms(self, high_values, low_values):
        mapping_to_most_similar = []
        for l in range(len(low_values)):
            highest_value = 0
            most_similar = 0
            for h in range(len(high_values)):
                similarity = self.getSimilarity(high_values[h], low_values[l])
                if similarity > highest_value:
                    highest_value = similarity
                    most_similar = h
            mapping_to_most_similar.append(most_similar)
        return mapping_to_most_similar

    def createClusteredPlane(self, high_value_plane, low_value_planes):
        np_high_plane = np.asarray(high_value_plane)
        np_low_plane = []
        for lvp in low_value_planes:
            np_low_plane.append(np.asarray(np_low_plane))
        total_low_plane = []
        for lp in np_low_plane:
            total_low_plane = total_low_plane + lp
        total_plane = total_low_plane + np_high_plane
        clustered_plane = total_plane / len(np_low_plane) + 1
        return clustered_plane

    """

    Given an array of high value planes, and a matching array of arrays of similar low matching planes
    Return the clusters of the combined high value planes with their array of similar low value planes

    """

    def createClusteredPlanes(self, high_value_planes, low_value_planes):
        clustered_planes = []
        for h in high_value_planes:
            for l in low_value_planes:
                clustered_planes.append(self.createClusteredPlane(h, l))
        return clustered_planes
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

        # For every keyword
        for x in range(keyword_amount-1):
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

        top_scores = np.argpartition(kappa_scores, -amount_of_scores)[-amount_of_scores:]
        top_keywords = []
        top_kappas = []
        for i in top_scores:
            top_keywords.append(file_names[i])
            top_kappas.append(kappa_scores[i])

        print kappa_scores
        high_kappas = np.where(np.diff(kappa_scores) > 0.5)[0] + 1
        low_kappas = np.where(np.diff(kappa_scores) > 0.1)[0] + 1
        ultra_low_kappas = np.where(np.diff(kappa_scores) < 0.1)[0] + 1
        print high_kappas
        print low_kappas
        print ultra_low_kappas

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


        overall_kappa = totals[0] / keyword_amount
        overall_accuracy = totals[1] / keyword_amount
        overall_f1 = totals[2] / keyword_amount




        return top_keywords, top_kappas, high_keywords, ultra_low_keywords, low_keywords, high_directions, low_directions, overall_kappa, overall_accuracy, overall_f1, directions

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

    newSVM = SVM(input_size=20, name_distinction="Normal-20")
    newSVM = SVM(input_size=50, name_distinction="Normal-50")
    newSVM = SVM(input_size=100, name_distinction="Normal-100")
    newSVM = SVM(input_size=200, name_distinction="Normal-100")
    newSVM = SVM(input_size=20, class_type="Phrases/nonbinary", name_distinction="NBNormal-20")
    newSVM = SVM(input_size=50, class_type="Phrases/nonbinary", name_distinction="NBNormal-50")
    newSVM = SVM(input_size=100, class_type="Phrases/nonbinary", name_distinction="NBNormal-100")
    newSVM = SVM(input_size=200, class_type="Phrases/nonbinary", name_distinction="NBNormal-100")

    newSVM = SVM(vector_path="newdata/spaces/All-Layer-20-L.mds", name_distinction="All-20")
    newSVM = SVM(vector_path="newdata/spaces/All-Layer-50-L.mds", name_distinction="All-50")
    newSVM = SVM(vector_path="newdata/spaces/All-Layer-100-L.mds", name_distinction="All-100")
    newSVM = SVM(vector_path="newdata/spaces/All-Layer-200-L.mds", name_distinction="All-200")
    newSVM = SVM(vector_path="newdata/spaces/Keywords-Layer-20-L.mds", name_distinction="Keywords-20")
    newSVM = SVM(vector_path="newdata/spaces/Keywords-Layer-50-L.mds", name_distinction="Keywords-50")
    newSVM = SVM(vector_path="newdata/spaces/Keywords-Layer-100-L.mds", name_distinction="Keywords-100")
    newSVM = SVM(vector_path="newdata/spaces/Keywords-Layer-200-L.mds", name_distinction="Keywords-200")
    newSVM = SVM(vector_path="newdata/spaces/Genres-Layer-20-L.mds", name_distinction="Keywords-20")
    newSVM = SVM(vector_path="newdata/spaces/Genres-Layer-50-L.mds", name_distinction="Keywords-50")
    newSVM = SVM(vector_path="newdata/spaces/Genres-Layer-100-L.mds", name_distinction="Keywords-100")
    newSVM = SVM(vector_path="newdata/spaces/Genres-Layer-200-L.mds", name_distinction="Keywords-200")

    """
    Nonbinary!
    """

    newSVM = SVM(vector_path="newdata/spaces/All-Layer-20-L.mds", class_type="Phrases/nonbinary", name_distinction="NBAll-20")
    newSVM = SVM(vector_path="newdata/spaces/All-Layer-50-L.mds", class_type="Phrases/nonbinary", name_distinction="NBAll-50")
    newSVM = SVM(vector_path="newdata/spaces/All-Layer-100-L.mds", class_type="Phrases/nonbinary", name_distinction="NBAll-100")
    newSVM = SVM(vector_path="newdata/spaces/All-Layer-200-L.mds", class_type="Phrases/nonbinary", name_distinction="NBAll-200")
    newSVM = SVM(vector_path="newdata/spaces/Keywords-Layer-20-L.mds", class_type="Phrases/nonbinary", name_distinction="NBKeywords-20")
    newSVM = SVM(vector_path="newdata/spaces/Keywords-Layer-50-L.mds", class_type="Phrases/nonbinary", name_distinction="NBKeywords-50")
    newSVM = SVM(vector_path="newdata/spaces/Keywords-Layer-100-L.mds", class_type="Phrases/nonbinary", name_distinction="NBKeywords-100")
    newSVM = SVM(vector_path="newdata/spaces/Keywords-Layer-200-L.mds", class_type="Phrases/nonbinary", name_distinction="NBKeywords-200")
    newSVM = SVM(vector_path="newdata/spaces/Genres-Layer-20-L.mds", class_type="Phrases/nonbinary", name_distinction="NBKeywords-20")
    newSVM = SVM(vector_path="newdata/spaces/Genres-Layer-50-L.mds", class_type="Phrases/nonbinary", name_distinction="NBKeywords-50")
    newSVM = SVM(vector_path="newdata/spaces/Genres-Layer-100-L.mds", class_type="Phrases/nonbinary", name_distinction="NBKeywords-100")
    newSVM = SVM(vector_path="newdata/spaces/Genres-Layer-200-L.mds", class_type="Phrases/nonbinary", name_distinction="NBKeywords-200")


if  __name__ =='__main__':main()
