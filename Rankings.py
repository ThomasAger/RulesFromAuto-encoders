import numpy as np
import DataTasks as dt
from collections import OrderedDict
class Rankings:
    def __init__(self, directions_fn, vectors_fn, cluster_names_fn, vector_names_fn, fn, percent, percentage_increment, by_vector):

        directions = dt.importVectors(directions_fn)
        vectors = dt.importVectors(vectors_fn)
        cluster_names = dt.importString(cluster_names_fn)
        vector_names = dt.importString(vector_names_fn)

        rankings  = self.getRankings(directions, vectors, cluster_names, vector_names)
        rankings = np.array(rankings)
        #labels = self.createLabels(rankings, percent)
        #labels = np.asarray(labels)
        discrete_labels = self.createDiscreteLabels(rankings, percentage_increment)
        discrete_labels = np.asarray(discrete_labels)
        if by_vector:
            #labels = labels.transpose()
            discrete_labels = discrete_labels.transpose()
            rankings = rankings.transpose()
        #dt.write2dArray(labels, "Rankings/" + fn + "P" + str(percent) +".labels")
        dt.write2dArray(rankings, "Rankings/" + fn + ".space")
        dt.write2dArray(discrete_labels, "Rankings/" + fn + "P" + str(percentage_increment) + ".discrete")
        array = []
        short_array = []
        """ Disabled names for quick view now
        for key, value in ranking_names.iteritems():
            array.append([key, value])
            short_array.append([key, value[:25]])
        dt.write1dArray(array, "Rankings/" + fn + ".names")
        dt.write1dArray(short_array, "Rankings/Short" + fn + ".names")
        """

    # Collect the rankings of movies for the given cluster directions
    def getRankings(self, cluster_directions, vectors, cluster_names, vector_names):
        rankings = []
        #ranking_names = OrderedDict()
        for d in range(len(cluster_directions)):
            cluster_ranking = []
            #cluster_ranking_names = []
            for v in range(len(vectors)):
                cluster_ranking.append(np.dot(cluster_directions[d], vectors[v]))
                #cluster_ranking_names.append(vector_names[v])
            #cluster_ranking_names = dt.sortByArray(cluster_ranking_names, cluster_ranking)
            #ranking_names[cluster_names[d]] = cluster_ranking_names
            rankings.append(cluster_ranking)
            print "Cluster:", cluster_names[d]#, "Movies:", cluster_ranking_names[0], cluster_ranking[0],\
                #cluster_ranking_names[1], cluster_ranking[1], cluster_ranking_names[2], cluster_ranking[2]
        return rankings#, ranking_names

    # Create binary vectors for the top % of the rankings, 1 for if it is in that percent and 0 if not.
    def createLabels(self, rankings, percent):
        np_rankings = np.asarray(rankings)
        labels = []
        for r in np_rankings:
            label = [0 for x in range(len(rankings[0]))]
            sorted_indices = r.argsort()
            top_indices = sorted_indices[:len(rankings[0]) * percent]
            for t in top_indices:
                label[t] = 1
            labels.append(label)
        return labels

    def createDiscreteLabels(self, rankings, percentage_increment):
        np_rankings = np.asarray(rankings)
        labels = []
        for r in np_rankings:
            label = ["100%" for x in range(len(rankings[0]))]
            sorted_indices = r.argsort()[::-1]
            for i in range(0, 100, percentage_increment):
                top_indices = sorted_indices[len(rankings[0]) * (i*0.01):len(rankings[0]) * ((i+percentage_increment) * 0.01)]
                for t in top_indices:
                    label[t] = str(i+percentage_increment) + "%"
            labels.append(label)
        return labels


def main(low_threshold, high_threshold, percent, discrete_percent, cluster_fn, vector_fn, cluster_names_fn, vector_names_fn, rank_fn, by_vector):
    Rankings(cluster_fn, vector_fn, cluster_names_fn, vector_names_fn, rank_fn, percent, discrete_percent, by_vector)

filename = "films100"
if  __name__ =='__main__':main(0.13, 0.46,  0.02, 1,
"Directions/" + filename + ".directions",
"filmdata/films100.mds/" + filename +".mds",
"SVMResults/" + filename +  ".names",
"filmdata/filmNames.txt", filename, False)