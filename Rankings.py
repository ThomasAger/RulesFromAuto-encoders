import numpy as np
import DataTasks as dt
from collections import OrderedDict
class Rankings:
    def __init__(self, directions_fn, vectors_fn, cluster_names_fn, vector_names_fn, fn, percents, percentage_increment):

        directions = dt.importVectors(directions_fn)
        vectors = dt.importVectors(vectors_fn)
        cluster_names = dt.importString(cluster_names_fn)
        vector_names = dt.importString(vector_names_fn)

        rankings, ranking_names = self.getRankings(directions, vectors, cluster_names, vector_names)
        rankings = np.array(rankings)
        for percent in percents:
            labels = self.createLabels(rankings, percent)
            labels = np.asarray(labels)
            labels = labels.transpose()
            dt.write2dArray(labels, "Rankings/" + fn + "P" + str(percent) +".labels")
        discrete_labels = self.createDiscreteLabels(rankings, percentage_increment)
        print discrete_labels[0]
        discrete_labels = np.asarray(discrete_labels)
        discrete_labels = discrete_labels.transpose()
        rankings = rankings.transpose()

        dt.write2dArray(rankings, "Rankings/" + fn + ".space")
        dt.write2dArray(discrete_labels, "Rankings/" + fn + "P" + str(percentage_increment) + ".discrete")
        array = []
        short_array = []
        for key, value in ranking_names.iteritems():
            array.append([key, value])
            short_array.append([key, value[:25]])
        dt.write1dArray(array, "Rankings/" + fn + ".names")
        dt.write1dArray(short_array, "Rankings/Short" + fn + ".names")

    # Collect the rankings of movies for the given cluster directions
    def getRankings(self, cluster_directions, vectors, cluster_names, vector_names):
        rankings = []
        ranking_names = OrderedDict()
        for d in range(len(cluster_directions)):
            cluster_ranking = []
            cluster_ranking_names = []
            for v in range(len(vectors)):
                cluster_ranking.append(np.dot(cluster_directions[d], vectors[v]))
                cluster_ranking_names.append(vector_names[v])
            cluster_ranking_names = dt.sortByArray(cluster_ranking_names, cluster_ranking)
            ranking_names[cluster_names[d]] = cluster_ranking_names
            rankings.append(cluster_ranking)
            print "Cluster:", cluster_names[d], "Movies:", cluster_ranking_names[0], cluster_ranking[0],\
                cluster_ranking_names[1], cluster_ranking[1], cluster_ranking_names[2], cluster_ranking[2]
        return rankings, ranking_names

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
            sorted_indices = r.argsort()
            for i in range(0, 100, percentage_increment):
                print i, len(rankings[0]) * (i*0.01), len(rankings[0]) * ((i+percentage_increment) * 0.01)
                top_indices = sorted_indices[len(rankings[0]) * (i*0.01):len(rankings[0]) * ((i+percentage_increment) * 0.01)]
                for t in top_indices:
                    label[t] = str(i+percentage_increment) + "%"
            labels.append(label)
        return labels


def main():
    percent = [0.1, 0.05, 0.02, 0.01]
    percent_increment = 1

    """
    low_threshold = 0.1
    high_threshold = 0.44
    filename = "films200"
    cluster_fn = "Clusters/" + filename + "Cut2000"+  "LeastSimilarHIGH"+str(high_threshold)+","+str(low_threshold)+".clusters"
    movie_vector_fn = "filmdata/films200.mds/" + filename +".mds"
    cluster_names = "Clusters/" + filename + "Cut2000"+  "LeastSimilarHIGH"+str(high_threshold)+","+str(low_threshold)+".names"
    movie_names = "filmdata/filmNames.txt"
    rank_fn = filename + "P" + str(percent)
    print filename
    Rankings(cluster_fn, movie_vector_fn, cluster_names, movie_names, rank_fn, percent, percent_increment)
    """
    low_threshold = 0.1
    high_threshold = 0.418
    filename = "AUTOENCODERN0.6R0.0tanhtanhmse1tanh2004SDA1"
    cluster_fn = "Clusters/" + filename + "Cut2004" + "LeastSimilarHIGH"+str(high_threshold)+","+str(low_threshold)+".clusters"
    movie_vector_fn = "newdata/spaces/" + filename + ".mds"
    cluster_names = "Clusters/" + filename + "Cut2004" + "LeastSimilarHIGH"+str(high_threshold)+","+str(low_threshold)+".names"
    movie_names = "filmdata/filmNames.txt"
    rank_fn = filename
    print filename
    Rankings(cluster_fn, movie_vector_fn, cluster_names, movie_names, rank_fn, percent, percent_increment)

    low_threshold =  0.055
    high_threshold = 0.34
    filename = "AUTOENCODERN0.6R0.0tanhtanhmse1tanh1004SDA2"
    cluster_fn = "Clusters/" + filename + "Cut2005"+ "LeastSimilarHIGH"+str(high_threshold)+","+str(low_threshold)+".clusters"
    movie_vector_fn = "newdata/spaces/" + filename +".mds"
    cluster_names = "Clusters/" + filename + "Cut2005" + "LeastSimilarHIGH"+str(high_threshold)+","+str(low_threshold)+".names"
    movie_names = "filmdata/filmNames.txt"
    rank_fn = filename
    print filename
    Rankings(cluster_fn, movie_vector_fn, cluster_names, movie_names, rank_fn, percent, percent_increment)

    low_threshold = 0.033
    high_threshold = 0.26
    filename = "AUTOENCODERN0.6R0.0tanhtanhmse1tanh504SDA3"
    cluster_fn = "Clusters/" + filename + "Cut2006" + "LeastSimilarHIGH"+str(high_threshold)+","+str(low_threshold)+".clusters"
    movie_vector_fn = "newdata/spaces/" + filename +".mds"
    cluster_names = "Clusters/" + filename + "Cut2006" + "LeastSimilarHIGH"+str(high_threshold)+","+str(low_threshold)+".names"
    movie_names = "filmdata/filmNames.txt"
    rank_fn = filename
    print filename
    Rankings(cluster_fn, movie_vector_fn, cluster_names, movie_names, rank_fn, percent, percent_increment)

    low_threshold = 0.021
    high_threshold = 0.20
    filename = "AUTOENCODERN0.6R0.0tanhtanhmse1tanh254SDA4"
    cluster_fn = "Clusters/" + filename + "Cut2007" + "LeastSimilarHIGH"+str(high_threshold)+","+str(low_threshold)+".clusters"
    movie_vector_fn = "newdata/spaces/" + filename + ".mds"
    cluster_names = "Clusters/" + filename + "Cut2007" + "LeastSimilarHIGH"+str(high_threshold)+","+str(low_threshold)+".names"
    movie_names = "filmdata/filmNames.txt"
    rank_fn = filename
    print filename
    Rankings(cluster_fn, movie_vector_fn, cluster_names, movie_names, rank_fn, percent, percent_increment)









if  __name__ =='__main__':main()