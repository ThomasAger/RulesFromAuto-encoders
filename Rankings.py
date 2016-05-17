import numpy as np
import DataTasks as dt
from collections import OrderedDict
class Rankings:
    def __init__(self, directions_fn, vectors_fn, cluster_names_fn, vector_names_fn, fn, percent):

        directions = dt.importVectors(directions_fn)
        vectors = dt.importVectors(vectors_fn)
        cluster_names = dt.importString(cluster_names_fn)
        vector_names = dt.importString(vector_names_fn)

        rankings, ranking_names = self.getRankings(directions, vectors, cluster_names, vector_names)
        rankings = np.array(rankings)
        labels = self.createLabels(rankings, percent)
        labels = np.asarray(labels)
        labels = labels.transpose()
        rankings = rankings.transpose()
        dt.write2dArray(labels, "Rankings/" + fn + ".labels")
        dt.write2dArray(rankings, "Rankings/" + fn + ".space")
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



def main():
    percent = 0.1
    low_threshold = 0.1
    high_threshold = 0.44
    filename = "films200"
    cluster_fn = "Clusters/" + filename + "Cut2000"+  "LeastSimilarHIGH"+str(high_threshold)+","+str(low_threshold)+".clusters"
    movie_vector_fn = "filmdata/films200.mds/" + filename +".mds"
    cluster_names = "Clusters/" + filename + "Cut2000"+  "LeastSimilarHIGH"+str(high_threshold)+","+str(low_threshold)+".names"
    movie_names = "filmdata/filmNames.txt"
    rank_fn = filename + "P" + str(percent)
    print filename
    Rankings(cluster_fn, movie_vector_fn, cluster_names, movie_names, rank_fn, percent)
    low_threshold = 0.1
    high_threshold = 0.4
    filename = "AUTOENCODERN0.4R0.0tanhtanhmse2tanh2004SDA1"
    cluster_fn = "Clusters/" + filename + "Cut2000"+ "LeastSimilarHIGH"+str(high_threshold)+","+str(low_threshold)+".clusters"
    movie_vector_fn = "newdata/spaces/" + filename +".mds"
    cluster_names = "Clusters/" + filename + "Cut2000" + "LeastSimilarHIGH"+str(high_threshold)+","+str(low_threshold)+".names"
    movie_names = "filmdata/filmNames.txt"
    rank_fn = filename + "P" + str(percent)
    print filename
    Rankings(cluster_fn, movie_vector_fn, cluster_names, movie_names, rank_fn, percent)
    low_threshold = 0.08
    high_threshold = 0.37
    filename = "AUTOENCODERN0.6R0.0tanhtanhmse2tanh1004SDA2"
    cluster_fn = "Clusters/" + filename + "Cut2001" + "LeastSimilarHIGH"+str(high_threshold)+","+str(low_threshold)+".clusters"
    movie_vector_fn = "newdata/spaces/" + filename +".mds"
    cluster_names = "Clusters/" + filename + "Cut2001" + "LeastSimilarHIGH"+str(high_threshold)+","+str(low_threshold)+".names"
    movie_names = "filmdata/filmNames.txt"
    rank_fn = filename + "P" + str(percent)
    print filename
    Rankings(cluster_fn, movie_vector_fn, cluster_names, movie_names, rank_fn, percent)
    low_threshold = 0.06
    high_threshold = 0.28
    filename = "AUTOENCODERN0.8R0.0tanhtanhmse2tanh504SDA3"
    cluster_fn = "Clusters/" + filename + "Cut2002" + "LeastSimilarHIGH"+str(high_threshold)+","+str(low_threshold)+".clusters"
    movie_vector_fn = "newdata/spaces/" + filename + ".mds"
    cluster_names = "Clusters/" + filename + "Cut2002" + "LeastSimilarHIGH"+str(high_threshold)+","+str(low_threshold)+".names"
    movie_names = "filmdata/filmNames.txt"
    rank_fn = filename + "P" + str(percent)
    print filename
    Rankings(cluster_fn, movie_vector_fn, cluster_names, movie_names, rank_fn, percent)
    low_threshold = 0.04
    high_threshold = 0.21
    filename = "AUTOENCODERN1.0R0.0tanhtanhmse2tanh254SDA4"
    cluster_fn = "Clusters/" + filename + "Cut2003" + "LeastSimilarHIGH"+str(high_threshold)+","+str(low_threshold)+".clusters"
    movie_vector_fn = "newdata/spaces/" + filename + ".mds"
    cluster_names = "Clusters/" + filename + "Cut2003" + "LeastSimilarHIGH"+str(high_threshold)+","+str(low_threshold)+".names"
    movie_names = "filmdata/filmNames.txt"
    rank_fn = filename + "P" + str(percent)
    print filename
    Rankings(cluster_fn, movie_vector_fn, cluster_names, movie_names, rank_fn, percent)




if  __name__ =='__main__':main()