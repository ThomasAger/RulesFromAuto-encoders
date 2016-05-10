from scipy import stats, linalg, spatial
import numpy as np
import DataTasks as dt
from collections import defaultdict
class Cluster:

    def __init__(self, low_threshold, high_threshold,  filename):

        hdn, ldn, hd, ld = self.splitDirections("Directions/"+filename+".directions",
                                           "SVMResults/ALL_SCORES_"+filename+".txt",
                                           "SVMResults/ALL_NAMES_"+filename+".txt",
                                            low_threshold, high_threshold)
        dt.write2dArray(hd, "Directions/"+filename+"HIGH"+str(high_threshold)+","+str(low_threshold)+".directions")
        dt.write2dArray(ld, "Directions/"+filename+"LOW"+str(high_threshold)+","+str(low_threshold)+".directions")
        dt.write1dArray(hdn, "Directions/"+filename+"HIGH"+str(high_threshold)+","+str(low_threshold)+".names")
        dt.write1dArray(ldn, "Directions/"+filename+"LOW"+str(high_threshold)+","+str(low_threshold)+".names")

        least_similar_cluster_names, cluster_dict_names, cluster_dict_values, least_similar_clusters = self.createTermClusters(hd, ld, hdn, ldn)

        dt.write1dArray(least_similar_cluster_names, "Clusters/"+filename+"LeastSimilarHIGH"+str(high_threshold)+","+str(low_threshold)+".names")
        dt.write2dArray(least_similar_clusters, "Clusters/"+filename+"LeastSimilarHIGH"+str(high_threshold)+","+str(low_threshold)+".clusters")
        dt.write1dArray(cluster_dict_names, "Clusters/"+filename+"MostSimilarCLUSTER"+str(high_threshold)+","+str(low_threshold)+".names")
        dt.write1dArray(cluster_dict_values, "Clusters/"+filename+"MostSimilarValuesCLUSTER"+str(high_threshold)+","+str(low_threshold)+".names")

    def splitDirections(self, directions_fn, scores_fn, names_fn, low_threshold, high_threshold):
        directions = dt.importVectors(directions_fn)
        scores = dt.importString(scores_fn)
        names = dt.importString(names_fn)
        for s in range(len(scores)):
            scores[s] = float(scores[s].strip())
        high_direction_indexes = []
        high_direction_scores = []
        low_direction_indexes = []
        low_direction_scores = []
        for s in range(len(scores)):
            if scores[s] >= high_threshold:
                high_direction_indexes.append(s)
                high_direction_scores.append(scores[s])
            elif scores[s] >= low_threshold:
                low_direction_indexes.append(s)
                low_direction_scores.append(scores[s])
        sorted_h_indexes = dt.sortByArray(high_direction_scores,  high_direction_indexes)
        sorted_l_indexes = dt.sortByArray(low_direction_scores, low_direction_indexes)
        sorted_h_indexes.reverse()
        sorted_l_indexes.reverse()
        high_direction_names = []
        low_direction_names = []
        high_directions = []
        low_directions = []
        for s in sorted_h_indexes:
            high_directions.append(directions[s])
            high_direction_names.append(names[s])
        for s in sorted_l_indexes:
            low_directions.append(directions[s])
            low_direction_names.append(names[s])
        return high_direction_names, low_direction_names, high_directions, low_directions

    def getSimilarity(self, vector1, vector2):
        return 1 - spatial.distance.cosine(vector1, vector2)

    def getLeastSimilarTermIndex(self, term, terms_to_match, terms_to_ignore):
        lowest_term = 99999999
        term_index = 0
        for t in range(len(terms_to_match)):
            if dt.checkIfInArray(terms_to_ignore, t) is False:
                s = self.getSimilarity(term, terms_to_match[t])
                if s < lowest_term:
                    lowest_term = s
                    term_index = t
        return term_index

    def getMostSimilarTermIndex(self, term, terms_to_match, terms_to_ignore):
        highest_term = 0
        term_index = 0
        for t in range(len(terms_to_match)):
            if dt.checkIfInArray(terms_to_ignore, t) is False:
                s = self.getSimilarity(term, terms_to_match[t])
                if s > highest_term:
                    highest_term = s
                    term_index = t
        return term_index

    def createTermClusters(self, hv_directions, lv_directions, hv_names, lv_names):
        least_similar_clusters = []
        least_similar_cluster_ids = []
        least_similar_cluster_names = []
        print hv_names
        for i in range(len(hv_directions[0])):
            if i == 0:
                least_similar_cluster_ids.append(i)
                least_similar_clusters.append(hv_directions[i])
                least_similar_cluster_names.append(hv_names[i])
                print "Least Similar Term", hv_names[i]
            else:
                combined_terms = dt.mean_of_array(least_similar_clusters)
                ti = self.getLeastSimilarTermIndex(combined_terms, hv_directions, least_similar_cluster_ids)
                least_similar_cluster_ids.append(ti)
                least_similar_clusters.append(hv_directions[ti])
                least_similar_cluster_names.append(hv_names[ti])
                print str(i)+"/"+str(len(hv_directions[0])), "Least Similar Term", hv_names[ti]
        cluster_name_dict = defaultdict(list)
        cluster_amt_dict = defaultdict(int)
        for d in range(len(lv_directions)):
            i = self.getMostSimilarTermIndex(lv_directions[d], least_similar_clusters, [])
            least_similar_clusters[i] = dt.mean_of_array([least_similar_clusters[i], lv_directions[d]])
            print str(d)+"/"+str(len(lv_directions)), "Most Similar to", lv_names[d], "Is", least_similar_cluster_names[i]
            cluster_name_dict[least_similar_cluster_names[i]].append(lv_names[d])
            cluster_amt_dict[least_similar_cluster_names[i]] += 1
        cluster_dict_names = []
        cluster_dict_values = []
        for key in sorted(cluster_name_dict.items()):
            cluster_dict_names.append([key])
            print key
        for key in sorted(cluster_amt_dict.items()):
            cluster_dict_values.append([key])

        return least_similar_cluster_names, cluster_dict_names, cluster_dict_values, least_similar_clusters

def main():
    low_threshold = 0.3
    high_threshold = 0.4
    filename = "normal200"
    Cluster(low_threshold, high_threshold, filename)
    low_threshold = 0.3
    high_threshold = 0.4
    filename = "AUTOENCODER1.0tanhtanhmse2tanh2004SDACut2001"
    Cluster(low_threshold, high_threshold, filename)
    low_threshold = 0.12
    high_threshold = 0.35
    filename = "AUTOENCODER1.0tanhtanhmse2tanh2004SDACut2002"
    Cluster(low_threshold, high_threshold, filename)
    low_threshold = 0.08
    high_threshold = 0.30
    filename = "AUTOENCODER1.0tanhtanhmse2tanh2004SDACut2003"
    Cluster(low_threshold, high_threshold, filename)
    low_threshold = 0.01
    high_threshold = 0.22
    filename = "AUTOENCODER1.0tanhtanhmse2tanh2004SDACut2004"
    Cluster(low_threshold, high_threshold, filename)
    low_threshold = 0.01
    high_threshold = 0.15
    filename = "AUTOENCODER1.0tanhtanhmse2tanh2004SDACut2005"
    Cluster(low_threshold, high_threshold, filename)

if  __name__ =='__main__':main()
