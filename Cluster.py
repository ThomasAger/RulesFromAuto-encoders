from scipy import stats, linalg, spatial
import numpy as np
import DataTasks as dt
from collections import defaultdict


def getSimilarity(vector1, vector2):
    return 1 - spatial.distance.cosine(vector1, vector2)

def getXLeastSimilarIndex(term, terms_to_match, terms_to_ignore, amt):
    least_similar_term_indexes = []
    for a in range(amt):
        lowest_term = 99999999
        term_index = 0
        for t in range(len(terms_to_match)):
            if dt.checkIfInArray(terms_to_ignore, t) is False:
                s = getSimilarity(term, terms_to_match[t])
                if s < lowest_term and dt.checkIfInArray(least_similar_term_indexes, t) is False:
                    lowest_term = s
                    term_index = t
        least_similar_term_indexes.append(term_index)
    return least_similar_term_indexes

def getXMostSimilarIndex(term, terms_to_match, terms_to_ignore, amt):
    most_similar_term_indexes = []
    for a in range(amt):
        highest_term = 0
        term_index = 0
        for t in range(len(terms_to_match)):
            if dt.checkIfInArray(terms_to_ignore, t) is False:
                s = getSimilarity(term, terms_to_match[t])
                if s > highest_term and dt.checkIfInArray(most_similar_term_indexes, t) is False:
                    highest_term = s
                    term_index = t
        most_similar_term_indexes.append(term_index)
    return most_similar_term_indexes


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

        most_similar, least_similar, \
        least_similar_cluster_names, cluster_dict_names, cluster_dict_values, least_similar_clusters = self.createTermClusters(hd, ld, hdn, ldn)

        dt.write1dArray(most_similar, "Clusters/"+filename+"MostSimilarNames"+str(high_threshold)+","+str(low_threshold)+".names")
        dt.write1dArray(least_similar, "Clusters/"+filename+"LeastSimilarNames"+str(high_threshold)+","+str(low_threshold)+".names")
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
        sorted_h_indexes = dt.sortByArray(high_direction_indexes,   high_direction_scores)
        sorted_l_indexes = dt.sortByArray(low_direction_indexes , low_direction_scores)
        sorted_h_indexes.reverse()
        sorted_l_indexes.reverse()
        high_direction_names = []
        low_direction_names = []
        high_directions = []
        low_directions = []
        for s in sorted_h_indexes:
            high_directions.append(directions[s])
            high_direction_names.append(names[s][6:])
        for s in sorted_l_indexes:
            low_directions.append(directions[s])
            low_direction_names.append(names[s][6:])
        return high_direction_names, low_direction_names, high_directions, low_directions

    def createTermClusters(self, hv_directions, lv_directions, hv_names, lv_names):
        least_similar_clusters = []
        least_similar_cluster_ids = []
        least_similar_cluster_names = []
        directions_to_add = []
        names_to_add = []

        # Create high-valued clusters
        amt_of_clusters = len(hv_directions[0])
        for i in range(amt_of_clusters):
            if i == 0:
                least_similar_cluster_ids.append(i)
                least_similar_clusters.append(hv_directions[i])
                least_similar_cluster_names.append(hv_names[i])
                print "Least Similar Term", hv_names[i]
            elif i >= amt_of_clusters:
                directions_to_add.append(hv_directions[i])
                names_to_add.append(hv_names[i])
            else:
                combined_terms = dt.mean_of_array(least_similar_clusters)
                ti = getXLeastSimilarIndex(combined_terms, hv_directions, least_similar_cluster_ids, 1)[0]
                least_similar_cluster_ids.append(ti)
                least_similar_clusters.append(hv_directions[ti])
                least_similar_cluster_names.append(hv_names[ti])
                print str(i+1)+"/"+str(amt_of_clusters), "Least Similar Term", hv_names[ti]
        # Add remaining high value directions to the low value direction list
        directions_to_add.reverse()
        names_to_add.reverse()
        for i in range(len(directions_to_add)):
            lv_directions.insert(0, directions_to_add[i])
            lv_names.insert(0, names_to_add[i])

        # Initialize dictionaries for printing / visualizing later
        cluster_name_dict = defaultdict(list)
        cluster_amt_dict = defaultdict(int)
        cluster_index_dict = defaultdict(list)
        for c in least_similar_cluster_names:
            cluster_name_dict[c] = []
            cluster_amt_dict[c] = 0
            cluster_index_dict[c] = []

        # For every low value direction, find the high value direction its most similar to and append it to the directions
        every_cluster_direction = []
        for i in least_similar_clusters:
            every_cluster_direction.append([i])
        for d in range(len(lv_directions)):
            i = getXMostSimilarIndex(lv_directions[d], least_similar_clusters, [], 1)[0]
            every_cluster_direction[i].append(lv_directions[d])
            print str(d+1)+"/"+str(len(lv_directions)), "Most Similar to", lv_names[d], "Is", least_similar_cluster_names[i]
            cluster_name_dict[least_similar_cluster_names[i]].append(lv_names[d])
            cluster_amt_dict[least_similar_cluster_names[i]] += 1
            cluster_index_dict[least_similar_cluster_names[i]].append(d)

        # Get the means of all the clustered directions
        cluster_directions = []
        for l in range(len(least_similar_clusters)):
            cluster_directions.append(dt.mean_of_array(every_cluster_direction[l]))

        # Create names and values to save later
        cluster_dict_names = []
        cluster_dict_values = []
        cluster_dict_directions = []
        for k in sorted(cluster_amt_dict, key=cluster_amt_dict.get, reverse=True):
            cluster_dict_values.append([k, cluster_amt_dict[k]])
            cluster_dict_names.append([k, cluster_name_dict[k]])
            cluster_dict_directions.append(cluster_index_dict[k])
            print k, cluster_amt_dict[k]
            print k, cluster_name_dict[k]

        # Get the 10 most similar and least similar directions to save later
        most_similar = []
        least_similar = []
        most_similar_indexes = []
        least_similar_indexes = []
        indexes_to_find = []
        for k in sorted(cluster_amt_dict, key=cluster_amt_dict.get, reverse=True):
            name_to_get_most_similar = k
            index_to_find = dt.getIndexInArray(hv_names, name_to_get_most_similar)
            amt = 10
            indexes_to_find.append(index_to_find)
            most_similar_index = getXMostSimilarIndex(hv_directions[index_to_find], hv_directions, [index_to_find], amt)
            least_similar_index = getXLeastSimilarIndex(hv_directions[index_to_find], hv_directions, [index_to_find], amt)
            most_similar_indexes.append(most_similar_index)
            least_similar_indexes.append(least_similar_index)
        for m in range(len(most_similar_indexes)):
            line_to_append = []
            for v in range(len(most_similar_indexes[m])):
                line_to_append.append(hv_names[most_similar_indexes[m][v]])
            most_similar.append([cluster_dict_names[m][0], line_to_append])
        for l in range(len(least_similar_indexes)):
            line_to_append = []
            for v in range(len(least_similar_indexes[l])):
                line_to_append.append(hv_names[least_similar_indexes[l][v]])
            least_similar.append([cluster_dict_names[l][0], line_to_append])

        return most_similar, least_similar, least_similar_cluster_names, cluster_dict_names, cluster_dict_values, cluster_directions

def main():
    """
    filename = "AUTOENCODER1.0tanhtanhmse2tanh2004SDACut2001HIGH0.4,0.3"
    directions_fn = "Directions/"+filename+".directions"
    names_fn = "Directions/"+filename+".names"
    directions = dt.importVectors(directions_fn)
    names = dt.importString(names_fn)
    for n in names:


    dt.write1dArray(, "Clusters/"+filename+"MostSimilar".txt)
    dt.write1dArray(least_similar_index, "Clusters/"+filename+"LeastSimilar".txt)
    print "Checking:", name_to_get_most_similar
    for i in most_similar_index:
        print names[i]
    """

    low_threshold = 0.1
    high_threshold = 0.44
    filename = "films200Cut2000"
    print filename
    Cluster(low_threshold, high_threshold, filename)
    low_threshold = 0.1
    high_threshold = 0.4
    filename = "AUTOENCODERN0.4R0.0tanhtanhmse2tanh2004SDA1Cut2000"
    print filename
    Cluster(low_threshold, high_threshold, filename)
    low_threshold = 0.08
    high_threshold = 0.37
    filename = "AUTOENCODERN0.6R0.0tanhtanhmse2tanh1004SDA2Cut2001"
    print filename
    Cluster(low_threshold, high_threshold, filename)
    low_threshold = 0.06
    high_threshold = 0.28
    filename = "AUTOENCODERN0.8R0.0tanhtanhmse2tanh504SDA3Cut2002"
    print filename
    Cluster(low_threshold, high_threshold, filename)
    low_threshold = 0.04
    high_threshold = 0.21
    filename = "AUTOENCODERN1.0R0.0tanhtanhmse2tanh254SDA4Cut2003"
    print filename
    Cluster(low_threshold, high_threshold, filename)

if  __name__ =='__main__':main()
