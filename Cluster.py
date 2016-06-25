from scipy import stats, linalg, spatial
import numpy as np
import DataTasks as dt
from collections import OrderedDict


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

def getNextClusterTerm(cluster_terms, terms_to_match, terms_to_ignore, amt):
    min_value = 999999999999999
    min_index = 0
    for t in range(len(terms_to_match)):
        max_value = 0
        if dt.checkIfInArray(terms_to_ignore, t) is False:
            for c in range(len(cluster_terms)):
                s = getSimilarity(cluster_terms[c], terms_to_match[t])
                if s > max_value:
                    max_value = s
            if max_value < min_value:
                min_value = max_value
                min_index = t
    return min_index

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
                                           "SVMResults/"+filename+".scores",
                                           "SVMResults/"+filename+".names",
                                            low_threshold, high_threshold)

        least_similar_cluster_names, cluster_name_dict, least_similar_clusters = self.createTermClusters(hd, ld, hdn, ldn)

        dt.write1dArray(least_similar_cluster_names, "Clusters/"+filename+"LeastSimilarHIGH"+str(high_threshold)+","+str(low_threshold)+".names")
        dt.write2dArray(least_similar_clusters, "Clusters/"+filename+"LeastSimilarHIGH"+str(high_threshold)+","+str(low_threshold)+".clusters")
        dt.writeArrayDict(cluster_name_dict, "Clusters/"+filename+"MostSimilarCLUSTER"+str(high_threshold)+","+str(low_threshold)+".names")

    # Splitting into high and low directions based on threshold
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

        print "Overall amount of HV directions: ", len(hv_directions)

        # Create high-valued clusters
        amt_of_clusters = len(hv_directions[0]) * 2
        for i in range(len(hv_directions)):
            if i == 0:
                least_similar_cluster_ids.append(i)
                least_similar_clusters.append(hv_directions[i])
                least_similar_cluster_names.append(hv_names[i])
                print "Least Similar Term", hv_names[i]
            elif i >= amt_of_clusters:
                directions_to_add.append(hv_directions[i])
                names_to_add.append(hv_names[i])
                print "Added", hv_names[i], "To the remaining directions to add"
            else:
                ti = getNextClusterTerm(least_similar_clusters, hv_directions, least_similar_cluster_ids, 1)
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

        # Initialize dictionaries for printing / visualizing
        cluster_name_dict = OrderedDict()
        for c in least_similar_cluster_names:
            cluster_name_dict[c] = []

        # For every low value direction, find the high value direction its most similar to and append it to the directions
        every_cluster_direction = []
        for i in least_similar_clusters:
            every_cluster_direction.append([i])

        # Reversing so that the top names and directions are first
        lv_names.reverse()
        lv_directions.reverse()

        # Finding the most similar directions to each cluster_centre
        # Creating a dictionary of {cluster_centre: [cluster_direction(1), ..., cluster_direction(n)]} pairs
        for d in range(len(lv_directions)):
            i = getXMostSimilarIndex(lv_directions[d], least_similar_clusters, [], 1)[0]
            every_cluster_direction[i].append(lv_directions[d])
            print str(d+1)+"/"+str(len(lv_directions)), "Most Similar to", lv_names[d], "Is", least_similar_cluster_names[i]
            cluster_name_dict[least_similar_cluster_names[i]].append(lv_names[d])

        # Mean of all directions = cluster direction
        cluster_directions = []
        for l in range(len(least_similar_clusters)):
            cluster_directions.append(dt.mean_of_array(every_cluster_direction[l]))

        """
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
        """

        return least_similar_cluster_names, cluster_name_dict, cluster_directions

def main(low_threshold, high_threshold, directions_filename):
    """
    filename = "AUTOENCODERN0.6R0.0tanhtanhmse1tanh254SDA4Cut2007HIGH0.201,0.021"#"AUTOENCODERN0.6R0.0tanhtanhmse1tanh254SDA4Cut2007HIGH0.2,0.021"#"AUTOENCODERN0.6R0.0tanhtanhmse1tanh2004SDA1Cut2004HIGH0.41,0.1"#
    low_fn = "AUTOENCODERN0.6R0.0tanhtanhmse1tanh254SDA4Cut2007LOW0.201,0.021"
    directions_fn = "Directions/"+filename+".directions"
    low_directions_fn = "Directions/"+low_fn+".directions"
    low_names_fn = "Directions/"+low_fn+".names"
    names_fn = "Directions/"+filename+".names"
    directions = dt.importVectors(directions_fn)
    low_directions = dt.importVectors(low_directions_fn)
    low_names = dt.importString(low_names_fn)
    names = dt.importString(names_fn)
    term = "learned"
    direction = []
    for n in range(len(names)):
        if names[n] == term:
            direction.append(directions[n])
    least_sim = getXMostSimilarIndex(direction[0], low_directions, [], 10)
    for n in least_sim:
        print low_names[n]

    """
    """
    low_threshold = 0.1
    high_threshold = 0.44
    filename = "films200Cut2000"
    print filename
    Cluster(low_threshold, high_threshold, filename)
    """
    # Check if the input is an array or not, so it can be processed by the for loop
    if isinstance(low_threshold, list) is False:
        low_threshold = [low_threshold]
        high_threshold = [high_threshold]
        directions_filename = [directions_filename]

    for i in range(len(low_threshold)):
        Cluster(low_threshold[i], high_threshold[i], directions_filename[i])
    """
    low_threshold = 0.021
    high_threshold = 0.20
    filename = "AUTOENCODERN0.6R0.0tanhtanhmse1tanh254SDA4Cut2007"
    print filename
    Cluster(low_threshold, high_threshold, filename)

    low_threshold = 0.1
    high_threshold = 0.41
    filename = "AUTOENCODERN0.6R0.0tanhtanhmse1tanh2004SDA1Cut2004"
    print filename
    Cluster(low_threshold, high_threshold, filename)

    low_threshold = 0.055
    high_threshold = 0.34
    filename = "AUTOENCODERN0.6R0.0tanhtanhmse1tanh1004SDA2Cut2005"
    print filename
    Cluster(low_threshold, high_threshold, filename)

    low_threshold = 0.033
    high_threshold = 0.26
    filename = "AUTOENCODERN0.6R0.0tanhtanhmse1tanh504SDA3Cut2006"
    print filename
    Cluster(low_threshold, high_threshold, filename)
    """




if  __name__ =='__main__':main(0.07, 0.34, "films100N0.6H25L3Cut")
