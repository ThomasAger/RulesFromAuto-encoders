from scipy import stats, linalg, spatial

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

"""

Given high directions and low directions, find a similarity mapping such that
each high value direction has a list of low value directions associated with it
according to a certain threshold, choosing 2n directions with n being the amount
of dimensions in the input space and each directions in the sequence being chosen
based on how dissimilar it was from the previous.

"""

def createTermClusters(self, hv_directions, lv_directions, input_size, similarity_threshold):
    hl_clusters = {}
    h = []
    selected_terms = []
    for i in range(input_size * 2):
        if i == 0:
            h = hv_directions[0]
        else:
            # Using a mean of all previous terms here to avoid the problem of finding a term very similar to
            # ...the previous one, e.g. "documentary" and "documentarys"
            combined_terms = dt.mean_of_array(selected_terms)
            h = self.getLeastSimilarTerm(h, hv_directions)
        selected_terms.append(h)
        for l in lv_directions:
            if self.getSimilarity(l, h) > 0.5:
                hl_clusters[h].append(l)
    return hl_clusters

def getLeastSimilarTerm(self, term, terms_to_match):
    lowest_term = 99999999
    term_index = 0
    for t in range(len(terms_to_match)):
        s = self.getSimilarity(term, terms_to_match[t])
        if s < lowest_term:
            lowest_term = s
            term_index = t
    return terms_to_match[term_index]


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

Sample the data such that there is a unique amount of sampled movies for each
phrase, according to how often that phrase occurs.

"""

def getSampledData(file_names, movie_labels, amount_to_cut_at, largest_cut):

    print len(movie_labels)
    print len(movie_labels[0])

    for yt in range(len(movie_labels) - 1):
        y1 = 0
        y0 = 0
        for y in range(len(movie_labels[yt])):
            if movie_labels[yt][y] == 1:
                y1 += 1
            if movie_labels[yt][y] == 0:
                y0 += 1

        print yt, "len(0):", y0, "len(1):", y1, file_names[yt]

        if y1 < amount_to_cut_at or y1 > largest_cut:
            movie_labels[yt] = None
            file_names[yt] = None

            print yt, "skipped deleted", file_names[yt]

            continue

    file_names = [x for x in file_names if x is not None]
    movie_labels = [x for x in movie_labels if x is not None]

    return file_names, movie_labels

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