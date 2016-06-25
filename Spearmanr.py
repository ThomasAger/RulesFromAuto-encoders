import numpy as np
import DataTasks as dt
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import average_precision_score
from sklearn import preprocessing
def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])

def gini(solution, submission):
    df = zip(solution, submission, range(len(solution)))
    df = sorted(df, key=lambda x: (x[1],-x[2]), reverse=True)
    rand = [float(i+1)/float(len(df)) for i in range(len(df))]
    totalPos = float(sum([x[0] for x in df]))
    cumPosFound = [df[0][0]]
    for i in range(1,len(df)):
        cumPosFound.append(cumPosFound[len(cumPosFound)-1] + df[i][0])
    Lorentz = [float(x)/totalPos for x in cumPosFound]
    Gini = [Lorentz[i]-rand[i] for i in range(len(df))]
    return sum(Gini)

def normalized_gini(solution, submission):
    normalized_gini = gini(solution, submission)/gini(solution, solution)
    return normalized_gini
def dcg_at_k(r, k, method=0):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> dcg_at_k(r, 1)
    3.0
    >>> dcg_at_k(r, 1, method=1)
    3.0
    >>> dcg_at_k(r, 2)
    5.0
    >>> dcg_at_k(r, 2, method=1)
    4.2618595071429155
    >>> dcg_at_k(r, 10)
    9.6051177391888114
    >>> dcg_at_k(r, 11)
    9.6051177391888114
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.
def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])

def ndcg_at_k(r, k, method=0):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> ndcg_at_k(r, 1)
    1.0
    >>> r = [2, 1, 2, 0]
    >>> ndcg_at_k(r, 4)
    0.9203032077642922
    >>> ndcg_at_k(r, 4, method=1)
    0.96519546960144276
    >>> ndcg_at_k([0], 1)
    0.0
    >>> ndcg_at_k([1], 2)
    1.0
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Normalized discounted cumulative gain
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max

class Rankings:
    def __init__(self, direction_fn, ppmi_fn, phrases_fn, phrases_to_check_fn, fn):
        ppmi = dt.importLabels(ppmi_fn)
        ppmi = np.asarray(ppmi)
        phrases = dt.importString(phrases_fn)

        indexes_to_get = []
        if phrases_to_check_fn != "":
            phrases_to_check = dt.importString(phrases_to_check_fn)
            for pc in range(len(phrases_to_check)):
                for p in range(len(phrases)):
                    if phrases_to_check[pc] == phrases[p][6:]:
                        indexes_to_get.append(p)
        indexes_to_get.sort()
        ppmi = ppmi.transpose()
        print len(ppmi), len(ppmi[0])
        scores = []
        pvalues = []
        scores_kendall = []
        pvalues_kendall = []
        agini = []
        agini1 = []
        angini1 = []
        angini = []
        amap = []
        andcg = []
        counter = 0
        averages = []
        with open(direction_fn) as f:
            for line in f:
                exists = True
                if phrases_to_check_fn != "":
                    exists = False
                    for i in indexes_to_get:
                        if i == counter:
                            exists = True
                            break
                if exists:
                    total = 0
                    amt = 0
                    direction = line.split()
                    for d in range(len(direction)):
                        direction[d] = float(direction[d])
                    new_direction = []
                    new_ppmi = []
                    direction_rank = np.argsort(direction)
                    ppmi_rank = np.argsort(ppmi[counter])
                    for d in range(len(ppmi[counter])):
                        if ppmi[counter][d] != 0:
                            total += ppmi[counter][d]
                            amt += 1
                            new_direction.append(direction_rank[d])
                            new_ppmi.append(ppmi_rank[d])
                    average = total / amt

                    min_max_scaler = preprocessing.MinMaxScaler()
                    normalized_ppmi = min_max_scaler.fit_transform(ppmi[counter])
                    normalized_dir = min_max_scaler.fit_transform(direction)

                    ginis = gini(normalized_ppmi, normalized_dir)

                    ranked_ppmi = dt.sortByArray(new_ppmi, new_direction)
                    nr_ppmi = min_max_scaler.fit_transform(ranked_ppmi)
                    ndcgs = ndcg_at_k(nr_ppmi, len(nr_ppmi))

                    #binarizer = preprocessing.Binarizer()
                    #binary_ppmi = binarizer.transform(normalized_ppmi)
                    #normalized_dir = np.ndarray.tolist(normalized_dir)
                    map = 0#average_precision_score(normalized_ppmi, normalized_dir)

                    rho, pvalue = spearmanr(new_ppmi, new_direction)
                    rhok, pvaluek = kendalltau(new_ppmi, new_direction)

                    scores.append(rho)
                    pvalues.append(pvalue)
                    scores_kendall.append(rhok)
                    pvalues_kendall.append(pvaluek)
                    andcg.append(ndcgs)
                    agini.append(ginis)
                    amap.append(map)
                    averages.append(average)
                    print phrases[counter] + ":", map, ginis

                counter += 1
        dt.write1dArray(scores, "RuleType/s" + fn + ".score")
        dt.write1dArray(pvalues, "RuleType/s" + fn + ".pvalue")
        dt.write1dArray(scores_kendall, "RuleType/k" + fn + ".score")
        dt.write1dArray(pvalues_kendall, "RuleType/k" + fn + ".pvalue")
        dt.write1dArray(phrases, "RuleType/" + fn + ".names")
        dt.write1dArray(averages, "RuleType/" + fn + ".averages")
        dt.write1dArray(agini, "RuleType/gn" + fn + ".score")
        dt.write1dArray(andcg, "RuleType/ndcg" + fn + ".score")
        dt.write1dArray(amap, "RuleType/map" + fn + ".score")


def main():

    direction_fn = "Rankings/films100phrase.space"
    ppmi_fn = "filmdata/classesPhrases/nonbinary/class-all"
    phrases_fn = "SVMResults/films100.names"
    phrases_to_check_fn = ""#"RuleType/top1ksorted.txt"
    fn = "films100"

    Rankings(direction_fn, ppmi_fn, phrases_fn, phrases_to_check_fn, fn)









if  __name__ =='__main__':main()