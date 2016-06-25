import numpy as np
import DataTasks as dt
from scipy.stats import spearmanr, kendalltau


class Gini:
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

        ppmi = ppmi.transpose()
        print len(ppmi), len(ppmi[0])
        scores = []
        pvalues = []
        scores_kendall = []
        pvalues_kendall = []
        counter = 0
        averages = []
        with open(direction_fn) as f:
            for line in f:
                if indexes_to_get is not []:
                    for i in indexes_to_get:
                        if i == counter:
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
                            rho, pvalue = spearmanr(new_ppmi, new_direction)
                            scores.append(rho)
                            pvalues.append(pvalue)
                            scores_kendall.append(rhok)
                            pvalues_kendall.append(pvaluek)
                            averages.append(average)
                            print phrases[counter] + ":", rho, pvalue, average
                else:
                    direction = line.split()
                    for d in range(len(direction)):
                        direction[d] = float(direction[d])
                    direction_rank = np.argsort(direction)
                    ppmi_rank = np.argsort(ppmi[counter])
                    rho, pvalue = spearmanr(direction_rank, ppmi_rank)
                    scores.append(rho)
                    pvalues.append(pvalue)
                    print phrases[counter] + ":", rho, pvalue
                counter += 1
        dt.write1dArray(scores, "RuleType/s" + fn + ".score")
        dt.write1dArray(pvalues, "RuleType/s" + fn + ".pvalue")
        dt.write1dArray(scores_kendall, "RuleType/k" + fn + ".score")
        dt.write1dArray(pvalues_kendall, "RuleType/k" + fn + ".pvalue")
        dt.write1dArray(phrases, "RuleType/" + fn + ".names")
        dt.write1dArray(averages, "RuleType/" + fn + ".averages")


def main():

    direction_fn = "Rankings/films100phrase.space"
    ppmi_fn = "filmdata/classesPhrases/nonbinary/class-all"
    phrases_fn = "SVMResults/films100.names"
    phrases_to_check_fn = "RuleType/top1k.txt"
    fn = "films100"

    Gini(direction_fn, ppmi_fn, phrases_fn, phrases_to_check_fn, fn)

if  __name__ =='__main__':main()