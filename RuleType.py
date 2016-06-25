import numpy as np
import DataTasks as dt
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def saveGraph(discrete_labels, ppmi, fn):
    x = []
    for i in range(len(discrete_labels)):
        x.append(int(discrete_labels[i][:-1]))
    y = ppmi

    amts = [0] * 100
    totals = [0] * 100
    for i in range(len(x)):
        amts[x[i]-1] += y[i]
        totals[x[i]-1] += 1
    avgs = []
    excess = []
    for i in range(len(amts)):
        avgs.append(totals[i] / amts[i])
        if totals[i] == 0 and amts[i] == 0:
            excess.append(i)
    print len(excess)
    y = avgs
    x = range(0,100)
    for e in reversed(excess):
        del y[e]
        del x[e]
    plt.plot(x, y)
    plt.savefig('RuleType/Graphs/'+fn+".png")
    plt.clf()
    print "worked"

class Rankings:
    def __init__(self, discrete_labels_fn, ppmi_fn, phrases_fn, phrases_to_check_fn, fn):
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
        counter = 0
        with open(discrete_labels_fn) as f:
            for line in f:
                exists = True
                if phrases_to_check_fn != "":
                    exists = False
                    for i in indexes_to_get:
                        if i == counter:
                            exists = True
                            break
                if exists:
                    discrete_labels = line.split()
                    saveGraph(discrete_labels, ppmi[counter], fn + " " + phrases[counter][6:])
                    print phrases[counter]
                counter += 1




def main():

    discrete_labels_fn = "Rankings/films100P1.discrete"
    ppmi_fn = "filmdata/classesPhrases/nonbinary/class-all"
    phrases_fn = "SVMResults/films100.names"
    phrases_to_check_fn = ""#"RuleType/top1ksorted.txt"
    fn = "films 100"

    Rankings(discrete_labels_fn, ppmi_fn, phrases_fn, phrases_to_check_fn, fn)









if  __name__ =='__main__':main()