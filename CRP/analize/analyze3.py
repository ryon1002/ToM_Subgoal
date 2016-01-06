import pickle, pylab
import numpy as np
import graph, analyseCommon
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages 
import matplotlib.gridspec as gridspec

modelData = {i:pickle.load(open("../result/result" + str(i) + ".pkl", "r")) for i in range(1, 23)}
key = {int(k):tuple(sorted(v[0][0].keys())) for k, v in modelData.viewitems()}
key = analyseCommon.sortKey(key, modelData)
modelResult = analyseCommon.makeModelResult(modelData, key)

def barProt(data, start, end, key):
    xbar = np.arange(len(key))[start:end] - start
    plt.xlim([-1, len(xbar)])
    plt.ylim([0, 1])
#     plt.xticks(xbar, key[start:end], fontsize=34)
    plt.xticks(xbar, key[start:end], fontsize=20)
    plt.yticks(fontsize=20)
    plt.bar(xbar - 0.36 , data[0][start:end], color="y", width=0.18, align="center", label="Human")
    plt.bar(xbar - 0.18 , data[4][start:end], color="r", width=0.18, align="center", label="CRP Model")
    plt.bar(xbar - 0.0 , data[2][start:end], color="b", width=0.18, align="center", label="Independent")
    plt.bar(xbar + 0.18 , data[1][start:end], color="g", width=0.18, align="center", label="Logical Possibility")
    plt.bar(xbar + 0.36 , data[3][start:end], color="orange", width=0.18, align="center", label="Copy")

agentList = range(1, 23)
# agentList = [9]

humanResult, humanResultTime = analyseCommon.getHumanResult("human-result.csv", key)
humanResult = analyseCommon.formatHumanResult(humanResult, humanResultTime, agentList)
graph_data = pickle.load(open("../data.pkl", "r"))

for a in agentList:
    G = gridspec.GridSpec(17, 4)
    pp = PdfPages("result_graph/exp1_agent_" + str(a) + ".pdf")
    pylab.figure(figsize=(24, 22))
    for n, data in enumerate(graph_data[a]):
        plt.subplot(G[(n / 4) * 5:5 + (n / 4) * 5, n % 4])
        graph.drawGraph2(data[1], data[0], "path")
    data = {k + 1:v[a] for k, v in modelResult.viewitems()}
    data[0] = humanResult[a]
    plt.subplot(G[10:13, :])
    midPoint = len(key[a]) / 2
    barProt(data, 0, midPoint, key[a])
    plt.subplot(G[14:17, :])
    barProt(data, midPoint, len(key[a]) + 1, key[a])
    pylab.legend(ncol=2, fontsize=34)
    pp.savefig(bbox_inches="tight", pad_inches=0.05)
    pp.close()
