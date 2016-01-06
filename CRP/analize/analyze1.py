import pickle, itertools
import numpy as np
import analyseCommon

modelData = {i:pickle.load(open("../result/result" + str(i) + ".pkl", "r")) for i in range(1, 23)}
key = {int(k):tuple(sorted(v[0][0].keys())) for k, v in modelData.viewitems()}
modelResult = analyseCommon.makeModelResult(modelData, key)

humanResult, humanResultTime = analyseCommon.getHumanResult("human-result.csv", key)
humanResult = analyseCommon.formatHumanResult(humanResult, humanResultTime, range(1, 23))
resultConcat = lambda x : np.array([i for i in itertools.chain(*[x[a] for a in range(1, 23)])])

import scipy.stats
print scipy.stats.pearsonr(resultConcat(humanResult), resultConcat(modelResult[0]))[0]
print scipy.stats.pearsonr(resultConcat(humanResult), resultConcat(modelResult[1]))[0]
print scipy.stats.pearsonr(resultConcat(humanResult), resultConcat(modelResult[2]))[0]
print scipy.stats.pearsonr(resultConcat(humanResult), resultConcat(modelResult[3]))[0]
