import numpy as np
import pickle
from crp import CrpGibbs_PreCache
import FlagSampler, FlagMDP
import AlternativeModel

# import datetime
# st = datetime.datetime.now()

np.random.seed(1)
np.set_printoptions(edgeitems=3200, linewidth=1000, precision=3)

beta = 6
prior = FlagSampler.FlagSampler(0.5, 3)
gamma = 0.013
graph_data = pickle.load(open("data.pkl", "r"))
targetAgent = range(1, 23)
# targetAgent = [1]

for n in targetAgent:
    print n
    result = {}
    tmp_graph_data = [(d[0], FlagMDP.FlagMDP(d[1], beta)) for d in graph_data[n]]
    
    result[len(result)] = (AlternativeModel.logicalProbModel(tmp_graph_data), "LogicalPossiblity", "red")
    result[len(result)] = (AlternativeModel.independentModel(tmp_graph_data, prior), "Independent(beta=1, p=0.5)", "blue")
    result[len(result)] = (AlternativeModel.sumerizeProbs(AlternativeModel.copyModel(tmp_graph_data)), "Copy", "orange")

    crp2 = CrpGibbs_PreCache.CrpGibbs_PreCache(prior, gamma)
    result[len(result)] = (AlternativeModel.sumerizeProbs(crp2.gibbs(tmp_graph_data, 5000, 1000, True)), "CRP (beta=1, p=0.5, gamma=0.0001)", "purple")
    pickle.dump(result, open("result/result" + str(n) + ".pkl", "wb"))

# print datetime.datetime.now() - st
# 
# import tmp_compare
# tmp_compare.check(targetAgent)


