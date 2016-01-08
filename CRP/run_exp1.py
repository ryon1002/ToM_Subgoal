import numpy as np
import pickle
import run_crp, CompareTool
from environment import FlagFactor

np.random.seed(1)

graph_data = pickle.load(open("data.pkl", "r"))
targetAgent = range(1, 23)
# targetAgent = [1]

factor = FlagFactor.FlagFactor(0.5, 3)
for n in targetAgent:
    print n
    CompareTool.checkStoreResult(run_crp.makeAllResult(graph_data[n], factor, 6, 0.013), "result/result" + str(n) + ".pkl", False, False)
