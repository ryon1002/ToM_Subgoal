from crp_core import CrpGibbs_PreCache
from environment import FlagMDP
import AlternativeModel

def makeAllResult(graph_data, prior, beta, gamma):
    result = {}
    tmp_graph_data = [(d[0], FlagMDP.FlagMDP(d[1], beta)) for d in graph_data]
    
    result[len(result)] = (AlternativeModel.logicalProbModel(tmp_graph_data), "LogicalPossiblity", "red")
    result[len(result)] = (AlternativeModel.independentModel(tmp_graph_data, prior), "Independent(beta=1, p=0.5)", "blue")
    result[len(result)] = (AlternativeModel.sumerizeProbs(AlternativeModel.copyModel(tmp_graph_data)), "Copy", "orange")

    crp2 = CrpGibbs_PreCache.CrpGibbs_PreCache(prior, gamma)
    result[len(result)] = (AlternativeModel.sumerizeProbs(crp2.gibbs(tmp_graph_data, 5000, 1000, True)), "CRP (beta=1, p=0.5, gamma=0.0001)", "purple")
    return result
