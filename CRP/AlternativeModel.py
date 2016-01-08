import numpy as np
from collections import defaultdict
import itertools

def sumerizeProbs(probs):
    ret = defaultdict(int)
    for sbs, p in probs:
        for sb in set(sbs):
            ret[sb] += p
    return ret

def sumerizeProbs2(dists):
    prob = {}
    dists = [{sg:p for sg, p in dist} for dist in dists]
    subgoalSet = set()
    for dist in dists:
        subgoalSet.update([sg for sg in dist.viewkeys()])
     
    for sg in subgoalSet:
        p = 0
        prev_p = 1
        for dist in dists:
            if sg in dist:
                p += dist[sg] * prev_p
                prev_p *= (1 - dist[sg])
        prob[sg] = p
    return prob

def independentModel(graph_data, factor):
    dists = []
    for d in graph_data:
        dist = factor.getDistThetaGivenX(d)
        dists.append(zip(*dist))
    return sumerizeProbs2(dists)

def copyModel(graph_data):
    subgoals = []
    for data in graph_data:
        subgoals.append(tuple(sorted({f for s, f in data[1].flagMap.viewitems() if s in data[0]})))
    return [(tuple(subgoals), 1.0)]

def logicalProbModel(graph_data):
    prob = defaultdict(float)
    for d in graph_data:
        items = np.array(sorted({f for s, f in d[1].flagMap.viewitems() if s in d[0]}))
        param = [tuple(items[np.array(f)]) for f in itertools.product([True, False], repeat=len(items))]
        for p in param:
            prob[p] += 1
    prob = {k:v / len(graph_data) for k, v in prob.viewitems()}
    return prob
