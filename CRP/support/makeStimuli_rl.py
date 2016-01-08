import numpy as np
import matplotlib.pyplot as plt
import itertools
from collections import defaultdict
import math, sys
import Agent

def addSeq(t, seq, nodeData, i_nodeData, edgeData):
    pos = nodeData[seq[-1]]
    
    while seq[-1] != i_nodeData[t] :
        step = np.sign(t[0] - pos[0])
        if step != 0 and (i_nodeData[pos], i_nodeData[(pos[0] + step, pos[1])]) in edgeData:
            pos = (pos[0] + step, pos[1])
        else:
            pos = (pos[0] , pos[1] + 1)
        seq.append(i_nodeData[pos])

def makePath(start, flags, goal, graph_data, nodeData, i_nodeData):
    edgeData = {i[0] for i in graph_data["edges"]}
    seq = [start]
    flags = [tuple(i) for i in zip(flags, [6, 10, 14])]
    for f in flags:
        addSeq(f, seq, nodeData, i_nodeData, edgeData)
    addSeq(goal, seq, nodeData, i_nodeData, edgeData)
    return seq

def getAllNextStep(current, layer, layerFlagData, flags, i_flagData):
    nextTarget = flags[min({k for k in flags.keys() if k > layer})]
    candidate = range(min(current, nextTarget), max(current, nextTarget) + 1)
    moveFlags = layerFlagData[layer].intersection(candidate)
    if layer in flags:
        if flags[layer] in moveFlags:
            opti = 1
#             if len(moveFlags) > 1:
#                 opti = 2
#             else:
#                 opti = 1
        else :
            if math.fabs(current - flags[layer]) > 1:
                opti = -2
            else: 
                opti = -1
        
#         opti = (nextTarget - flags[layer]) * (flags[layer] - current) >= 0
#         opti = 1 if opti else -1
        yield flags[layer], opti, i_flagData[(flags[layer], layer)]
    else:
        opti = 0
        if len(moveFlags) != 0:
            candidate = sorted(moveFlags)
            opti = 1
#             if len(moveFlags) > 1:
#                 opti = 2
#             else:
#                 opti = 1
        for i in candidate:
            yield i, opti, i_flagData[(i, layer)] if opti == 1 else -layer
            
def calcScore(seq, flags):
    l = sum([math.fabs(seq[i] - seq[i + 1]) for i in range(len(seq) - 1)])
    f = sum([i != 0 for i in flags])
    return int(2 * l - 3 * f)

def makePoints(layerFlagData, subgoals, i_flagData):
    for g in range(11):
        flags = subgoals.copy()
        flags[4] = g
        for l0 in range(11):
            for l1, o1, f1 in getAllNextStep(l0, 1, layerFlagData, flags, i_flagData):
                for l2, o2, f2 in getAllNextStep(l1, 2, layerFlagData, flags, i_flagData):
                    for l3, o3, f3 in getAllNextStep(l2, 3, layerFlagData, flags, i_flagData):
                        seq = (l0, l1, l2, l3, g)
                        o = (o1, o2, o3)
#                         yield seq, tuple([o[i - 1] for i in range(1, 4) if i in subgoals]), l0, g, (f1, f2, f3), calcScore(seq, o)
                        yield seq, o, l0, g, (f1, f2, f3), calcScore(seq, o)

def maxDifference(flags, experienceflag, experienceflagSet):
    tmpflags = tuple([pi for pi in flags if pi > 0])
    base = 1000 * len(tmpflags) if tuple(tmpflags) in experienceflagSet else 0    
    return base + sum([experienceflag[f] ** 2 for f in flags])

def makeLearnStimuli(mdp, graph_data, gtSubgoal, length):
    fullScinario = []
    agent = Agent.Worker(mdp)
    for _g, sgs in sorted(gtSubgoal.items()):
        scinario = []
        for sg in sgs:
            for i in np.random.choice(11, length, replace=False):
                scinario.append((agent.getPath(mdp, sg, i % 11), graph_data))
        fullScinario.append(scinario)
    return fullScinario
