import pickle, itertools
from collections import defaultdict
from environment import FlagMDP
import numpy as np
import Agent
import random

def convertResultFormat(result, mdp, goal):
    return [(tuple([mdp.i_flagMap[s] for s in sb] + [goal]), p) for sb, p in result.items()]

# def convertSBProb(mdp, goal, probs):
#     ret = []
# #     print goal
#     denominator = sum(probs.viewvalues())
#     for sbs, prob in probs.viewitems():
#         convSbs = tuple(sorted([s for s, i in mdp.flagMap.viewitems() if i in sbs] + [goal]))
#         ret.append((convSbs, prob / denominator))
# #     print ret
#     return ret

def calcScore(a_path, h_path):
    score = 100 - 2 * len(a_path)
    return score

def runSupprot(mdp, gtSubgoal, crp_result):
    result = []
    roopNum = 22
    sbData = {210:0, 214:1, 217:2}
    
    subgoals = {0:{}, 1:{}, 2:{}, 3:{}, 4:{}, 5:{}}
    for g, sbIndex in sbData.viewitems():
        subgoals[0][g] = [((g,), 1)]
        for i in crp_result[sbIndex].keys():
            subgoals[i + 1][g] = convertResultFormat(crp_result[sbIndex][i][0], mdp, g)
        subgoals[5][g] = [(sg, 1) for sg in gtSubgoal[g]]
    
    debug_ret = []
    for i in subgoals.keys():
        score = 0
        com = Agent.Communication(mdp, subgoals[i])
        for _j in range(roopNum):
            for g in [210, 214, 217]:
                for sg in  gtSubgoal[g]:
                    # ret = com.run(random.randint(0, 11), 132, g, sg)
                    ret = com.run(_j % 11, 132, g, sg)
                    debug_ret.append(ret)
                    score += calcScore(*ret[1])
        result.append(float(score) / (roopNum * sum([len(v) for v in gtSubgoal.values()])))
    return np.array(result), debug_ret
