import numpy as np
from rl_core import MDP

class FlagMDP(MDP.MDP):
    def __init__(self, graph_data, beta):
        
        self.nodePos = {i:tuple(p[0]) for i, p in graph_data[u"nodes"]}
        self.numberSet = {i for i, p in graph_data[u"nodes"] if p[1] != ""}
        self.flagMap = {i:j[1] for i, j in graph_data["nodes"]if j[1] != ""}
        self.i_flagMap = {v:k for k, v in self.flagMap.viewitems()}
        self.beta = beta
        self.policyStore = {}
        MDP.MDP.__init__(self, len(self.nodePos), 4)
        
        edgeSet = {e[0] for e in graph_data[u"edges"]}
        for f in range(self.s):
            t = f + 11 if f + 11 < self.s else f
            self.setTR(0, f, t)
            t = f - 11 if f - 11 >= 0 else f
            self.setTR(3, f, t)
            t = f - 1 if (f, f - 1) in edgeSet else f
            self.setTR(1, f, t)
            t = f + 1 if (f, f + 1) in edgeSet else f
            self.setTR(2, f, t)
        
        self.org_t = self.t.copy()
        self.org_r = self.r.copy()
        self.starts = graph_data["starts"]
        self.goals = graph_data["goals"]
    
    def setTR(self, a, f, t):
        self.t[a, f, t] = 1
        if a == 3:
            self.r[a, f, t] = -2
        else:
            self.r[a, f, t] = -2
#         if a == 0 and t in self.numberSet:
#             if f not in self.numberSet:
#                 self.r[a, f, t] += 0
#         if t in self.numberSet:
#             if f not in self.numberSet:
#                 self.r[a, f, t] += 3

    def makePolicy(self, goal):
        if goal not in self.policyStore :
            self.policyStore[goal] = self._makePolicy(goal)
        return self.policyStore[goal]

    def _makePolicy(self, goal):
        self.t[:, goal, :] = 0
        self.t[:, goal, goal] = 1
        self.r[:, goal, :] = 0
        ret = self.makeSoftmaxPolicy(self.beta)
        self.t[:, goal, :] = self.org_t[:, goal, :]
        self.r[:, goal, :] = self.org_r[:, goal, :]
        return ret

    def calcOneStateSeqProb(self, goals, seq):
        prob = 1
        for i in range(len(goals)):
            g = goals[i]
            if g not in seq:
                return 0
            index = seq.index(g)
            prob *= self.calcOneStateSeqProb_1goal(g, seq[:index + 1], 0)
            seq = seq[index:]
        if len(seq) != 1:
            return 0
        return prob
    
    def calcOneStateSeqProb_1goal(self, goal, seq, reward):
        prob = 1
        policy = self.makePolicy(goal)
        for i in range(len(seq) - 1):
            trans = self.t[:, seq[i], seq[i + 1]]
            if np.sum(trans) == 0:
                return None
            action = np.argmax(trans)
            prob *= policy[action, seq[i]]
        return prob
