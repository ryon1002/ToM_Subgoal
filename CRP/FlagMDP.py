import numpy as np
import valit

class FlagMDP():
    def __init__(self, graph_data, beta):
        self.nodePos = {i:tuple(p[0]) for i, p in graph_data[u"nodes"]}
        self.numberSet = {i for i, p in graph_data[u"nodes"] if p[1] != ""}
        self.beta = beta
        self.policyMap = {}
        self.s = len(self.nodePos)
        self.a = 4
        self.t = np.zeros((self.a, self.s, self.s))
        self.r = np.zeros_like(self.t)
        self.flagMap = {i:j[1] for i, j in graph_data["nodes"]if j[1] != ""}
        self.i_flagMap = {v:k for k, v in self.flagMap.viewitems()}
        
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
#         va = self.va.copy()
#         va[goal, :] = 0
#         return valit.makeSoftmaxPolicy_with_invalidAction(self.t, va, self.r, self.beta)
        t = self.t.copy()
        t[:, goal, :] = 0
        t[:, goal, goal] = 1
        r = self.r.copy()
        r[:, goal, :] = 0
        return valit.makeSoftmaxPolicy(t, r, self.beta)

    def makePolicy_r(self, goal, reward):
        t = self.t.copy()
        t[:, goal, :] = 0
        t[:, goal, goal] = 1
        r = self.r.copy()
        r[:, :, goal] -= reward
        r[:, goal, :] = 0
        return valit.makeSoftmaxPolicy(t, r, self.beta)

    def makePolicy2(self, goal):
        va = self.va.copy()
        va[goal, :] = 0
        
        search = goal
        
        while len(np.where(self.t[1, :, search] > 0)[0]) + len(np.where(self.t[2, :, search] > 0)[0]) == 0:
            search = np.where(self.t[0, :, search] > 0)[0][0]
        va[search, [1, 2]] = 0
        searchT = search
        while len(np.where(self.t[1, :, searchT] > 0)[0]) == 1:
            searchT = np.where(self.t[1, :, searchT] > 0)[0][0]
            va[searchT, [0, 2]] = 0
        searchT = search
        while len(np.where(self.t[2, :, searchT] > 0)[0]) == 1:
            searchT = np.where(self.t[2, :, searchT] > 0)[0][0]
            va[searchT, [0, 1]] = 0
#             print np.where(self.t[1, :, search] > 0)
        for a, s in zip(*np.where(self.t[:, :, goal] > 0)):
            f = np.ones(self.a, dtype=np.bool)
            f[a] = False
            va[s, f] = False
        return valit.makeSoftmaxPolicy_with_invalidAction(self.t, va, self.r, self.beta)

    def makePolicy3(self, goal):
        va = self.va.copy()
        va[goal, :] = 0
        r = self.r.copy()
        r[:, :, goal] = 10
        for a, s in zip(*np.where(self.t[:, :, goal] > 0)): 
            f = np.ones(self.a, dtype=np.bool)
            f[a] = False
            va[s, f] = False
        return valit.makeSoftmaxPolicy_with_invalidAction(self.t, va, self.r, self.beta)

    def calcOneStateSeqProb(self, goals, seq):
        prob = 1
        for i in range(len(goals)):
            g = goals[i]
            if g not in seq:
                return 0
            index = seq.index(g)
            minDist = self.makeMinDist(goals[i:])
            prob *= self.calcOneStateSeqProb_1goal(g, seq[:index + 1], minDist * 2)
            seq = seq[index:]
        if len(seq) != 1:
            return 0
        return prob

    def calcOneStateSeqProb2(self, goals, seq):
        prob = 1
        for i in range(len(goals)):
            g = goals[i]
            if g not in seq:
                index = len(seq)
            else:
                index = seq.index(g)
            minDist = self.makeMinDist(goals[i:])
            prob *= self.calcOneStateSeqProb_1goal(g, seq[:index + 1], minDist * 2)
            seq = seq[index:]
        return prob
    
    def getProbability(self, seq, theta):
        goals = tuple([self.i_flagMap[th] for th in theta]) + (seq[-1],)
        return self.calcOneStateSeqProb(goals, seq)
    
    def makeMinDist(self, goals):
        return 0
        ret = 0        
        current = self.nodePos[goals[0]]
        for i in range(1, len(goals)):
            nextpos = self.nodePos[goals[i]]
            ret += abs(current[0] - nextpos[0]) + abs(current[1] - nextpos[1])
            current = next
        return ret

    def calcOneStateSeqProb_1goal(self, goal, seq, reward):
        if (goal, reward) not in self.policyMap:
            self.policyMap[(goal, reward)] = self.makePolicy_r(goal, reward)
        prob = 1
        policy = self.policyMap[(goal, reward)]
        for i in range(len(seq) - 1):
            trans = self.t[:, seq[i], seq[i + 1]]
            if np.sum(trans) == 0:
                return None
            action = np.argmax(trans)
            prob *= policy[action, seq[i]]
        return prob

    def calcOneActionSeqProb(self, policy, seq, state, state_prob):
        if len(seq) == 0 :
            return state_prob
        if policy[seq[0], state] == 0:
            return 0
        aProb = state_prob * policy[seq[0], state]
        next_s_Prob = self.t[seq[0], state] * aProb
        return sum(self.calcOneActionSeqProb(policy, seq[1:], ns[0], next_s_Prob[ns[0]]) for ns in np.where(next_s_Prob != 0))

    def sampleAction(self, policy, state):
        if np.sum(policy[:, state]) == 0:
            return None, None
        action = np.random.choice(np.arange(len(policy)), p=policy[:, state])
        return action, self.getNextState(action, state)

    def getNextState(self, action, state):
        return np.random.choice(np.arange(len(self.t[0])), p=self.t[action, state])
