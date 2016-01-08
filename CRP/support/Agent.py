import numpy as np
from collections import defaultdict
import itertools
from rl_core import Agent

class AgentBase(Agent.Agent):
    def __init__(self, mdp):
        Agent.Agent.__init__(self, mdp, np.random.choice)
        self.policies = {}
        self.target = {157:0, 160:1, 163:2}
        self.i_target = {v:k for k, v in self.target.viewitems()}
        self.goals = {210:0, 214:1, 217:2}
        self.i_goals = {v:k for k, v in self.goals.viewitems()}
    
class Worker(AgentBase):
    def init(self, start, subgoals):
        Agent.Agent.init(self, start)
        self.subgoalList = list(subgoals)
        self.setGoal(self.subgoalList[0])
    
    def setGoal(self, goal):
        if goal not in self.policies:
            self.policies[goal] = self.mdp.makePolicy(goal)
        self.policy = self.policies[goal]
        
    def removeSubgoal(self, subgoal):
        if subgoal in self.subgoalList:
            self.subgoalList.remove(subgoal)
            self.setGoal(self.subgoalList[0])
    
    def _getNextAction(self):
        self.last_state = self.state
        self.last_action, self.state = self.choiceDoAction()
        return (self.last_action, self.state)
    
    def checkGoal(self):
        if self.subgoalList[0] == self.state:
            self.subgoalList.pop(0)
            if len(self.subgoalList) == 0:
                self.end = True
                return 
            self.setGoal(self.subgoalList[0])

    def getPath(self, mdp, subgoals, start):
        self.init(start, subgoals)
        return Agent.Agent.getPath(self, start)

class Helper(AgentBase):
    def __init__(self, mdp, subgoals):
        AgentBase.__init__(self, mdp)
        self.orgSubgoals = subgoals.copy()
        self.makeGoalTargetMap()
        self.decisionPoint = 0.7
        self.orgPolicies = self.makePolicys(subgoals)
        self.nsgpolicies = np.array([self.mdp.makePolicy(g) for g in sorted(self.goals.viewkeys())])
    
    def init(self, start, worker, goal):
        Agent.Agent.init(self, start)
        self.worker = worker
        self.subgoals = self.orgSubgoals.copy()
        self.agentGoalProb = np.ones(3) / 3
        self.mode = 0;
        self.helpertarget = None
        self.policies = self.orgPolicies.copy()
        
        self.goal = goal
    
    def makeGoalTargetMap(self):
        self.goalTargetProb = np.zeros((len(self.goals), len(self.target)))
        for g, v in self.orgSubgoals.items():
            for sb, p in v:
                for t in self.target.viewkeys():                  
                    if t in sb:
                        self.goalTargetProb[self.goals[g]][self.target[t]] += p
        self.validTarget = True if np.sum(self.goalTargetProb) else False

    def makePolicys(self, subgoals):
        policys = []
        for g in sorted(self.goals.keys()):
            policys.append(np.sum(np.array([self.mdp.makePolicy(sbs[0]) * p for sbs, p in subgoals[g]]), axis=0))
        return  np.array(policys)
    
    def removeSubgoal(self, subgoals, goal):
        ret = {}
        for g, sl in subgoals.viewitems():
            ret[g] = [(tuple([sg for sg in sb if sg != goal]), p) for sb, p in sl]
        return ret
    
    def getGoalWeight(self):
        if self.worker.end:
            prob = np.zeros(len(self.goals))
            prob[self.goals[self.worker.state]] = 1
            return prob
        self.agentGoalProb *= self.policies[:, self.worker.last_action, self.worker.last_state]
        self.agentGoalProb /= np.sum(self.agentGoalProb)
        return self.agentGoalProb
    
    def _getNextAction(self):
        gProb = self.getGoalWeight()
#         if len(self.goalTargetMap) > 0:
#             return (None, None)
#         print gProb
        if self.mode == 0:
            if self.validTarget:
                prob = np.sum(self.goalTargetProb.T * gProb, axis=1)
                prob /= np.sum(prob)
                if np.max(prob) > self.decisionPoint:
                    self.helpertarget = self.i_target[np.argmax(prob)]
                    self.subgoals = self.removeSubgoal(self.subgoals, self.helpertarget)
                    self.policies = self.makePolicys(self.subgoals)
                    self.policy = self.mdp.makePolicy(self.helpertarget)
                    self.worker.removeSubgoal(self.helpertarget)
                    self.mode = 1
                    return self.choiceDoAction()
        elif self.mode == 1:
            action, _state = self.choiceDoAction()
            if self.state == self.helpertarget:
                self.policies = self.nsgpolicies.copy()
                self.mode = 2
            return (action, _state)
        else:
            lastPolicy = np.sum(self.nsgpolicies[:, :, self.state].T * gProb, axis=1)
            return self.choiceDoActionwithProb(lastPolicy)
        return (None, None)
    
    def checkGoal(self):
        self.end = self.mode == 2 and self.state == self.goal
    
    def isValid(self):
        return not self.end and self.mode != 0

class Communication():
    def __init__(self, mdp, subgoals):
        self.worker = Worker(mdp)
        self.helper = Helper(mdp, subgoals)
    
    def run(self, a_s, h_s, goal, subgoals):
        self.worker.init(a_s, subgoals)
        self.helper.init(h_s, self.worker, goal)
        
        self.a_path = [a_s]
        self.h_path = [h_s]
        while True:
            if not self.worker.end:
                self.a_path.append(self.worker.getNextAction()[1])
            if not self.helper.end:
                _h_a, h_s = self.helper.getNextAction()
                if h_s is not None:
                    self.h_path.append(h_s)
            if self.worker.end and not self.helper.isValid(): break
        return (len(self.a_path), (self.a_path, self.h_path))
