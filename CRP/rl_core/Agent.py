import numpy as np

class Agent():
    def __init__(self, mdp, choiceFunc=np.min):
        self.mdp = mdp
        self.choiceFunc = choiceFunc

    def init(self, start):
        self.state = start
        self.end = False

    def choiceDoAction(self):
        return self.choiceDoActionwithProb(self.policy[:, self.state])

    def choiceDoActionwithProb(self, probs):
        maxScore = np.amax(probs)
        aCandidate = np.where(probs == maxScore)[0]
        action = self.choiceFunc(aCandidate)
        self.state = self.mdp.getNextState(action, self.state)
        return (action, self.state)

    def getNextAction(self):
        if self.end:
            return (None, None)
        action, state = self._getNextAction()
        self.checkGoal()
        return action, state
    
    def _getNextAction(self):
        return choiceDoAction()
    
    def checkGoal(self):
        pass

    def getPath(self, start):
        return [s for s in self._getPath(start)]

    def _getPath(self, start):
        yield self.state
        while not self.end:
            yield self.getNextAction()[1]