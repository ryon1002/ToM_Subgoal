import numpy as np
from crp import CrpGibbs
import itertools

class CrpGibbs_PreCache(CrpGibbs.CrpGibbs):
    def preCalc(self):
        CrpGibbs.CrpGibbs.preCalc(self)
        xDist = {}
        for xi, x in self.i_xMap.viewitems():
            items = np.array(sorted({f for s, f in x[1].flagMap.viewitems() if s in x[0]}))
            param = [items[np.array(f)] for f in itertools.product([True, False], repeat=len(items))]
            for p in param:
                xDist[(self.thetaMap[tuple(p)], xi)] = x[1].getProbability(x[0], p)
        self.theta0 = np.array([self.prior.getThetaProb(sg) for sg, _i in sorted(self.thetaMap.viewitems(), key=lambda x:x[1])])

        self.xGtheta = np.zeros((len(self.thetaMap), len(self.i_xMap)))
        for k, v in xDist.items():
            self.xGtheta[k] = v
        self.xGtheta /= np.sum(self.xGtheta, axis=0)

        self.thetaGx = self.xGtheta * self.theta0[:, np.newaxis]
        self.new_table_probs = (self.gamma / (self.xNum - 1 + self.gamma)) * np.sum(self.thetaGx, axis=0)
        self.thetaGx /= np.sum(self.thetaGx, axis=0)        
            
    def drawGivenX(self, xIndex):
        return np.random.choice(self.thetas, p=self.thetaGx[:, xIndex][:, 0])
    
    def drawGivenXs(self, xIndexes):
        prob = np.prod(self.xGtheta[:, xIndexes], axis=1) * self.theta0
        prob /= sum(prob)
        return np.random.choice(self.thetas, p=prob)

    def setNewTableProb(self, prob, setIndexes, xIndex):
        prob[setIndexes] = self.new_table_probs[xIndex]

    def getxGtheta(self, xIndex, theta):
        return self.xGtheta[theta, xIndex]
