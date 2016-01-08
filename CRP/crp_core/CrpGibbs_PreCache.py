import numpy as np
from crp_core import CrpGibbs
import itertools

class CrpGibbs_PreCache(CrpGibbs.CrpGibbs):
    def preCalc(self):
        CrpGibbs.CrpGibbs.preCalc(self)
        xDist = {}
        for xi, x in self.i_xMap.viewitems():
            self.thetas = self.factor.getThetasGivenX(x)
            for theta in self.thetas:
                xDist[(self.thetaMap[tuple(theta)], xi)] = self.factor.getPThetaGivenX(x, theta)
        self.theta0 = np.array([self.factor.getThetaProb(sg) for sg, _i in sorted(self.thetaMap.viewitems(), key=lambda x:x[1])])

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
