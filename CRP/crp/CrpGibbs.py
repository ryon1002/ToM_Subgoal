import numpy as np
from collections import defaultdict

class CrpGibbs():
    def __init__(self, prior, gamma):
        self.prior = prior
        self.gamma = float(gamma)

    def gibbs(self, graph_data, repeat, burnIn, adjustTheta=False):
        self.xMap = {}
        self.i_xMap = {i:d for i, d in enumerate(graph_data)}
        self.xNum = len(graph_data)
        xIndexs = np.arange(self.xNum)
        
        self.preCalc()
        self.i_thetaMap = {v:k for k, v in self.thetaMap.viewitems()}
        self.thetas = np.arange(len(self.thetaMap)) 

        self.zs = np.zeros(self.xNum)
        self.zCount = np.zeros(self.xNum)
        self.zMap = {}
        self.initialize()

        self.aCount = defaultdict(int)
        
        for _n in range(burnIn):
            for i in range(self.xNum):
                self.reAssign(i)
        for _n in range(repeat):
            for i in range(self.xNum):
                self.reAssign(i)
            if adjustTheta:
                for z in set(self.zs):
                    self.drawGivenXList(z, xIndexs[np.where(self.zs == z)[0]])
            self.countCurrentStatus()

        self.aCount = {tuple([self.i_thetaMap[z] for z in k]):float(v) / repeat for k, v in self.aCount.viewitems()}
        
        return sorted(self.aCount.viewitems(), key=lambda x: x[1], reverse=True)
    
    def initialize(self):
        for i in range(self.xNum):
            self.zMap[i] = self.drawGivenX([i])
            self.assign(i, i)

    def assign(self, index, z):
        self.zs[index] = z
        self.zCount[z] += 1

    def reAssign(self, xIndex):
        self.zCount[self.zs[xIndex]] -= 1
        
        prob = np.zeros(len(self.drawIndexes))
        for z in set(self.zs):
            prob[z] = (self.zCount[z] / (self.xNum - 1 + self.gamma)) * self.getxGtheta(xIndex, self.zMap[z])
        self.setNewTableProb(prob, -1, xIndex)
        prob /= np.sum(prob)

        new_z = np.random.choice(self.drawIndexes, p=prob)
        if new_z >= self.xNum:
            new_theta = self.getNewTheta(new_z, xIndex)
            new_z = np.where(self.zCount == 0)[0][0]
            self.zMap[new_z] = new_theta
        self.assign(xIndex, new_z)
    
    def drawGivenXList(self, z, xIndexes):
        if len(xIndexes) == 1:
            self.zMap[z] = self.drawGivenX(xIndexes)
        else:
            self.zMap[z] = self.drawGivenXs(xIndexes)

    def countCurrentStatus(self):
        key = tuple([self.zMap[z] for z in self.zs])
        self.aCount[key] += 1

    def preCalc(self):
        self.thetaMap = defaultdict(lambda: len(self.thetaMap))
        self.drawIndexes = np.arange(self.xNum + 1)
        self.xGtheta = {}

    def drawGivenX(self, xIndex):
        raise NotImplementedError()
    
    def drawGivenXs(self, xIndexes):
        raise NotImplementedError()

    def setNewTableProb(self, prob, setIndexes, xIndex):
        raise NotImplementedError()
    
    def getNewTheta(self, new_z, xIndex):
        return self.drawGivenX([xIndex])
    
    def getxGtheta(self, xIndex, theta):
        if (xIndex, theta) not in self.xGtheta:
            self.xGtheta[(theta, xIndex)] = self.i_xMap[xIndex][1].getProbability(self.i_xMap[xIndex][0], self.i_thetaMap[theta])
        return self.xGtheta[(xIndex, theta)]
