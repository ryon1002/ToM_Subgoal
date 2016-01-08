import numpy as np

class FactorDist():
    def getDistThetaGivenX(self, x):
        thetas = self.getThetasGivenX(x)
        ret = ([tuple(t) for t in thetas], np.array([self.getThetaProb(t) * self.getPThetaGivenX(x, t) for t in thetas]))
        return (ret[0], ret[1] / np.sum(ret[1]))

    def getThetasGivenX(self, x):
        raise NotImplementedError()

    def getThetaProb(self, theta):
        raise NotImplementedError()

    def getPThetaGivenX(self, x, theta):
        raise NotImplementedError()
