import numpy as np
import itertools
from crp_core import FactorDist

class FlagFactor(FactorDist.FactorDist):
    def __init__(self, p, maxItemNum):
        self.pMap = {i:p ** i * (1 - p) ** (maxItemNum - i) for i in range(maxItemNum + 1)}

    def getThetasGivenX(self, (seq, mdp)):
        items = np.array(sorted({f for s, f in mdp.flagMap.viewitems() if s in seq}))
        thetas = [tuple(items[np.array(f)]) for f in itertools.product([True, False], repeat=len(items))]
        return thetas    

    def getThetaProb(self, theta):
        return self.pMap[len(theta)]

    def getPThetaGivenX(self, (seq, mdp), theta):
        goals = tuple([mdp.i_flagMap[th] for th in theta]) + (seq[-1],)
        return mdp.calcOneStateSeqProb(goals, seq)
