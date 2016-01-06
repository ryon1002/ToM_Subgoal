import numpy as np
import itertools

class FlagSampler():
    def __init__(self, p, maxItemNum):
        self.pMap = {i:p ** i * (1 - p) ** (maxItemNum - i) for i in range(maxItemNum + 1)}
#     
    def getParamDist(self, seq, mdp):
        items = np.array(sorted({f for s, f in mdp.flagMap.viewitems() if s in seq}))
        param = [items[np.array(f)] for f in itertools.product([True, False], repeat=len(items))]
        return ([tuple(p) for p in param], [self.pMap[len(p)] for p in param])

    def getThetaProb(self, theta):
        return self.pMap[len(theta)]

    def draw(self, seq):
        sample = []
        for s in seq[1:-1]:
            if np.random.sample() < self.p:
                sample.append(s)
        return tuple(sample)
