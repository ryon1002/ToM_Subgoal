import numpy as np

class MDP():
    def __init__(self, s, a):
        self.s = s
        self.a = a
        self.t = np.zeros((self.a, self.s, self.s))
        self.r = np.zeros_like(self.t)

    def makeSoftmaxPolicy(self, beta, d=1):
        val = self.valueIteration(d)
        exp = np.exp(val * beta)
        return self.normilization(exp, 0)

    def valueIteration(self, d):
        v = np.zeros(self.t.shape[1])
        for _i in range(500):
            t_n_v = np.sum(self.t * self.r + self.t * v * d, axis=2)
            n_v = np.max(t_n_v, axis=0)
            chk = np.sum(np.abs(n_v - v))
            if chk < 1e-6:
                break
            v = n_v
        return np.sum(self.t * self.r + self.t * n_v * d, axis=2)

#     def sampleAction(self, policy, state):
#         if np.sum(policy[:, state]) == 0:
#             return None, None
#         action = np.random.choice(np.arange(len(policy)), p=policy[:, state])
#         return action, self.getNextState(action, state)

    def getNextState(self, action, state):
        return np.random.choice(np.arange(len(self.t[0])), p=self.t[action, state])

    def normilization(self, arr, axis):
        normalization_coefficient = np.sum(arr, axis=axis)
        normalization_coefficient[np.where(normalization_coefficient == 0)] = 1
        return arr / normalization_coefficient
