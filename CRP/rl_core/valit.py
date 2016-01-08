import numpy as np

def normilization(arr, axis):
    normalization_coefficient = np.sum(arr, axis=axis)
    normalization_coefficient[np.where(normalization_coefficient == 0)] = 1
    return arr / normalization_coefficient

def valit(T, R, d=1):
    v = np.zeros(T.shape[1])
    for _i in range(500):
        t_n_v = np.sum(T * R + T * v * d, axis=2)
        n_v = np.max(t_n_v, axis=0)
        chk = np.sum(np.abs(n_v - v))
        if chk < 1e-6:
            break
        v = n_v
      
    return np.sum(T * R + T * n_v * d, axis=2)

def valit_with_invalidAction(oT, VA, R, d=1):
#     print T[0, 0, :]
#     print VA[0]
    v = np.zeros(oT.shape[1])
    T = oT.copy()
    T[VA.T == False] = 0
    for _i in range(500):
        t_n_v = np.sum(T * R + T * v * d, axis=2)
        n_v = np.zeros(len(t_n_v[0]))
        for i in range(len(t_n_v[0])):
            nextVal = t_n_v[:, i][VA[i]]
            if len(nextVal) > 0:
                n_v[i] = np.max(nextVal)
        chk = np.sum(np.abs(n_v - v))
        if chk < 1e-6:
            break
        v = n_v
#     print T * n_v
#     print T * R
#     print T * n_v + T * R
    return np.sum(T * R + T * n_v * d, axis=2)

def makeSoftmaxPolicy_with_invalidAction(T, VA, R, beta, d=1):
        val = valit_with_invalidAction(T, VA, R)
        exp = np.exp(val * beta)
        exp[VA.T == False] = 0
        return normilization(exp, 0)

def makeSoftmaxPolicy(T, R, beta, d=1):
        val = valit(T, R)
#         print val.T
        exp = np.exp(val * beta)
        return normilization(exp, 0)

def makeVal(T, R, beta, d=1):
        val = valit(T, R)
        return val
# #         print val.T
#         exp = np.exp(val * beta)
#         return exp
