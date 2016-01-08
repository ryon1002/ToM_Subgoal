# import makeStimuli_rl  # , run_crp, support
from support import makeStimuli_rl
import itertools, random, sys, os
import numpy as np

np.random.seed(1)
random.seed(1)

itemData = {1:{0:67, 1:70, 2:73}, 2:{0:112, 1:115, 2:118}, 3:{0:157, 1:160, 2:163}}
goalIndex = [210, 214, 217]

layer = ((2, 3),)
# layer = ((3,), (3,))
# layer = ((2,3), (2,3))
layer = ((3,),)
comb = list(itertools.permutations(range(3), 3))

def getOneSubgoal(layer):
    for sbIndexs in itertools.product(comb, repeat=sum([len(l) for l in layer])):
        subgoals = {}
        for gn, g in enumerate(goalIndex):
            subgoal = []
            count = 0
            for l in layer:
                sb = ()
                for li in l:
                    sb += (itemData[li][sbIndexs[count][gn]],)
                    count += 1
                subgoal.append(sb + (g,))
            subgoals[g] = tuple(subgoal)
        yield subgoals             

# tryNum = 6;
tryNum = 1;
# length = int(sys.argv[1]) if len(sys.argv) > 1 else 11;
length = 4
# for i in range(1, 12):
graph_data = makeFlagData.makeData()

for i in [length]:
    result = np.zeros((0, 6))
    for _t in range(tryNum):
        tmpResult = np.zeros((0, 6))
        for count, subgoal in enumerate(getOneSubgoal(layer)):
#             print subgoal
#             filename = "result" + str(i)
#             if not os.path.exists(filename):
#                 os.mkdir(filename)
            makeStimuli_rl.makeLearnStimuli(graph_data, subgoal, i)
#             # tochuu
#             break
            pickle.dump(run_crp.makeAllResult(stimuli[n], prior, 6, 0.013), open("tmp.pkl", "wb"))
#             run_flag_crp.runCRP(filename)
            tmpResult = np.r_[tmpResult, [support.runSupprot(subgoal, filename)]]
        result = np.r_[result, [np.average(tmpResult, axis=0)]]
    resulta = np.average(result, axis=0)
    print "{0:02d}".format(i), result.shape,
    for r in resulta : print r,
    result = np.var(result, axis=0)
    print 
    print "{0:02d}".format(i),
    for r in result : print r,
    print 
