import numpy as np
import pickle, itertools, sys
from support import makeStimuli_rl, support
from environment import makeFlagData, FlagMDP, FlagFactor
import run_crp, CompareTool

np.random.seed(1)
np.set_printoptions(edgeitems=3200, linewidth=1000, precision=6)

layer = ((3,),)
# layer = ((2, 3),)
# layer = ((3,), (3,))
# layer = ((2, 3), (2, 3))
comb = list(itertools.permutations(range(3), 3))
itemData = {1:{0:67, 1:70, 2:73}, 2:{0:112, 1:115, 2:118}, 3:{0:157, 1:160, 2:163}}
goalIndex = [210, 214, 217]
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

import datetime
measure_start_time = datetime.datetime.now()

subgoals = []
subgoals.append({217: ((163, 217),), 210: ((157, 210),), 214: ((160, 214),)})
subgoals.append({217: ((163, 217), (163, 217)), 210: ((157, 210), (157, 210)), 214: ((160, 214), (160, 214))})
subgoals.append({217: ((118, 163, 217),), 210: ((112, 157, 210),), 214: ((115, 160, 214),)})
# subgoals = {n:i for n, i in enumerate(getOneSubgoal(layer))}
# subgoals[0] = {217: ((163, 217), (157, 217)), 210: ((157, 210), (160, 210)), 214: ((160, 214), (163, 214))}
input_num = int(sys.argv[1]) if len(sys.argv) > 1 else 11;

graph_data = makeFlagData.makeData()
mdp = FlagMDP.FlagMDP(graph_data, 2)
factor = FlagFactor.FlagFactor(0.5, 3)

def printResult(result, sumFunc, preStr):
    ret = sumFunc(result, axis=0)
    print preStr + "{0:02d}".format(input_num),
    for r in ret : print r,
    print

tryNum = 2
np.random.seed(1)
result = np.zeros((0, 6))
for _t in range(tryNum):
    for i, sb in enumerate(subgoals):
        tmpResult = np.zeros((0, 6))
        stimuli = CompareTool.checkStoreResult(makeStimuli_rl.makeLearnStimuli(mdp, graph_data, sb, input_num), "test_stimulus.pkl", False, False)
        crp_results = CompareTool.checkStoreResult({n:run_crp.makeAllResult(stimuli[n], factor, 6, 0.013) for n in range(len(stimuli))}, "test_crp_" + str(i) + ".pkl", False, False)
#         prev_seq = pickle.load(open("test_support_" + str(i) + ".pkl", "r"))
        score, seq = support.runSupprot(mdp, sb, crp_results)
#         pickle.dump(seq, open("test_support_" + str(i) + ".pkl", "w"))
        tmpResult = np.r_[tmpResult, [score]]
#         print prev_seq == seq
    result = np.r_[result, [np.average(tmpResult, axis=0)]]
printResult(result, np.average, "s")
printResult(result, np.var, "v")

print datetime.datetime.now() - measure_start_time
