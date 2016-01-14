import numpy as np
import pickle, itertools, sys
from support import makeStimuli_rl, support
from environment import makeFlagData, FlagMDP, FlagFactor
import run_crp, CompareTool

np.random.seed(1)
np.set_printoptions(edgeitems=3200, linewidth=1000, precision=6)

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

input_num = int(sys.argv[1]) if len(sys.argv) > 1 else 11;

layers = [((3,),), ((2, 3),), ((3,), (3,))]
# layers = [((3,),)]
sbGens = [getOneSubgoal(l) for l in layers]
subgoals = [sb for sb in itertools.chain(*sbGens)]

graph_data = makeFlagData.makeData()
mdp = FlagMDP.FlagMDP(graph_data, 2)
factor = FlagFactor.FlagFactor(0.5, 3)

def printResult(result, sumFunc, preStr):
    ret = sumFunc(result, axis=0)
    print preStr + "{0:02d}".format(input_num),
    for r in ret : print r,
    print

tryNum = 6
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
