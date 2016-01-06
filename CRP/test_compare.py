import pickle

def compareResult(data1, data2):
    for a in data1.keys():
        for i in data1[a].viewkeys():
            sortdata1 = sorted(data1[a][i][0].viewitems())
            sortdata2 = sorted(data2[a][i][0].viewitems())
            if sortdata1 != sortdata2:
                print a, i
                print sortdata1
                print sortdata2
                return False
    return True

def check(targetAgent):
    modelData = {i:pickle.load(open("analize_1/result/result" + str(i) + ".pkl", "r")) for i in targetAgent}
    modelData2 = {i:pickle.load(open("result/result" + str(i) + ".pkl", "r")) for i in targetAgent}
    print compareResult(modelData, modelData2)

# check([9])
# check(range(1, 23))
