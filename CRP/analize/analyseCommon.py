from collections import defaultdict
import datetime, json, csv
import numpy as np

def getHumanResult(filename, key):
    humanResult = defaultdict(lambda:defaultdict(dict))
    humanResultTime = defaultdict(lambda:defaultdict(dict))
    for user_id, agent, data, timestamp in csv.reader(open(filename)):
        timestamp = datetime.datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        if timestamp > datetime.datetime(2015, 8, 25):
            agent = int(agent)
            if agent > 0:
                data = json.loads(data)
                result = np.zeros(len(key[agent]))
                if len(data) == 0:
                    result[key[agent].index(())] = 1
                else:
                    for sb in set([tuple(i) for i in data]):
                        if sb in key[agent]:
                            result[key[agent].index(sb)] = 1
                humanResult[user_id][agent] = result
                humanResultTime[user_id][agent] = timestamp
    return humanResult, humanResultTime

def formatHumanResult(result, resultTime, agentList):
    humanResult = {k:v for k, v in result.items() if len(v) == 22}
    
    for u in humanResult.keys():
        order = [k for k, _v in sorted(resultTime[u].items(), key=lambda x : x[1])]
        count = 0
        for a in order:
            count += np.sum(humanResult[u][a])
            del humanResult[u][a]
            if count >= 4 : break
    
    humanResult1 = {}
    agentCount = defaultdict(int)
    for a in agentList:
        for score in humanResult.values():
            if a in score:
                if a not in humanResult1:
                    humanResult1[a] = score[a]
                else:
                    humanResult1[a] += score[a]
                agentCount[a] += 1
    humanResult1 = {k:v / agentCount[k] for k, v in humanResult1.items()}
    return humanResult1

def makeModelResult(modelData, key):
    modelResult = {}
    for m in range(len(modelData[1])):
        aResult = {}
        for a, data in modelData.viewitems():
            result = np.zeros(len(key[a]))
            for sb, p in data[m][0].viewitems():
                result[key[a].index(sb)] = p    
            aResult[a] = result
        modelResult[m] = aResult
    return modelResult

def sortKey(key, modelData):
    retKey = {}
    for k, v in key.items():
        keyScore = defaultdict(float)
        for i in v:
            for m in range(len(modelData[k])):
                keyScore[i] += modelData[k][m][0][i]
        retKey[k] = [i[0] for i in sorted(keyScore.items(), key=lambda x : x[1], reverse=True)]
    return retKey