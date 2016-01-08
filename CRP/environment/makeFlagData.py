import itertools, json

def makeData():
    bridgeRate = 4
    corridorNum = 11
    corridorlength = 20
    flagNum = 2
    fromflag = 10

    bridgeLayers = list(range(bridgeRate, corridorlength, bridgeRate))
    itemLayers = {l + bridgeRate / 2:n for n, l in enumerate(bridgeLayers[:-1])}
    bridgeLayers = set(bridgeLayers)
    
    nodes = []
    edges = []
    fIndex = 1
    for n, (y, x) in enumerate(itertools.product(range(corridorlength), range(corridorNum))):
        flag = ""
        if y < corridorlength - 1:
            edges.append(((n, n + corridorNum), (x, y + 1) , 1))
        if y in bridgeLayers:
            if x < corridorNum - 1:
                edges.append(((n, n + 1), (x + 1, y) , 1))
            if x > 0 :
                edges.append(((n, n - 1), (x - 1, y) , 1))
        if y in itemLayers:
            if x - itemLayers[y] in [1, 4, 7]:
                flag = fIndex
                fIndex += 1
        nodes.append([n, ((x, y), flag)])
     
    graph_data = {}
    graph_data["nodes"] = nodes
    graph_data["actionNum"] = 4
    graph_data["goals"] = list(range((corridorNum * (corridorlength - 1)), (corridorNum * corridorlength)))
    graph_data["starts"] = list(range(corridorNum))
    graph_data["edges"] = edges
   
    return graph_data

