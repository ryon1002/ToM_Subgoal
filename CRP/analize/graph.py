import networkx as nx
import json, itertools
import pylab
import matplotlib.pyplot as plt

colorMap = {0:"w", 1:"purple", 2:"cyan", 3:"orange"}

def drawGraph(graph_data, seq, type):
    G = nx.Graph()
     
    nodeData = {i:tuple(p[0]) for i, p in graph_data[u"nodes"]}
#     nodeColor = [colorMap[p[1]] for i, p in graph_data[u"nodes"]]
#     colorMap = {"g":[], "b":[]}
#     nodeData = {"g":{}, "b":{}}
#     for i, p in graph_data[u"nodes"]:
#         nodeData[p[1]][i] = tuple(p[0])
#         colorMap[p[1]].append(i)
#     
#     G.add_nodes_from(nodeData["b"].keys())
#     G.add_nodes_from(nodeData["g"].keys())
    
    for (f, t), _f, _t in graph_data[u"edges"] :
        G.add_edge(f, t)
    
    nx.draw_networkx_nodes(G, nodeData, node_color="w")
    nodeLabel = {i:p[1] for i, p in graph_data[u"nodes"]}
    nx.draw_networkx_labels(G, nodeData, nodeLabel, font_weight="bold")
#     nx.draw_networkx_nodes(G, nodeData, node_list=colorMap["g"], node_color="g")
    nx.draw_networkx_edges(G, nodeData)

    if type == "path":
#         pathedge = [(seq[l], seq[l + 1]) for l in range(len(seq) - 1) if (seq[l], seq[l + 1]) in graph_data[u"edges"]]
        pathedge = [(seq[l], seq[l + 1]) for l in range(len(seq) - 1)]
#         nx.draw_networkx_nodes(G, nodeData, nodelist=seq, node_color="r", linewidths=4)
        nx.draw_networkx_nodes(G, nodeData, nodelist=seq, node_color="cyan")
        nx.draw_networkx_edges(G, nodeData, edgelist=pathedge, edge_color="cyan", width=4)
    elif type == "subgoal":
        nx.draw_networkx_nodes(G, nodeData, nodelist=seq, node_color="g", linewidths=3)

    # plt.savefig("test_fig.png")
    plt.axis('off') 

def drawGraph2(graph_data, seq, type, goals=None):
    G = nx.Graph()
    nodeSize = 40
    nodeData = {i:tuple(p[0]) for i, p in graph_data[u"nodes"]}
#     nodeColor = [colorMap[p[1]] for i, p in graph_data[u"nodes"]]
#     colorMap = {"g":[], "b":[]}
#     nodeData = {"g":{}, "b":{}}
#     for i, p in graph_data[u"nodes"]:
#         nodeData[p[1]][i] = tuple(p[0])
#         colorMap[p[1]].append(i)
#     
#     G.add_nodes_from(nodeData["b"].keys())
#     G.add_nodes_from(nodeData["g"].keys())
    
    for (f, t), _f, _t in graph_data[u"edges"] :
        G.add_edge(f, t)
    
    flags = [i[0] for i in graph_data["nodes"] if i[1][1] != ""]
#     flagPos = {i[0]:i[1][0] for i in graph_data["nodes"] if i[1][1] != "" and i[0] in seq}
#     print nodeData
#     print flagPos
    nodes = nx.draw_networkx_nodes(G, nodeData, node_color="k", node_size=nodeSize, linewidths=0)
    nodes.set_edgecolor('k')
    nodes = nx.draw_networkx_nodes(G, nodeData, nodelist=flags, node_color="w", node_size=nodeSize + 150, linewidths=0)
    nodes.set_edgecolor('w')
    nodeLabel = {i:p[1] for i, p in graph_data[u"nodes"] if i in seq and i in flags}
    nodeLabel2 = {i:p[1] for i, p in graph_data[u"nodes"] if i not in seq and i in flags}
    nx.draw_networkx_labels(G, nodeData, nodeLabel, font_color="#FF3030", font_size=28, font_weight="bold")
    nx.draw_networkx_labels(G, nodeData, nodeLabel2, font_size=28, font_weight="bold")
    
#     nx.draw_networkx_nodes(G, nodeData, node_list=colorMap["g"], node_color="g")
    nx.draw_networkx_edges(G, nodeData, edge_color="k")

    if type == "path":
#         pathedge = [(seq[l], seq[l + 1]) for l in range(len(seq) - 1) if (seq[l], seq[l + 1]) in graph_data[u"edges"]]
        pathedge = [(seq[l], seq[l + 1]) for l in range(len(seq) - 1)]
#         nx.draw_networkx_nodes(G, nodeData, nodelist=seq, node_color="r", linewidths=4)
        seq_no_flag = set(seq).difference(set(flags))
        nodes = nx.draw_networkx_nodes(G, nodeData, nodelist=seq_no_flag, node_color="#FF3030", node_size=nodeSize + 30, linewidths=0)
        nodes.set_edgecolor('#FF3030')
#         seq_flag = set(seq).intersection(set(flags))
#         nx.draw_networkx_labels(G, nodeData, nodeLabel, nodelist=seq_flag , font_color="r")
        pathedge1 = [i for i in pathedge if i[0] not in nodeLabel and i[1] not in nodeLabel]
        pathedge2 = [i for i in pathedge if i[0]  in nodeLabel or i[1]  in nodeLabel]
        nx.draw_networkx_edges(G, nodeData, edgelist=pathedge1, width=2, edge_color="#FF3030")
        nx.draw_networkx_edges(G, nodeData, edgelist=pathedge2, width=2, edge_color="w")

        nx.draw_networkx_labels(G, nodeData, {seq[-1]:"G"}, font_color="#FF3030", font_size=28, font_weight="bold")
        nodes = nx.draw_networkx_nodes(G, nodeData, nodelist=[seq[-1]], node_color="w", node_size=nodeSize + 150, linewidths=0)
        nodes.set_edgecolor('w')            
        
    elif type == "subgoal":
        nx.draw_networkx_nodes(G, nodeData, nodelist=seq, node_color="g", linewidths=3)
    
    plt.xlim(-1, max([v[0] for v in nodeData.viewvalues()]) + 1)
    plt.ylim(-1, max([v[1] for v in nodeData.viewvalues()]) + 1)
    # plt.savefig("test_fig.png")
    plt.axis('off') 

def plot(paths, graph_data, estimate, gamma, p):
    for n, path in enumerate(paths):
        plt.subplot(5, 5, n + 1)
        drawGraph(graph_data, path, "path")
    print estimate[1]
    for n, sgs in enumerate(estimate[1][:3]):
        for i, sg in enumerate(sgs[0]):
            plt.subplot(5, 5, n * 5 + i + 6)
            drawGraph(graph_data, sg, "subgoal")
            if i == 0: 
                plt.title(round(sgs[1], 4))
    for i, sg in enumerate(estimate[2]):
        plt.subplot(5, 5, i + 21)
        drawGraph(graph_data, sg[0], "subgoal")
        plt.title(round(sg[1], 4))
#     for n, sg in enumerate(set(estimate[0])):
#         plt.subplot(5, 5, n + 21)
#         drawGraph(graph_data, sg, "subgoal")
    plt.suptitle('gamma = ' + str(gamma) + ", p = " + str(p))
    plt.show() 

# graph_data = json.load(open("structure.json", "r"))
# # drawGraph(graph_data, (5, 6), "subgoal")
# drawGraph(graph_data, (0, 5, 11, 15, 22), "path")
# plt.show() 

