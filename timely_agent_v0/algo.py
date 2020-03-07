import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
import random
from copy import deepcopy
from queue import Queue
import replayMemory
from torch.optim import Adam
import argparse
import time
import os
import logging

#%matplotlib notebook

class feature(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.params_p = torch.Tensor(dim,6).normal_()
        self.params_p2 = torch.Tensor(dim, 2, 2).normal_()
        self.params_pp = torch.Tensor(dim, dim, 2).normal_()
        self.params_p.requires_grad_()
        self.params_p2.requires_grad_()
        self.params_pp.requires_grad_()

# <<<<<<< HEAD
    def forward(self, G, node, dest, rTime, visited = True):
# =======
#     def forward(self, G, node, dest, rTime):
# >>>>>>> patch-1

        if G.nodes[node]['dest'] != dest:
            G.nodes[node]['visited'] = 0
            G.nodes[node]['dest'] = dest
            G.nodes[node]['feature'] = dict()
        adj_feature = torch.zeros(self.dim)
        adj_feature_num = 0
        adj_cost_feature = 0
        adj_time_feature = 0
        adj_min_cost = [1e4, 0]
        adj_min_time = [0, 1e4]
        try:
            # whether can we go from node to dest
            shortest_path = nx.shortest_path_length(G, node, dest, 'time')
        except:
            shortest_path = 1e2 # Very large punishment
        
        for adj in G.adj[node]:
            adj_cost_feature += F.relu(self.params_p[:, 2] * G[node][adj]['cost'])
            adj_time_feature += F.relu(self.params_p[:, 3] * G[node][adj]['time'])
            if G[node][adj]['cost'] < adj_min_cost[0]:
                adj_min_cost[0] = G[node][adj]['cost']
                adj_min_cost[1] = G[node][adj]['time']
            if G[node][adj]['time'] < adj_min_time[1]:
                adj_min_time[0] = G[node][adj]['cost']
                adj_min_time[1] = G[node][adj]['time']
            if G.nodes[adj]['dest'] == dest and '{:.3f}'.format(rTime - G[node][adj]['time']) in G.nodes[adj]['feature']:
                adj_feature += G.nodes[adj]['feature']['{:.3f}'.format(rTime - G[node][adj]['time'])]
                adj_feature_num += 1
        # Take average by number of nodes
# <<<<<<< HEAD
        if len(list(G.adj)) != 0:
            adj_cost_feature /= len(list(G.adj))
            adj_time_feature /= len(list(G.adj))
        if adj_feature_num != 0:
            adj_feature /= adj_feature_num
        feature = self.params_p[:, 0] * G.nodes[node]['visited']
        if not visited:
            feature = torch.zeros(feature.shape)
# =======
#         adj_cost_feature /= len(list(G.adj))
#         adj_time_feature /= len(list(G.adj))
#         adj_feature /= adj_feature_num
#         feature = self.params_p[:, 0] * G.nodes[node]['visited']
# >>>>>>> patch-1
        feature += self.params_p[:, 4] * shortest_path
        feature += self.params_p[:, 5] * rTime
        if len(list(G.adj[node])) == 0:
            return feature
        feature += self.params_p[:, 1] * adj_feature
        feature = feature.unsqueeze(1)
        
        feature += torch.matmul(self.params_pp[:, :, 0],  adj_cost_feature.unsqueeze(1))
        feature += torch.matmul(self.params_pp[:, :, 1],  adj_time_feature.unsqueeze(1))
        feature += torch.matmul(self.params_p2[:,:,0], torch.tensor(adj_min_cost).unsqueeze(1))
        feature += torch.matmul(self.params_p2[:,:,1], torch.tensor(adj_min_time).unsqueeze(1))
        assert feature.shape == (self.dim, 1)
        return feature.squeeze(1)


class qvalue(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.params_p2 = torch.Tensor(2 * dim).normal_()
        self.params_pp = torch.Tensor(dim, dim, 2).normal_()
        self.params_p2.requires_grad_()
        self.params_pp.requires_grad_()
    
    def forward(self, G, traceSet, v, featureDict):
        sum_feature = 0
        sum_feature_num = 0
        for node in traceSet:
            sum_feature += featureDict[node]
            sum_feature_num += 1
# <<<<<<< HEAD
        if sum_feature_num != 0:
            sum_feature = sum_feature / sum_feature_num
# =======
#         sum_feature /= sum_feature
# >>>>>>> patch-1

        inp = torch.cat((torch.matmul(self.params_pp[:,:, 0], sum_feature.unsqueeze(1)), torch.matmul(self.params_pp[:,:, 1], featureDict[v].unsqueeze(1))), dim=0)
        return torch.matmul(self.params_p2, F.relu(inp))

def cal_feature(G, f, nodeSet, visitedDict, rTimeDict, iteration):
    
    emptyDict = dict()
    for key in G.nodes:
        if key not in visitedDict:
            visitedDict[key] = 0
        emptyDict[key] = dict()
    nx.set_node_attributes(G, emptyDict, 'feature')
    nx.set_node_attributes(G, visitedDict, 'visited')
    currentNodeSet = deepcopy(nodeSet)
    currentRTimeDict = deepcopy(rTimeDict)  # value is list 
    for key, value in currentRTimeDict.items():
        currentRTimeDict[key] = [value]
    while iteration > 0:
        toAddSet = set()
        for node in currentNodeSet:
            for rTime in currentRTimeDict[node]:
                G.nodes[node]['feature']['{:.3f}'.format(rTime)] = f(G, node, G.nodes[node]['dest'], rTime)
            for _ in list(nx.bfs_successors(G, node, depth_limit=1)): #[{key: []}]
                for v in _[1]:
                    toAddSet.add(v)
                    if v not in currentRTimeDict:
                        currentRTimeDict[v] = [rTime - G[node][v]['time']]
                    else:
                        currentRTimeDict[v].append(rTime - G[node][v]['time'])
        currentNodeSet = currentNodeSet.union(toAddSet)
        iteration -= 1
    featureDict = {}
    for node in nodeSet:
        featureDict[node] = G.nodes[node]['feature']["{:.3f}".format(rTimeDict[node])] # value of rTimeDict is float
    return featureDict # Return all nodes reached 
        
def reward(G, depart, arrive, dest, rTime, success = 100, fail = -50):
    if arrive == dest and rTime >=0:
        return success
    elif arrive == dest and rTime < 0:
        return fail / 2
    elif rTime < 0:
        return fail
    else:
        try:
            # Whether can we go from arrive to dest
            nx.shortest_path_length(G, arrive, dest, 'time')
        except:
            # We entered into a dead end
            return fail
    return -G[depart][arrive]['cost']
def isterminate(G, depart, arrive, dest, rTime):
    if rTime < 0:
        logging.info('Time out! Terminated')
        return 1
    elif dest == arrive:
        logging.info('Arrived! Terminated')
        return 1
    else:
        try:
            nx.shortest_path_length(G, arrive, dest, 'time')
        except:
            logging.info('Dead end! Terminated')
            return 1
    return 0

# <<<<<<< HEAD
def baseline(G, source, dest, base_line='fastest_path', returnPath=False):
# =======
# def baseline(G, source, dest, base_line='fastest_path'):
# >>>>>>> patch-1
    if base_line == 'fastest_path': # Run the baseline algorithm
        path = nx.shortest_path(G, source, dest, 'time')
        rd = 100
        depart = source
        for node in path:
            if node == source:
                continue
            arrive = node
            rd -= G[depart][arrive]['cost']
            depart = arrive
# <<<<<<< HEAD
        if returnPath:
            return path
        else:
            return rd
def generateFeasiblePair(G, n, baseline=False):
    result = []
    for i in range(n):
        flag = 0
        while not flag:
            DDL = random.random() * 5
            source, dest = random.sample(list(G.nodes), 2)
            tryTime = 10
            while tryTime > 0:
                try:
                    total_time = nx.shortest_path_length(G, source, dest, 'time')
                except:
                    source, dest = random.sample(list(G.nodes), 2)
                    tryTime -= 1
                    continue
                if total_time < DDL:
                    if baseline:
                        path = baseline(G, source, dest, returnPath=True)
                    flag = 1
                    break
                else:
                    tryTime -= 1
        if baseline:
            result.append((source, dest, DDL, path))
        else:
            result.append((source, dest, DDL))
    return result

def run(G, M, f, f_fixed, q, q_fixed, DDL, epsilon = 0.2, guide_prob = 0.1, time_window = 4, gamma = 0.8, fixed_sd=False, source=None, dest=None, guidance=False, SINGLE=False):
# =======
#         return rd

#def run(G, M, f, f_fixed, q, q_fixed, DDL, e_greedy = 1, epsilon = 0.2, time_window = 4, gamma = 0.8, fixed_sd=False, source=None, dest=None, guidance=None, SINGLE=False):
# >>>>>>> patch-1
    if fixed_sd: # Run the algo for the given s-d pair
        try:
            total_time = nx.shortest_path_length(G, source, dest, 'time')
        except:
            logging.critical('invalid s-d pair: {}-{}, exit!'.format(source, dest))
    else:
        source, dest = random.sample(list(G.nodes), 2)
        while 1: # try to find a feasible source-dest pair
            try:
                total_time = nx.shortest_path_length(G, source, dest, 'time')
            except:
                source, dest = random.sample(list(G.nodes), 2)
                continue
            if total_time < DDL:
                break
            else:
                source, dest = random.sample(list(G.nodes), 2)

    # initialize node attributes and other variables
    nx.set_node_attributes(G, 0, 'visited')
    nx.set_node_attributes(G, dest, 'dest')
    G.nodes[source]['visited'] = 1
    G.nodes[source]['dest'] = dest

    depart = source
    arrive = source
    rd = 0
    que_reward = Queue(time_window)
    que_node = Queue(time_window + 1)
    logging.info('source: {}, destination: {}'.format(source, dest))
    trace = [source]

    infoDict = {source:[1,DDL]}
    nodeSet = {source}
    visitedDict = {source: 1}
# <<<<<<< HEAD
    removedEdge = []
# =======
# >>>>>>> patch-1
    rTimeDict = {source: DDL}

    if guidance:
        guide = list(nx.shortest_path(G, source, dest, 'time')) # guide is the fastest path 
    
    while not isterminate(G, depart, arrive, dest, rTimeDict[arrive]):
# <<<<<<< HEAD
        if arrive != depart:
            infoDict[arrive] = [1, infoDict[depart][1] - G[depart][arrive]['time']]
            nodeSet.add(arrive)
            edge = G[depart][arrive]
            G.remove_edge(depart, arrive)
            removedEdge.append([depart, arrive, edge])
# =======
#         if arrive != source:
#             infoDict[arrive] = [1, infoDict[depart][1] - G[depart][arrive]['time']]
#             nodeSet.add(arrive)
# >>>>>>> patch-1
        depart = arrive
        arrive = None
        

        max_qvalue = -100000
        q_value_dict = dict()
        # if not guidance:
        for n in G.adj[depart]:  # Find the max Q
                # First check feasibility
# <<<<<<< HEAD
            if nx.has_path(G, n, dest) and (G.nodes[n]['visited'] != 1 or guidance): # Guarantee to goto not a dead end, and not loopback
# =======
#             if nx.has_path(G, n, dest) and G.nodes[n]['visited'] != 1: # Guarantee to goto not a dead end, and not loopback
# >>>>>>> patch-1
                    #cal_feature(G, f, n, dest, G.nodes[depart]['rTime_real'] - G[depart][n]['time'], 4)
                nodeSet.add(n)
                visitedDict[n] = 0
                rTimeDict[n] = rTimeDict[depart] - G[depart][n]['time']
                if SINGLE:
                    featureDict = cal_feature(G, f, {depart, n}, visitedDict, rTimeDict, 4)
                else:
                    featureDict = cal_feature(G, f, nodeSet, visitedDict, rTimeDict, 4)
                nodeSet.remove(n)

                if SINGLE:
                    c_qvalue = q(G, {depart}, n, featureDict)
                else:
                    c_qvalue = q(G, nodeSet, n, featureDict)

                logging.info('Q({}, {}) = {}'.format(depart, n, c_qvalue.detach().float()))
                q_value_dict[n] = c_qvalue
                visitedDict.pop(n)
                rTimeDict.pop(n)
# <<<<<<< HEAD
        print(q_value_dict.keys())
        print(list(G.adj[depart]))
        if guidance:
            if epsilon == 0 and guide_prob == 0:
                print('Error! ')
                exit()
# =======
#         if guidance:
# >>>>>>> patch-1
            p = guide.index(depart)
            arrive = guide[p + 1]
            max_qvalue = q_value_dict[arrive]
            logging.info('Selected {} by guidance, Q({}, {}) = {}'.format(arrive, depart, arrive, max_qvalue.detach().float()))
        
# <<<<<<< HEAD
        else:  # Run the e-greedy policy
            if random.random() < epsilon: # Random selection
                for tryTime in range(10):
                    arrive = random.randint(0, len(list(G.adj[depart])) - 1)
                    arrive = list(G.adj[depart])[arrive]
                    if nx.has_path(G, arrive, dest) is False or G.nodes[arrive]['visited'] == 1:
                        arrive = None
                    else:
                        break
                if arrive is not None:
                    max_qvalue = q_value_dict[arrive]
                    logging.info('random selected {}, q({}, {}) is {}'.format(arrive, depart, arrive, max_qvalue.detach().float()))
            elif random.random() - epsilon < guide_prob: # Switch to the guidance part
                guidance = True
                # Try to remove the previous edge to prevent turning back
                try:
                    print('try to remove edge {}->{}'.format(depart, trace[-2]))
                    edge = G[depart][trace[-2]]
                    G.remove_edge(depart, trace[-2])
                    print('removed edge {}->{}'.format(depart, trace[-2]))
                except: # No back edge from depart to previous node
                    pass
                try:
                    guide = list(nx.shortest_path(G, depart, dest, 'time'))
                    flag = 1
                except:
                    flag = 0
                try:
                    G.add_edge(depart, trace[-2], time=edge['time'], cost=edge['cost'])
                except (UnboundLocalError, IndexError) as e:
                    pass
                if flag == 1:
                    arrive = depart
                    continue
                    # p = guide.index(depart)
                    # arrive = guide[p + 1]
                    # max_qvalue = q_value_dict[arrive]
                    # logging.info('Selected {} by guidance, Q({}, {}) = {}'.format(arrive, depart, arrive, max_qvalue.detach().float()))
                else:
                    arrive = None
                
            else: # Greedy selection
                max_qvalue = -1e4
                for n, qv in q_value_dict.items():
                    if qv is not None and (qv > max_qvalue or max_qvalue == -100000):
                        max_qvalue = qv
                        arrive = n
                logging.info('greedy selected {}'.format(arrive))

        if arrive is None:
            rd = -50
# =======
#         elif e_greedy: # Run the e-greedy policy
#             if random.random() < epsilon:
#                 arrive = random.randint(0, len(list(G.adj[depart])) - 1)
#                 arrive = list(G.adj[depart])[arrive]
#                 while nx.has_path(G, arrive, dest) is False or G.nodes[arrive]['visited'] == 1:
#                     arrive = random.randint(0, len(list(G.adj[depart])) - 1)
#                     arrive = list(G.adj[depart])[arrive]
#                 max_qvalue = q_value_dict[arrive]
#                 logging.info('random selected {}, q({}, {}) is {}'.format(arrive, depart, arrive, max_qvalue.detach().float()))
#         else:
#             max_qvalue = -1e4
#             for n, qv in q_value_dict.items():
                
#                 if qv is not None and qv > max_qvalue or max_qvalue == -100000:
#                     max_qvalue = qv
#                     arrive = n
#             logging.info('greedy selected {}'.format(arrive))
#         if arrive is None:
# >>>>>>> patch-1
            logging.info('Walked into a dead end, break')
            break
        # current_time += 1
        
        rTimeDict[arrive] = infoDict[depart][1] - G[depart][arrive]['time']
        visitedDict[arrive] = 1
# <<<<<<< HEAD
        
        current_rd = reward(G, depart, arrive, dest, rTimeDict[arrive])
        isT = isterminate(G, depart, arrive, dest, rTimeDict[arrive])
        rd += current_rd  # The cumulative reward
        M.append([deepcopy(infoDict), depart, arrive, dest, current_rd, isT])
# =======
#         rd += reward(G, depart, arrive, dest, rTimeDict[arrive]) # The cumulative reward
#         assert max_qvalue is not None
#         M.append([deepcopy(infoDict), depart, arrive, dest, reward(G, depart, arrive, dest, rTimeDict[arrive]), isterminate(G, depart, arrive, dest, rTimeDict[arrive])])
# >>>>>>> patch-1


        logging.debug('from: {} to: {}'.format(depart, arrive))
        trace.append(arrive)
# <<<<<<< HEAD
    for edge in removedEdge:
        depart, arrive, info = edge
        G.add_edge(depart, arrive, time=info['time'], cost=info['cost'])
    logging.info('From {} to {}, Trace: {}'.format(source, dest, ''.join([str(i)+'->' for i in trace]))[:-2])
    logging.info('rd: {}'.format(rd))
    return source, dest, rd

def init(G, f, q, gamma, baseline=baseline, rd=100, off=-50, enableBack=True, **kwargs):
    optim = torch.optim.Adam([f.params_p, f.params_p2, f.params_pp, q.params_p2, q.params_pp])
    criterion = torch.nn.MSELoss()
    taskPair = generateFeasiblePair(G, 50, baseline=baseline)
    targetList = []
    total_num = 0
    for i in taskPair: # (source, dest, DDL, path)
        pathList = [] # [(node, time, qvalue, action)]
        source = i[0]
        dest = i[1]
        DDL = i[2]
        path = i[3]
        for i, node in enumerate(path):
            if node == source:
                remaining_time = DDL
            else:
                remaining_time = pathList[i - 1][1] - G[path[i - 1]][node]['time']
            if node == dest:
                action = None
            else:
                action = path[i + 1]
            
            pathList.append([node, remaining_time, 0, action])
        for i, s in enumerate(reversed(pathList)):
            node = s[0]
            time = s[1]
            if node == dest:
                qvalue = rd
            else:
                next_node = list(reversed(pathList))[i - 1][0]
                next_Q = list(reversed(pathList))[i - 1][2]
                qvalue = -G[node][next_node]['time'] + gamma * next_Q # -G[node][next node] + gamma * Q(next node)
            s[2] = qvalue
        for node in path:
            adj_node_list = list(nx.bfs_successors(G, node, depth_limit=1))[0][1] # adj nodes of nodes that on the fastest path
            
            for adj_node in adj_node_list:
                remaining_time = DDL - nx.shortest_path_length(G, source, adj_node, 'time') # source->path_node
                if adj_node in path:
                    continue
                
                if enableBack: # Enable the agent to go back
                    try: 
                        fastest_path = nx.shortest_path(G, adj_node, dest, 'time')
                    except: # If this action leads to a dead end, return -rd/2
                        qvalue = -rd / 2
                        pathList.append([node, remaining_time, qvalue, adj_node])
                        continue
                else:
                    try: # Remove the edge to prevent going back
                        edge = G[adj_node][dest]
                        G.remove_edge(adj_node, dest)
                        flag = 1
                    except:
                        flag = 0
                    try: # Try to find an alternative path from adj_node to dest
                        fastest_path = nx.shortest_path(G, adj_node, dest, 'time')
                    except: 
                        if flag == 1:
                            G.add_edge(adj_node, dest, time=edge['time'], cost=edge['cost'])
                        else:
                            pass
                        qvalue = -rd / 2
                        pathList.append([node, remaining_time, qvalue, adj_node])
                        continue
                    if flag == 1: # We have find an alternative path from adj_node to dest
                        G.add_edge(adj_node, dest, time=edge['time'], cost=edge['cost'])

                action = fastest_path[1] 
                qvalue = 0
                passed_time = 0
                for i, path_node in enumerate(reversed(fastest_path)):  # dest->path_node
                    if path_node == dest:
                        if remaining_time < 0:
                            qvalue += -50
                        else:
                            qvalue += rd
                    else:
                        qvalue -= G[path_node][list(reversed(fastest_path))[i - 1]]['time']
                    qvalue *= gamma
                pathList.append([node, remaining_time, qvalue, adj_node])
        total_num += len(pathList)
        targetList.append([source, dest, pathList])
    
    # Finished calculating relevant nodes' qvalue
    # Performing SGD
    iteration = 0
    avg_loss200 = 0
    while 1:
        iteration += 1
        pathList = targetList[random.randint(0, len(targetList) - 1)]
        source = pathList[0]
        dest = pathList[1]
        pathList = pathList[2]
        nx.set_node_attributes(G, dest, 'dest')
        optim.zero_grad()
        avg_loss = 0
        for pair in pathList:
            node = pair[0]
            remaining_time = pair[1]
            if remaining_time == None:
                remaining_time = 10
            target = pair[2]
            action = pair[3]
            if action == None:
                continue
            featureDict = cal_feature(G, f, {node, action}, {node: 0}, {node: remaining_time, action: remaining_time - G[node][action]['time']}, 4)
            qvalue = q(G, {node}, action, featureDict)
            loss = criterion(qvalue, torch.tensor(target))
            print('qvalue: {}, target: {}'.format(qvalue.detach().float(), target))
            print("{}, {}, {}".format(node, action, featureDict))
            avg_loss += loss.detach().float()
            loss.backward()
        avg_loss /= len(pathList)
        avg_loss200 += avg_loss
        logging.info('avg_loss: {}'.format(avg_loss))
        optim.step()
        if iteration % 200 == 1:
            avg_loss200 /= 200
            torch.save({
                          'iteration': iteration,
                          'state_dict': [f.params_p, f.params_p2, f.params_pp, q.params_p2, q.params_pp], 
            }, '{}/init_avgloss_{}_iter_{}_'.format(version, avg_loss200, iteration) + '.t7')
            avg_loss200 = 0
            logging.info('Init model saved at iter {}'.format(iteration))
# =======
#     logging.info('From {} to {}, Trace: {}'.format(source, dest, ''.join([str(i)+'->' for i in trace]))[:-2])
#     return source, dest, rd


# >>>>>>> patch-1
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', help='learning rate for Adam optimizer', type=float, default=1e-2)
    parser.add_argument('-v', '--version', help='version of this runing code', type=str, default=time.strftime("%m-%d-%H-%M-%S", time.localtime()))
    parser.add_argument('-s', help='source', type=int, default=0)
    parser.add_argument('-d', help='dest', type=int, default=0)
    parser.add_argument('-f', '--feature', help='feature dimension', type=int, default=16)
    parser.add_argument('-e', '--epsilon', help='initial epsilon for linaer decay e-greedy policy', type=float, default=0.5)
    args=parser.parse_args()
    ROOT = args.version
    DIM=args.feature
    os.makedirs(ROOT, exist_ok=True)

    epsilon_max = args.epsilon
    fixed_sd=0
    if args.s and args.d:
        fixed_sd = 1
        S = args.s
        D = args.d
    else:
        S = None
        D = None
    M = replayMemory.ReplayMemory(4000) # Replay Memory initialization
    
    DDL = 4 # Time DDL
    G = nx.read_gml('reduced_graph', destringizer=eval) # Graph
    f = feature(DIM)
    q = qvalue(DIM)
    opt = Adam([f.params_p, f.params_p2, f.params_pp, q.params_p2, q.params_pp], lr=args.lr)
    criterion = nn.MSELoss()
    f_fixed = deepcopy(f) # Fixed network to reduce variance.
    q_fixed = deepcopy(q)
    e_greedy = True
    #epsilon_max = 0.5
    epsilon_min = 0.05
    epsilon = epsilon_max
    gamma = 0.8
    time_window = 4
    iteration = -1
    
    while M.insert_pos < 10:  # Pre run
        run(G, M, f, f_fixed, q, q_fixed, DDL, e_greedy=e_greedy, epsilon=epsilon, time_window=time_window, gamma=gamma, fixed_sd=fixed_sd, source=S, dest=D)
        
    
    while iteration < 1000000:
        iteration += 1
        print('iter: ', iteration)
        if not iteration % 1600:  # Save the params and change the epsilon
            epsilon = epsilon_max - (epsilon_max - epsilon_min)/1000000*iteration # Linear decay
            avg_rd = 0
            avg_rd_bsl = 0
            for i in range(100):
                source, dest, rd = run(G, M, f, f_fixed, q, q_fixed, DDL, e_greedy=0, epsilon=False, time_window=time_window, gamma=gamma, fixed_sd=fixed_sd, source=S, dest=D)
                avg_rd += rd
                avg_rd_bsl += baseline(G, source, dest)
            avg_rd /= 200
            avg_rd_bsl /= 200
            torch.save({
                          'iteration': iteration,
                          'state_dict': [f.params_p, f.params_p2, f.params_pp, q.params_p2, q.params_pp], 
            }, 't_{}_v_{}_rd,bsl_{},{}_iter_{}_'.format(time.strftime("%m-%d;%H:%M:%S", time.localtime()), args.version, avg_rd, avg_rd_bsl, iteration) + '.t7')
        if not iteration % 320: # Update the fixed network
            f_fixed = deepcopy(f)
            q_fixed = deepcopy(q)
        if not iteration % 32:
            # Perform SGD
            batch = M.sample(32)
            opt.zero_grad()
            for i in batch:
                node, rTime, dest, visited = i[0]
                nx.set_node_attributes(G, 0, 'visited')
                nx.set_node_attributes(G, visited, 'visited')
                nx.set_node_attributes(G, dest, 'dest')
                cal_feature(G, f, node, dest, rTime, 4)
                qv = q(G, node, dest, rTime)
                # TODO: Why this happens
                try:
                    print('qv: {}, i[1]: {}'.format(qv, i[1]))
                    loss = criterion(qv, torch.tensor(i[1]))
                    loss.backward()
                except RuntimeError:
                    print('RuntimeError, Trying to backward through the graph a second time... qv: {}, i[1]: {}'.format(qv, i[1]))
    
                #print('backwarded')
            opt.step()
    
        run(G, M, f, f_fixed, q, q_fixed, DDL, e_greedy=e_greedy, epsilon=epsilon, time_window=time_window, gamma=gamma, fixed_sd=fixed_sd, source=S, dest=D)


