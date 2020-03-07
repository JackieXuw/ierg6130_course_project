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
import sys

from algo import feature, qvalue, cal_feature, run, baseline, isterminate, reward, init

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', help='learning rate for Adam optimizer', type=float, default=1e-3)
    parser.add_argument('-v', '--version', help='version of this running code', type=str, default=time.strftime("%m-%d-%H-%M-%S", time.localtime()))
    parser.add_argument('-s', help='fixed source if assigned', type=int, default=0)
    parser.add_argument('-d', help='fixed dest if assigned', type=int, default=0)
    parser.add_argument('-f', '--feature', help='feature dimension', type=int, default=16)
    parser.add_argument('-e', '--epsilon', help='initial epsilon for linaer decay e-greedy policy', type=float, default=0.5)
    parser.add_argument('-m', '--memory', help='memory size', type=int, default=4000)
    parser.add_argument('-b', '--batch_size', help='batch size', type=int, default=32)
    parser.add_argument('-p', '--params', help='name of params dict', type=str, default='')
    parser.add_argument('-i', '--init', help='whether to enter init model', action='store_true')
    parser.add_argument('-S', '--single', help='whether to use SINGLE model', type=bool, default=True)
    parser.add_argument('--enableBack', help='whether to enable the agent to go back', action='store_true')
    parser.add_argument('--guide_prob', help='guide probability in e-greedy policy', default=0.3)
    args=parser.parse_args()
    if args.batch_size > args.memory:
        logging.error('batch_size {} > memory {}'.format(args.batch_size, args.memory))
        exit(-1)
    
    ROOT=args.version
    DIM=args.feature
    DDL=4  # Time DDL
    BATCH_SIZE = args.batch_size
    SINGLE=args.single
    epsilon_max = args.epsilon
    fixed_sd=0
    guide_prob = args.guide_prob
    #epsilon_max = 0.5
    epsilon_min = 0.05
    epsilon = epsilon_max
    gamma = 0.95
    time_window=4
    
    os.makedirs(ROOT, exist_ok=True)
    logging.basicConfig(format='%(asctime)s[%(levelname)s] %(filename)s:%(lineno)d| %(message)s', datefmt='%m/%d %H:%M:%S', level=logging.INFO, filename='{}/log'.format(ROOT), filemode='w')
    logging.info("Usage:\n{0}\n".format(" ".join([x for x in sys.argv])))
    logging.debug("All settings used:")
    for k, v in sorted(vars(args).items()):
        logging.debug("{0}: {1}".format(k, v))
    
    if args.s and args.d:
        fixed_sd = 1
        S = args.s
        D = args.d
    else:
        S = None
        D = None
    M = replayMemory.ReplayMemory(args.memory) # Replay Memory initialization
    G = nx.read_gml('reduced_graph', destringizer=eval) # Graph
    f = feature(DIM)
    q = qvalue(DIM)
    if args.params != '':
        data = torch.load(args.params)['state_dict']
        try:
            assert data[0].shape[0] == DIM
        except AssertionError:
            logging.error('asserting failed, dimension of dict not match')
            exit(-1)
        f.params_p, f.params_p2, f.params_pp, q.params_p2, q.params_pp = deepcopy(data)
        logging.info('load params {}'.format(args.params))

    opt = Adam([f.params_p, f.params_p2, f.params_pp, q.params_p2, q.params_pp], lr=args.lr)
    criterion = nn.MSELoss()
    f_fixed = deepcopy(f) # Fixed network to reduce variance.
    q_fixed = deepcopy(q)
    if args.init:
        init(G, f, q, gamma, enableBack=args.enableBack, version=args.version)
    
    # For plotting issue
    # visited_nodes = set()
    # visited_nodes_profile = dict()
    # current_updated_nodes = set()
    # current_updated_profile = []
    plt.ion()
    iteration = -1 
    logging.info('Begin running')
    while M.insert_pos < min(args.batch_size, args.memory - 1):  # Pre run
        run(G, M, f, f_fixed, q, q_fixed, DDL, guide_prob=guide_prob, epsilon=epsilon, time_window=time_window, gamma=gamma, fixed_sd=fixed_sd, source=S, dest=D, guidance=True)
    logging.info('Pre running is finished')
    
    while iteration < 1000000:
        iteration += 1
        print('iter: ', iteration)
        
        if not iteration % 300:
            epsilon = epsilon_max - (epsilon_max - epsilon_min) / 1000000 * iteration  # Linear decay
            logging.info('Updated epsilon={} at iteration {}'.format(epsilon, iteration))
            avg_rd = 0
            avg_rd_bsl = 0
            logging.info('Begin evaluation...')
            for i in range(50):
                source, dest, rd = run(G, M, f, f_fixed, q, q_fixed, DDL, epsilon=0.0, guide_prob=0.0, time_window=time_window, gamma=gamma, fixed_sd=fixed_sd, source=S, dest=D, guidance=False)
                avg_rd += rd
                avg_rd_bsl += baseline(G, source, dest)
            logging.info('End evaluation')
            avg_rd /= 50
            avg_rd_bsl /= 50
            logging.info('Performance at iteration {}, rd: {}, bsl: {}'.format(iteration, avg_rd, avg_rd_bsl))
            torch.save({
                          'iteration': iteration,
                          'state_dict': [f.params_p, f.params_p2, f.params_pp, q.params_p2, q.params_pp], 
            }, '{}/t_{}_v_{}_rd,bsl_{},{}_iter_{}_'.format(ROOT, time.strftime("%m-%d-%H-%M-%S", time.localtime()), args.version, avg_rd, avg_rd_bsl, iteration) + '.t7')
            logging.info('Model saved at iter {}'.format(iteration))
            
        if not iteration % 100: # Update the fixed network
            f_fixed = deepcopy(f)
            q_fixed = deepcopy(q)
        #if not iteration % BATCH_SIZE:
        # Perform SGD
        batch = M.sample(BATCH_SIZE)
        opt.zero_grad()
        avg_loss = 0
        for i in batch:
            #node, rTime, dest, visited = i[0]
            print(i)
            infoDict, d, v, dest, rd, isT = i
            nodeSet = set(infoDict.keys())
            visitedDict = {key: infoDict[key][0] for key in infoDict}
            rTimeDict = {key: infoDict[key][1] for key in infoDict}
            nodeSet.add(v)
            visitedDict[v] = 0
            rTimeDict[v] = rTimeDict[d] - G[d][v]['time']

            if SINGLE:
                featureDict = cal_feature(G, f, {d,v}, visitedDict, rTimeDict, 4)
            else:
                featureDict = cal_feature(G, f, nodeSet, visitedDict, rTimeDict, 4)
            nodeSet.remove(v)
            if SINGLE:
                qv = q(G, {d}, v, featureDict)
            else: 
                qv = q(G, nodeSet, v, featureDict)
            target = -1e4
            if not isT:
                nodeSet.add(v)
                for adj in G.adj[v]:
                    nodeSet.add(adj)
                    visitedDict[adj] = 0
                    rTimeDict[adj] = rTimeDict[v] - G[v][adj]['time']
                    if SINGLE:
                        featureDict = cal_feature(G, f, {v, adj}, visitedDict, rTimeDict, 4)
                    else:  
                        featureDict = cal_feature(G, f, nodeSet, visitedDict, rTimeDict, 4)
                    nodeSet.remove(adj)
                    if SINGLE:
                        current = q(G, {v}, adj, featureDict)
                    else:
                        current = q(G, nodeSet, adj, featureDict)
                    # print(current)

                    target = max([target, current])
                target *= gamma
            else:
                target = 0
            target += rd

                    
            # current_updated_profile.append([node, rTime, dest, visited, qv.detach()])
            # visited_nodes_profile[node] = [rTime, dest, visited, qv.detach()]

            print('qv: {}, target: {}'.format(qv, target))
            loss = criterion(qv, torch.tensor(target).float())
            avg_loss += loss.detach()
            loss.backward()
        
        avg_loss /= BATCH_SIZE
        logging.info('loss={} at iteration {}'.format(avg_loss, iteration))
        opt.step()
        # if not iteration % 30:  # plot graph
        #     plt.close('all')
        #     subG = G.subgraph(visited_nodes)
        #     pos = nx.spring_layout(subG)
        #     nx.draw(subG, pos, with_labels=True)
        #     labels = {}
        #     for node, j in visited_nodes_profile.items():
        #         rTime, dest, visited, qv = j
        #         G.nodes[node]['rTime_real'] = rTime
        #         nx.set_node_attributes(G, None, 'feature')
        #         nx.set_node_attributes(G, 0, 'visited')
        #         nx.set_node_attributes(G, visited, 'visited')
        #         nx.set_node_attributes(G, dest, 'dest')
        #         cal_feature(G, f, node, dest, rTime, 4)
        #         qv_new = q(G, node, dest, rTime)
        #         labels[node] = "{:.2f}, {}, {:.2f}, {:.2f}".format(rTime, dest, float(qv), float(qv_new.detach()))
        #     for i, j in pos.items():
        #         j[1] -= 0.05
        #     nx.draw_networkx_labels(subG, pos=pos, labels=labels, font_size=8, alpha=0.5)
        #     labels = {}            
        #     for i in current_updated_profile:
        #         node, rTime, dest, visited, qv = i
        #         G.nodes[node]['rTime_real'] = rTime
        #         nx.set_node_attributes(G, None, 'feature')
        #         nx.set_node_attributes(G, 0, 'visited')
        #         nx.set_node_attributes(G, visited, 'visited')
        #         nx.set_node_attributes(G, dest, 'dest')
        #         cal_feature(G, f, node, dest, rTime, 4)
        #         qv_new = q(G, node, dest, rTime)
        #         if node in labels:
        #             labels[node] += "->{:.2f}".format(float(qv))
        #         else:
        #             labels[node] = "{:.2f}, {}, {:.2f}".format(rTime, dest, float(qv))
        #             # labels[node] = [rTime, dest, visited, float(qv), float(qv_new.detach())]
        #     for i, j in pos.items():
        #         j[1] += 0.15
        #     nx.draw_networkx_labels(subG, pos=pos, labels=labels, font_size=8)
        #     current_updated_nodes = set()
        #     current_updated_profile = []
        if  iteration % 10:
            run(G, M, f, f_fixed, q, q_fixed, DDL, epsilon=epsilon, guide_prob=guide_prob, time_window=time_window, gamma=gamma, fixed_sd=fixed_sd, source=S, dest=D, guidance=False)
        else:
            run(G, M, f, f_fixed, q, q_fixed, DDL, epsilon=epsilon, time_window=time_window, gamma=gamma, fixed_sd=fixed_sd, source=S, dest=D, guidance=True)


