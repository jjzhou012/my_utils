#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@author: jjzhou012
@contact: jjzhou012@163.com
@file: kshell.py
@time: 2020/3/19 10:43
@desc:  k-shell: rank of vertex importance
        First, we decompose the network into its k-shells. We start by removing all nodes with one connection only (with their links),
        until no more such nodes remain, and assign them to the 1-shell. In the same manner, we recursively remove all nodes with degree 2 (or less),
        creating the 2-shell. We continue, increasing k until all nodes in the graph have been assigned to one of the shells.
        We name the highest shell index k max.
        The k-core is defined as the union of all shells with indices larger or equal to k.
        The k-crust is defined as the union of all shells with indices smaller or equal to k.
'''

import networkx as nx
import copy

def kShell(graph):
    g = copy.deepcopy(graph)
    # init importance dict
    importance_dict = {}
    #
    ks = 1
    #
    while g.nodes():
        # save nodes with degree ks (or less)
        temp = []

        # filter: recursively remove all nodes with degree ks (or less)
        while True:
            for node, degree in copy.deepcopy(nx.degree(g)):
                if degree <= ks:
                    temp.append(node)
                    g.remove_node(node)
            # print(ks, sorted(dict(g.degree()).values()))
            try:
                if min(dict(g.degree()).values()) > ks:
                    break
            except ValueError:
                break
        #
        importance_dict[ks] = temp
        ks += 1

    return importance_dict



def main():
    file = '../data/karate.gml'
    graph = nx.read_gml(file)
    importance_dict = kShell(graph)
    print(importance_dict)




if __name__ == '__main__':
    main()
