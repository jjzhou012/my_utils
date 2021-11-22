#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@author: jjzhou012
@contact: jjzhou012@163.com
@file: similarity.py
@time: 2019/11/6 16:55
@desc:      node similarity-based link prediction
            compute the lp scores of all pairwise-vertices in ebunch
'''

import networkx as nx
import numpy as np
import time
from scipy import sparse as sp
import time



def apply_prediction(g, similarityMatrix, ebunch=None):
    if ebunch is None:
        ebunch = nx.non_edges(g)

    if type(ebunch[0][0]) == str:
        ebunch = list(map(lambda x: (int(x[0]), int(x[1])), ebunch))

    edges = np.array(ebunch).T
    rows, cols = edges[0], edges[1]
    # print(similarityMatrix.shape)
    scores = np.nan_to_num(similarityMatrix[rows, cols])

    # triple = np.vstack((rows, cols, scores)).transpose().tolist()
    return scores.tolist()


# local similarity indices
def CN(g, ebunch=None):
    adj = nx.adjacency_matrix(g)
    similarityMatrix = sp.triu(np.dot(adj, adj), k=1).toarray()

    return apply_prediction(g, similarityMatrix, ebunch)


def Jaccard(g, ebunch=None):
    adj = nx.adjacency_matrix(g)

    # cn = np.dot(adj, adj)   # matrix
    # degree = adj.sum(axis=0)
    # similarityMatrix = cn / (degree + degree.T - cn)

    cn = sp.triu(np.dot(adj, adj), k=1)
    degree = np.asarray(adj.sum(axis=0))[0]

    x, y = cn.nonzero()
    scores = cn.data
    scores = scores / (degree[x] + degree[y] - scores)
    similarityMatrix = sp.coo_matrix((scores, (x, y)), shape=adj.shape).toarray()

    return apply_prediction(g, similarityMatrix, ebunch)


def Salton(g, ebunch=None):
    adj = nx.adjacency_matrix(g)
    cn = sp.triu(np.dot(adj, adj), k=0)
    degree = np.asarray(adj.sum(axis=0))[0]

    x, y = cn.nonzero()
    scores = cn.data
    scores = scores / np.sqrt(degree[x] * degree[y])
    similarityMatrix = sp.coo_matrix((scores, (x, y)), shape=adj.shape).toarray()

    return apply_prediction(g, similarityMatrix, ebunch)


def Sorenson(g, ebunch=None):
    adj = nx.adjacency_matrix(g)
    cn = sp.triu(np.dot(adj, adj), k=0)
    degree = np.asarray(adj.sum(axis=0))[0]

    x, y = cn.nonzero()
    scores = cn.data
    scores = 2 * scores / (degree[x] + degree[y])
    similarityMatrix = sp.coo_matrix((scores, (x, y)), shape=adj.shape).toarray()

    return apply_prediction(g, similarityMatrix, ebunch)


def HPI(g, ebunch=None):
    adj = nx.adjacency_matrix(g)
    cn = sp.triu(np.dot(adj, adj), k=0)
    degree = np.asarray(adj.sum(axis=0))[0]
    x, y = cn.nonzero()
    scores = cn.data

    dd = np.vstack((degree[x], degree[y]))
    minDegree = np.min(dd, axis=0)
    scores = scores / minDegree
    similarityMatrix = sp.coo_matrix((scores, (x, y)), shape=adj.shape).toarray()

    return apply_prediction(g, similarityMatrix, ebunch)


def HDI(g, ebunch=None):
    adj = nx.adjacency_matrix(g)
    cn = sp.triu(np.dot(adj, adj), k=0)
    degree = np.asarray(adj.sum(axis=0))[0]
    x, y = cn.nonzero()
    scores = cn.data

    dd = np.vstack((degree[x], degree[y]))
    minDegree = np.max(dd, axis=0)
    scores = scores / minDegree
    similarityMatrix = sp.coo_matrix((scores, (x, y)), shape=adj.shape).toarray()

    return apply_prediction(g, similarityMatrix, ebunch)


def LHN_I(g, ebunch=None):
    adj = nx.adjacency_matrix(g)
    cn = sp.triu(np.dot(adj, adj), k=0)
    degree = np.asarray(adj.sum(axis=0))[0]

    x, y = cn.nonzero()
    scores = cn.data
    scores = scores / (degree[x] * degree[y])
    similarityMatrix = sp.coo_matrix((scores, (x, y)), shape=adj.shape).toarray()

    return apply_prediction(g, similarityMatrix, ebunch)


def PA(g, ebunch=None):
    adj = nx.adjacency_matrix(g)
    degree = adj.sum(axis=0)

    similarityMatrix = np.asarray(np.dot(degree.T, degree))

    return apply_prediction(g, similarityMatrix, ebunch)


def AA(g, ebunch=None):
    adj = nx.adjacency_matrix(g).asformat("csr")
    degrees = adj.sum(axis=0)
    weights = sp.csr_matrix(np.divide(1, np.log2(degrees)))

    adj_weighted = adj.multiply(weights)
    similarityMatrix = sp.triu(adj_weighted * adj, k=1).toarray()

    return apply_prediction(g, similarityMatrix, ebunch)


def RA(g, ebunch=None):
    adj = nx.adjacency_matrix(g).asformat("csr")
    degrees = adj.sum(axis=0)
    weights = sp.csr_matrix(np.divide(1, degrees))

    adj_weighted = adj.multiply(weights)
    similarityMatrix = sp.triu(adj_weighted * adj, k=1).toarray()

    return apply_prediction(g, similarityMatrix, ebunch)


# path/global similarity indices
def LP(g, ebunch=None, alpha=0.5):
    adj = nx.adjacency_matrix(g)
    cn = adj @ adj
    triple_neighbors = alpha * adj @ adj @ adj

    lp = cn + triple_neighbors
    similarityMatrix = sp.triu(lp, k=1).toarray()

    return apply_prediction(g, similarityMatrix, ebunch)


def Katz(g, ebunch=None):
    adj = nx.adjacency_matrix(g)

    threshold = 1 / max(np.linalg.eig(adj.toarray())[0])
    # beta 必须小于邻接矩阵最大特征值的倒数
    beta = np.random.uniform(0, threshold)
    similarityMatrix = np.asarray(np.linalg.inv(np.eye(adj.shape[0]) - beta * adj) - np.eye(adj.shape[0]))
    return apply_prediction(g, similarityMatrix, ebunch)


# random walk similarity indices
def ACT(g, ebunch=None):
    # Laplacian matrix
    Laplac = nx.laplacian_matrix(g)
    # pinv of L
    pinvL = np.linalg.pinv(Laplac.toarray())

    pinvL_diag = np.diag(pinvL)  # [deg(1), deg(2), ...]
    matrix_one = np.ones(shape=Laplac.shape)
    pinvL_xx = pinvL_diag * matrix_one
    similarity_matrix = pinvL_xx + pinvL_xx.T - (2 * pinvL)
    similarityMatrix = 1 / similarity_matrix
    similarityMatrix[similarityMatrix < 0] = 0
    return apply_prediction(g, similarityMatrix, ebunch)


def Cos(g, ebunch=None):
    # Laplacian matrix
    Laplac = nx.laplacian_matrix(g)
    # pinv of L
    pinvL = np.linalg.pinv(Laplac.toarray())

    pinvL_diag = np.diag(pinvL)  # [deg(1), deg(2), ...]
    matrix_one = np.ones(shape=Laplac.shape)
    pinvL_xx = pinvL_diag * matrix_one
    similarityMatrix = pinvL / np.sqrt(pinvL_xx * pinvL_xx.T)
    # similarity_matrix = 1 / similarity_matrix
    similarityMatrix[similarityMatrix < 0] = 0

    return apply_prediction(g, similarityMatrix, ebunch)


def RWR(g, ebunch=None, c=0.85):
    adj = nx.adjacency_matrix(g).toarray()
    matrix_probs = adj / sum(adj)
    temp = np.eye(N=adj.shape[0]) - c * matrix_probs.T
    # matrix_rwr = (1 - c) * np.dot(np.linalg.inv(temp), np.eye(N=adj.shape[0]))
    try:
        matrix_rwr = (1 - c) * np.linalg.inv(temp)
    except np.linalg.LinAlgError:
        matrix_rwr = (1 - c) * np.linalg.pinv(temp)
    # except np.linalg.LinAlgError:


    similarityMatrix = matrix_rwr + matrix_rwr.T
    similarityMatrix[similarityMatrix < 0] = 0
    return apply_prediction(g, similarityMatrix, ebunch)


similarity_index_dict = {'cn': CN,
                         'jacc': Jaccard,
                         'salton': Salton,
                         'sorenson': Sorenson,
                         'hpi': HPI,
                         'hdi': HDI,
                         'lhn-1': LHN_I,
                         'pa': PA,
                         'aa': AA,
                         'ra': RA,
                         'lp': LP,
                         'katz': Katz,
                         'act': ACT,
                         'cos': Cos,
                         'rwr': RWR}

LPM = {'si': similarity_index_dict}
