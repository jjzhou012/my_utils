#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@author: jjzhou012
@contact: jjzhou012@163.com
@file: plot_hotmap.py
@time: 2021/11/22 18:58
@desc: 绘制热力图
'''

import numpy as np
import os
from functools import reduce
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import colors

# 设置文字、字体
from pylab import *  # 支持中文

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体以便支持中文
# mpl.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['text.usetex'] = True

# mpl.rcParams['font.size'] = 12
font = {'family': 'SimHei',
        'weight': 'normal',
        'size': 16}

font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 12}

font2 = {'family': 'SimHei',
         'weight': 'normal',
         'size': 13}

# plt.rc('font', family='SimHei')
import warnings

warnings.filterwarnings('ignore')  # 取消警告


######################################################### 定义画图函数 #########################################################
def heatmap(data, row_labels, col_labels, norm=None, ax=None, hasCbar=True, drow_ylabel=True, hasCbarTick=True,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    if norm:
        im = ax.imshow(data, norm=norm, **kwargs)
    else:
        im = ax.imshow(data, **kwargs)

    # Create colorbar
    # position=fig.add_axes([0.5, 0.5, 0.7, 0.5])#位置[左,下,右,上]
    if hasCbar:
        cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)  # orientation='horizontal',
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
        if not hasCbarTick:
            cbar.ax.set_yticklabels([])
    else:
        cbar = None

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels, font2)
    if drow_ylabel:
        ax.set_yticks(np.arange(data.shape[0]))
        ax.set_yticklabels(row_labels, font2)
    else:
        ax.set_yticks([])

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=False, bottom=True,
                   labeltop=False, labelbottom=True)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=15, ha="center",
             rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=15, ha="right",
             rotation_mode='anchor')

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)

    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)

    ax.tick_params(which="minor", bottom=False, left=False)

    return im  # , cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        #         threshold = im.norm(data.max())/2.
        threshold = im.norm(data.mean())

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


#########################################################  定义数据  #########################################################
datasets = ['Cora', 'Citeseer', 'Pubmed', 'Facebook', 'Github']
x_labels = ['AttrMask', 'AttrSimilar', 'EdgeRemove', 'KnnGraph', 'Identical']
y_labels = x_labels

cora_data = [[-0.31, 1.55, -0.24, 1.52, 0.82],
             [1.50, 1.39, 1.55, 1.46, 1.70],
             [-0.17, 1.49, 0.53, 1.50, 0.83],
             [1.52, 1.45, 1.49, -0.01, 1.45],
             [0.82, 1.70, 0.83, 1.45, 0.00]]

citeseer_data = [[0.56, 8.93, 1.11, 9.53, 5.17],
                 [9.47, 9.51, 9.42, 9.46, 9.36],
                 [0.64, 8.98, -1.20, 9.53, 5.29],
                 [9.47, 9.46, 9.47, 8.42, 9.61],
                 [5.17, 9.36, 5.29, 9.61, 0.00]]

pubmed_data = [[0.29, 0.46, 0.35, 0.22, 0.18],
               [0.58, 0.48, 0.43, 0.20, 0.02],
               [0.22, 0.23, 0.41, -0.03, -0.04],
               [0.08, 0.16, 0.36, 0.73, 0.30],
               [0.18, 0.02, -0.04, 0.30, 0.00]]

facebook_data = [[0.11, 0.04, 0.08, 0.07, 0.13],
                 [0.07, 0.10, 0.07, 0.03, 0.01],
                 [0.12, 0.37, 0.13, 0.14, 0.00],
                 [0.22, 0.08, 0.14, 0.23, 0.33],
                 [0.13, 0.01, 0.00, 0.33, 0.00]]

github_data = [[0.13, 0.26, 1.23, 0.87, 0.26],
               [0.18, 2.33, 0.10, 0.17, 0.58],
               [-0.12, 0.14, 0.67, 0.55, 0.30],
               [1.46, -0.08, 0.86, 0.69, 0.96],
               [0.26, 0.58, 0.30, 0.96, 0.00]]

cora_data = np.array(cora_data)
citeseer_data = np.array(citeseer_data)
pubmed_data = np.array(pubmed_data)
facebook_data = np.array(facebook_data)
github_data = np.array(github_data)

datas = [cora_data, citeseer_data, pubmed_data, facebook_data, github_data]

vmin = min(np.min(cora_data), np.min(citeseer_data), np.min(pubmed_data), np.min(facebook_data), np.min(github_data))
vmax = max(np.max(cora_data), np.max(citeseer_data), np.max(pubmed_data), np.max(facebook_data), np.max(github_data))
norm = colors.Normalize(vmin=vmin, vmax=vmax)


######################################################## 绘图1 ####################################################################
np.random.seed(19680801)
# plt.rc('font', family='SimHei')
# cmap_type = 'YlGn', 'RdPu', 'Blues' 'YlGnBu', 'PuOr', , 'magma_r', 'Wistia'
cmap_type = 'Blues'

fig_heatmap1, all_axes1 = plt.subplots(1, 2, figsize=(11.5, 4))

for i in range(len(all_axes1)):
    im = heatmap(datas[i], x_labels, y_labels, ax=all_axes1[i], cmap=cmap_type)
    annotate_heatmap(im, valfmt="{x:.2f}", size=16)

for i, dataset in enumerate(datasets[:2]):
    all_axes1[i].set_title(datasets[i], font)

plt.savefig('hotmap-1.pdf')
plt.show()

######################################################## 绘图2 ###################################################################
##
np.random.seed(19680801)
cmap_type = 'Blues'

new_datas = datas
new_datasets = datasets

fig_heatmap2, all_axes2 = plt.subplots(1, 5, figsize=(22, 4))

for i in range(len(all_axes2)):
    im = heatmap(new_datas[i], x_labels, y_labels, ax=all_axes2[i], cmap=cmap_type, hasCbar=False if i != 4 else True,
                 drow_ylabel=False if i else True, hasCbarTick=False)
    annotate_heatmap(im, valfmt="{x:.2f}", size=16)

# fig_heatmap2.colorbar(im, ax=[all_axes2[0], all_axes2[1],all_axes2[2], all_axes2[3]], fraction=0.03, pad=0.05)

for i, dataset in enumerate(new_datasets):
    all_axes2[i].set_title(new_datasets[i], font)
plt.tight_layout(pad=1.5, h_pad=0.18, w_pad=0., rect=None)
plt.subplots_adjust(top=0.9, bottom=0.16, hspace=0.12, wspace=0.)
plt.savefig('hotmap-2.pdf')
plt.show()