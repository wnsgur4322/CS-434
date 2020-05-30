# CS 434 - Spring 2020
# implmentation assignment 4
# Team members
# Junhyeok Jeong, jeongju@oregonstate.edu
# Youngjoo Lee, leey3@oregonstate.edu
# Haewon Cho, choha@oregonstate.edu

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.ticker import MaxNLocator
from matplotlib import gridspec

sns.set()

import argparse

from utils import load_data
from decompose import PCA
from clustering import KMeans


def load_args():
    parser = argparse.ArgumentParser(description='arguments')
    parser.add_argument('--pca', default=1, type=int,
                        help='set to 1 if we desire running pca, otherwise 0')
    parser.add_argument('--kmeans', default=1, type=int,
                        help='set to 1 if we desire running kmeans, otherwise 0')

    parser.add_argument('--pca_retain_ratio', default=.9, type=float)
    parser.add_argument('--kmeans_max_k', default=10, type=int)
    parser.add_argument('--kmeans_max_iter', default=20, type=int)
    parser.add_argument('--root_dir', default='../data/', type=str)
    args = parser.parse_args()

    return args


def plot_y_vs_x_list(y_vs_x, x_label, y_label, save_path):
    fld = os.path.join(args.root_dir, save_path)
    if not os.path.exists((fld)):
        os.mkdir(fld)

    plots_per_fig = 2

    ks_sses_keys = list(range(0, len(y_vs_x)))
    js = list(range(0, len(ks_sses_keys), plots_per_fig))

    for j in js:
        pp = ks_sses_keys[j:j + plots_per_fig]
        fig = plt.figure(constrained_layout=True)
        gs = gridspec.GridSpec(len(pp), 1, figure=fig)
        i = 0
        for k in pp:
            ax = fig.add_subplot(gs[i, :])
            ax.set_ylabel('%s (k=%d)' % (y_label, k))
            ax.set_xlabel(x_label)
            ax.plot(range(1, len(y_vs_x[k]) + 1), [x for x in y_vs_x[k]], linewidth=2)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            i += 1

        fig.savefig(os.path.join(fld, '%d_%d.png' % (pp[0], pp[-1])))

    print('Saved at : %s' % fld)


def plot_y_vs_x(ys_vs_x, x_label, y_label, save_path):
    fld = os.path.join(args.root_dir, save_path)
    if not os.path.exists((fld)):
        os.mkdir(fld)

    fig = plt.figure(constrained_layout=True)
    gs = gridspec.GridSpec(1, 1, figure=fig)
    ax = fig.add_subplot(gs[0, :])
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.plot(range(1, len(ys_vs_x) + 1), ys_vs_x, linewidth=2)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.savefig(os.path.join(fld, 'plot.png'))

    print('Saved at : %s' % fld)


def visualize(x_train, y_train):
    pass
    ##################################
    #      YOUR CODE GOES HERE       #
    ##################################

    #ver 1
    '''
    pc1 = x_train.dot(x_train[0])
    pc2 = x_train.dot(x_train[1])
    '''

    #ver 2
    x_trans = x_train.transpose()
    #x_trans = x_train
    print(x_trans.shape)

    pc1 = x_trans[0]
    pc2 = x_trans[1]



    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)

    train_unique = list(set(y_train))
    train_colors = ["r","b","g", "y", "m", "w"]
    
    for i, spec in enumerate(y_train):
        print(i)
        plt.scatter(pc1[i], pc2[i], label = spec, s = 20, c=train_colors[train_unique.index(spec)])
        #ax.annotate(str(i+1), (pc1[i],pc2[i]))
    
    from collections import OrderedDict
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), prop={'size': 15}, loc=4)
    
    
    plt.show()

def apply_kmeans(do_pca, x_train, y_train, x_test, y_test, kmeans_max_iter, kmeans_max_k):
    print('kmeans\n')
    train_sses_vs_iter = []
    train_sses_vs_k = []
    train_purities_vs_k = []

    ##################################
    #      YOUR CODE GOES HERE       #
    ##################################
    max_repeat = 7
    for repeat in range(0, max_repeat):
        if repeat == 0:
            for k in range(1, kmeans_max_k):
                kmeans = KMeans(k, kmeans_max_iter)
                sse_vs_iter = kmeans.fit(x_train)
                train_sses_vs_iter.append(sse_vs_iter)
                train_purities_vs_k.append(kmeans.get_purity(x_train, y_train))
                train_sses_vs_k.append(min(sse_vs_iter))
        elif repeat == max_repeat:
            for k in range(1, kmeans_max_k):
                kmeans = KMeans(k, kmeans_max_iter)
                sse_vs_iter = kmeans.fit(x_train)
                train_sses_vs_iter[k-1] += sse_vs_iter[k-1]
                train_sses_vs_iter[k-1] = train_sses_vs_iter[k-1]/repeat
                
                train_purities_vs_k[k-1] += kmeans.get_purity(x_train, y_train)
                train_purities_vs_k[k-1] = train_purities_vs_k[k-1]/repeat

                train_sses_vs_k[k-1] += min(sse_vs_iter)
                train_sses_vs_k[k-1] = train_sses_vs_k[k-1]/repeat

        else:
            for k in range(1, kmeans_max_k):
                kmeans = KMeans(k, kmeans_max_iter)
                sse_vs_iter = kmeans.fit(x_train)
                train_sses_vs_iter[k-1] += sse_vs_iter[k-1]
                train_purities_vs_k[k-1] += kmeans.get_purity(x_train, y_train)
                train_sses_vs_k[k-1] += min(sse_vs_iter)

    

    plot_y_vs_x_list(train_sses_vs_iter, x_label='iter', y_label='sse',
                     save_path='plot_sse_vs_k_subplots_%d'%do_pca)
    plot_y_vs_x(train_sses_vs_k, x_label='k', y_label='sse',
                save_path='plot_sse_vs_k_%d'%do_pca)
    plot_y_vs_x(train_purities_vs_k, x_label='k', y_label='purities',
                save_path='plot_purity_vs_k_%d'%do_pca)



if __name__ == '__main__':
    args = load_args()
    x_train, y_train, x_test, y_test = load_data(args.root_dir)

    if args.pca == 0:
        pca = PCA(args.pca_retain_ratio)
        pca.fit(x_train)
        x_train = pca.transform(x_train)
        x_test = pca.transform(x_test)
        visualize(x_train, y_train)

    if args.kmeans == 1:
        apply_kmeans(args.pca, x_train, y_train, x_test, y_test, args.kmeans_max_iter, args.kmeans_max_k)

    print('Done')
