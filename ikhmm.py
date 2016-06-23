# 
# DESCRIPTION
# Copyright (C) Weicong Kong 2016. All Right Reserved
#

import numpy as np
import pandas as pd
import copy
from sklearn.cluster import KMeans

class IterativeKmeansHMM:

    K = -1                  # the hidden state dimension
    obs_distns = list()     # observation distribution
    trans_mat = None        # trainsition matrix
    init_prob = None        # initial state probability

    ob = list()             # the observation sequence
    T = -1                  # the length of the observation sequence
    max_K = 9999            # the maximum hidden state dimension
    std_thres = 50           # the standard deviation threshold for iterative kmeans
    states = -1             # the hidden state labels

    def __init__(self, ob, max_k=10, std_thres=1):
        self.ob = ob
        self.max_K = max_k
        self.std_thres = std_thres
        self.T = len(ob)

    def fit(self):
        centers = self.iterative_kmeans()

        # convert to piece-wise constant profile
        pwc = np.zeros(self.T)
        for t in range(self.T):
            pwc[t] = centers[self.states[t]]

        # fit transition matrix
        trans_mat = np.zeros((self.K, self.K))
        for t in range(1, self.T):
            prev_state = self.states[t-1]
            curr_state = self.states[t]
            trans_mat[prev_state, curr_state] += 1
        trans_mat = trans_mat.astype(float) / trans_mat.sum(axis=1).reshape((self.K, 1))
        self.trans_mat = trans_mat

        # fit Gaussian for emissions
        obs_distns = pd.DataFrame(columns=['mu', 'sigma'])
        for k in range(self.K):
            mu, sigma = fit_weighted_gaussian(self.ob[self.states == k])
            obs_distns.loc[k, :] = [mu, sigma]
        self.obs_distns = obs_distns

        # fit initial state probability
        init_prob = np.zeros(self.K)
        for k in range(self.K):
            init_prob[k] = (self.states == k).sum() / float(self.T)
        self.init_prob = init_prob

        print('init_prob: \n%s\n' % self.init_prob)
        print('trans_mat: \n%s\n' % self.trans_mat)
        print('obs_distns: \n%s\n' % self.obs_distns)

        return

    def iterative_kmeans(self):

        # basic stats
        h, edges = np.histogram(self.ob, bins=100)
        bin_centers = (edges[:-1] + edges[1:]) / 2
        values = np.unique(h)
        std_list = list()
        perc_list = list()
        center_list = list()
        label_list = list()

        for n_clusters in range(2, self.max_K + 1):

            # evenly initialise starting points for iterative kmeans
            init = np.zeros((n_clusters,))
            for k in range(n_clusters):
                init[k] = min(self.ob) + k * (max(self.ob) / n_clusters)

            # adjust starting points to the closest density center
            for i in range(n_clusters):
                if i == 0:
                    continue
                for j in range(n_clusters):
                    ith = len(values) - 1 - j
                    if ith < 0:
                        ith = 0
                    ith_largest = values[ith]
                    idx_ith_largest = np.where(h == ith_largest)

                    if np.abs(bin_centers[idx_ith_largest[0][-1]] - init[i]) <= 0.8 * (values[-1] / self.max_K):
                        init[i] = bin_centers[idx_ith_largest[0][-1]]

            # perform k-means with the starting points
            kmeans = KMeans(n_clusters=n_clusters, n_init=1, init=init.reshape(n_clusters, 1))
            kmeans.fit(self.ob.values.reshape(self.T, 1))
            labels = kmeans.labels_
            centers = kmeans.cluster_centers_

            # get rid of empty cluster
            nan_idx = np.where(np.isnan(centers))
            for i in range(len(nan_idx[0])):
                labels[labels > nan_idx[i]] = labels[labels > nan_idx[i]] - 1
                nan_idx = nan_idx - 1
            centers = centers[~np.isnan(centers)]

            # calcualte std
            stds = np.zeros((len(centers),))
            perc = np.zeros((len(centers),))
            for i in range(len(centers)):
                stds[i] = np.std(self.ob[labels == i])
                perc[i] = float(len(self.ob[labels == i])) / self.T
                if stds[i] == 0:
                    stds[i] = 0.01
            std_list.append(stds)
            perc_list.append(perc)
            center_list.append(centers)
            label_list.append(labels)

            # test if std threshold is satisfied
            if max(stds) <= self.std_thres:
                break

        # find the lowest average std clustering if we hit the max_K
        weighted_mean_std = np.zeros((len(std_list),))
        for i in range(len(std_list)):
            weighted_mean_std[i] = np.sum(std_list[i] * perc_list[i])
        pick = np.where(weighted_mean_std == min(weighted_mean_std))
        pick = pick[0][0]  # get the first index if there are multiple

        centers = center_list[pick]
        labels = label_list[pick]

        # sort the center from low to high, also the label
        sorted_labels = centers.argsort()
        centers.sort()
        for new_label, old_label in enumerate(sorted_labels):
            labels[labels == old_label] = new_label * 10
        labels = labels / 10

        # remove clusters which have only or less than min_members
        min_members = 1
        orig_labels = copy.copy(labels)
        for idx, label in enumerate(np.unique(labels)):
            if len(labels[labels == label]) <= min_members:
                center = centers[idx]
                distance = np.abs(centers - center)
                for idx2, label2 in enumerate(np.unique(orig_labels)):
                    if len(labels[labels == label2]) <= min_members:
                        distance[idx2] = np.inf  # set the distance to inf to those clusters not having enough members
                cluster_to_merge = np.argmin(distance)
                self.ob[labels == label] = centers[cluster_to_merge]  # reset the 'outlier' to new cluster center
                labels[labels == label] = cluster_to_merge
                centers[idx] = np.nan

        # remove nan centers and re-labelling
        next_label = 0
        for idx, center in enumerate(centers):
            if labels[labels == idx].size != 0:
                labels[labels == idx] = next_label
                next_label += 1
        centers = centers[~np.isnan(centers)]

        # update relevant attributes
        self.states = labels
        self.K = len(centers)

        return centers

def fit_weighted_gaussian(x, w=None):
    if w is None:
        w = np.ones(x.shape).astype(float)

    mu = np.dot(w.transpose(), x) / sum(w)
    variance = np.dot(w.transpose(), x ** 2) / sum(w) - mu ** 2
    if variance <= 0:
        variance = 0.0001
    sigma = np.sqrt(variance)
    return mu, sigma





