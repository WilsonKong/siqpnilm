# 
# The library to evaluate NILM results
# Copyright (C) Weicong Kong 2016. All Right Reserved
#

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from math import sqrt


class Evaluator:

    n_appliances = 0
    T = 0
    ground_truth = pd.DataFrame()      # ground truth of appliance loads
    estimate = pd.DataFrame()          # estimate of appliance loads
    aggregate = pd.DataFrame()         # whole house load sequence
    timestamps = None
    standby = 0.0                      # the standby power/current, below that considered OFF

    count_tp = None
    count_tn = None
    count_fp = None
    count_fn = None

    count_inacc = None
    count_atp = None
    count_itp = None

    measure_est = None
    measure_truth = None
    measure_diff = None
    measure_diff_sq = None

    report = None


    def __init__(self, ground_truth, estimate, aggregate, standby=1.0):
        self.ground_truth = ground_truth
        self.estimate = estimate
        self.estimate.columns = self.ground_truth.columns
        self.aggregate = aggregate
        self.n_appliances = len(list(ground_truth))
        self.timestamps = ground_truth.index
        self.T = len(self.timestamps)
        self.report = pd.DataFrame(columns=self.ground_truth.columns.tolist() + ['OVERALL'])
        self.standby = standby
        self.update()

    def update(self, standby=1.0):
        self.estimate.index = range(self.T)
        self.ground_truth.index = range(self.T)
        apps = list(self.ground_truth)
        self.count_tp = np.zeros(self.n_appliances)
        self.count_tn = np.zeros(self.n_appliances)
        self.count_fp = np.zeros(self.n_appliances)
        self.count_fn = np.zeros(self.n_appliances)

        # tp, tn, fp, fn counts
        tp = (self.estimate >= standby) & (self.ground_truth >= standby)
        tn = (self.estimate < standby) & (self.ground_truth < standby)
        fp = (self.estimate >= standby) & (self.ground_truth < standby)
        fn = (self.estimate < standby) & (self.ground_truth >= standby)
        self.count_tp = tp.sum(0)
        self.count_tn = tn.sum(0)
        self.count_fp = fp.sum(0)
        self.count_fn = fn.sum(0)

        # generating report
        self.report.loc['TP', :-1] = self.count_tp
        self.report.loc['TP', 'OVERALL'] = self.count_tp.sum()
        self.report.loc['TN', :-1] = self.count_tn
        self.report.loc['TN', 'OVERALL'] = self.count_tn.sum()
        self.report.loc['FP', :-1] = self.count_fp
        self.report.loc['FP', 'OVERALL'] = self.count_fp.sum()
        self.report.loc['FN', :-1] = self.count_fn
        self.report.loc['FN', 'OVERALL'] = self.count_fn.sum()

        # calculating metrics
        self.precision()
        self.recall()
        self.f1_score()
        self.kolter()
        self.energy_percent_estimate()
        self.energy_percent_truth()

        return

    def energy_percent_estimate(self):
        func_name = self.energy_percent_estimate.__name__
        self.report.loc[func_name, :-1] = \
            quotient(self.estimate.sum(0), [self.estimate.sum(0).sum()] * self.n_appliances)
        self.report.loc[func_name, 'OVERALL'] = 1
        return

    def energy_percent_truth(self):
        func_name = self.energy_percent_truth.__name__
        self.report.loc[func_name, :-1] = \
            quotient(self.ground_truth.sum(0), [self.ground_truth.sum(0).sum()] * self.n_appliances)
        self.report.loc[func_name, 'OVERALL'] = 1
        return

    def f1_score(self):
        func_name = self.f1_score.__name__
        self.report.loc[func_name, :] = \
            2.0 * quotient(self.precision() * self.recall(), (self.precision() + self.recall()))
        return self.report.loc[func_name]

    def kolter(self):
        func_name = self.kolter.__name__
        # appliance-wise kolter metric
        self.report.loc[func_name, :-1] = \
            1.0 - quotient((self.estimate - self.ground_truth).abs().sum(0), 2.0 * self.ground_truth.sum(0))
        self.report.loc[func_name, 'OVERALL'] = \
            1.0 - quotient((self.estimate - self.ground_truth).abs().sum(0).sum(), 2.0 * self.ground_truth.sum(0).sum())
        return

    def precision(self):
        func_name = self.precision.__name__
        self.report.loc[func_name, :] = quotient(self.report.loc['TP'], (self.report.loc['TP'] + self.report.loc['FP']))
        return self.report.loc[func_name]

    def recall(self):
        func_name = self.recall.__name__
        self.report.loc[func_name, :] = \
            quotient(self.report.loc['TP'], (self.report.loc['TP'] + self.report.loc['FN']))
        return self.report.loc[func_name]

    def show(self, savefilename=None):
        fig = plt.figure(figsize=(16, 10))
        font = {'size': 14}
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        ax1.set_title('Ground Truth')
        ax2.set_title('NILM Estimate')
        y1 = np.row_stack(self.ground_truth.values.T)
        y2 = np.row_stack(self.estimate.values.T.astype(float))
        x = self.aggregate.index
        stack1 = ax1.stackplot(x, y1, linewidth=0)
        stack2 = ax2.stackplot(x, y2, linewidth=0)
        proxy_rects1 = [Rectangle((0, 0), 1, 1, fc=pc.get_facecolor()[0]) for pc in stack1]
        proxy_rects2 = [Rectangle((0, 0), 1, 1, fc=pc.get_facecolor()[0]) for pc in stack2]
        labels = list(self.ground_truth)
        ax1.legend(proxy_rects1, labels)
        ax2.legend(proxy_rects2, labels)
        plt.tight_layout()
        if savefilename is None:
            plt.show()
        else:
            plt.savefig(savefilename)
        plt.close(fig)
        return




def quotient(n, d):
    """From SparseNILM, Stephen Makonin, Modified for array calculation by Weicong Kong"""
    def scalar(n, d):
        a = -1
        if n != 0.0 and d == 0.0:
            a = 0.0
        elif n == 0.0 and d == 0.0:
            a = 0.0
        else:
            a = float(n) / float(d)
        return a
    try:
        iter(n)
        if len(n) == len(d):
            return np.array(map(scalar, n, d))
    except TypeError:
        return scalar(n, d)

def mean(a):
    """From SparseNILM, Stephen Makonin."""
    return float(sum(a)) / float(len(a))
