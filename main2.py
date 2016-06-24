# 
# DESCRIPTION
# Copyright (C) Weicong Kong 2016. All Right Reserved
#

import os
import pandas as pd
import collections
from nilmtk import DataSet, HDFDataStore
import nilm
from ikhmm import *
from evaluator import *

pd.set_option('display.width', 200)

train = DataSet(os.path.join('data', 'redd.h5'))
# train.set_window(end='30-4-2011')
test = DataSet(os.path.join('data', 'redd.h5'))
# test.set_window(start='30-4-2011')

buildings = [6]  # house 3 does not have electric stove
apps = ['fridge', 'light', 'dish washer', 'washer dryer', 'electric stove']
for building in buildings:
    index_train = train.buildings[building].elec['fridge'].load().next().resample('1min', np.median).index
    index_test = test.buildings[building].elec['fridge'].load().next().resample('1min', np.median).index
    data_train = pd.DataFrame(index=index_train, columns=apps)
    data_test = pd.DataFrame(index=index_test, columns=apps)
    hmms = collections.OrderedDict()
    step_thres = 50
    for app in apps:
        load_train = train.buildings[building].elec[app].load().next()
        load_train.fillna(0, inplace=True)
        load_train = load_train.resample('1min', np.median)
        load_train.fillna(0, inplace=True)

        load_test = test.buildings[building].elec[app].load().next()
        load_test.fillna(0, inplace=True)
        load_test = load_test.resample('1min', np.median)
        load_test.fillna(0, inplace=True)

        data_train[app] = load_train.loc[data_train.index, :].values
        data_test[app] = load_test.loc[data_test.index, :].values
        hmm = IterativeKmeansHMM(data_train[app], max_k=10, std_thres=step_thres)
        hmm.fit()
        hmms[app] = hmm
    aggregate = data_test.sum(axis=1)
    solver = nilm.SIQP(aggregate, hmms, step_thr=step_thres)
    solver.solve()
    ground_truth = data_test[apps]
    evaluator = Evaluator(ground_truth, solver.estimate, solver.aggregate, standby=10)
    print evaluator.report
    evaluator.report.to_csv(os.path.join('data', 'REDD_report_house_%i.csv' % (building)))


