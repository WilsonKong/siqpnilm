# 
# DESCRIPTION
# Copyright (C) Weicong Kong 2016. All Right Reserved
#

import pandas as pd
import os
from libDataLoaders import dataset_loader
import collections
import nilm
from ikhmm import *
from evaluator import *
from hmmlearn import hmm

pd.set_option('display.width', 200)
np.set_printoptions(linewidth=100)

model_args = pd.read_csv('model_building_args.csv', header=None)
sol_times = pd.DataFrame()

n_appliance = 11  # from 1 - 19
for n_appliance in range(1, 20):
    args = model_args.loc[n_appliance - 1, :]

    modeldb = args[0]
    dataset = args[1]
    precision = args[2]
    max_obs = args[3]
    denoised = args[4] == 'denoised'
    max_states = args[5]
    folds = args[6]
    ids = args[7].split(',')

    datasets_dir = './data/%s.csv'

    data = dataset_loader(datasets_dir % dataset, ids, precision=precision, denoised=denoised)

    # test ikhmm
    ob = data.BME.head(1440)
    hmm = IterativeKmeansHMM(ob, max_k=10, std_thres=1)
    centers = hmm.iterative_kmeans()

    # showing the first day
    # data.WHE.iloc[0:1440].plot()  # whole house current profile for day 1
    # data.iloc[0:1440, 1:n_appliance+1].sum(1).plot()

    # defining problem
    days = 3  # as the same length in SparseNILM demo
    length = 1440 * days  # 1 minute interval, 7 days
    aggregate = data.WHE.tail(36 * 1440)  # test on the last 36 days
    hmms = collections.OrderedDict()
    app_ids = list(data)[1:-1]
    for app_id in app_ids:
        hmm = IterativeKmeansHMM(data[app_id].head(length), max_k=4, std_thres=1)
        hmm.fit()
        hmms[app_id] = hmm

    solver = nilm.SIQP(aggregate, hmms=hmms, step_thr=2)
    solver.solve()
    ground_truth = data[app_ids].tail(36 * 1440)
    evaluator = Evaluator(ground_truth, solver.estimate, solver.aggregate)
    print evaluator.report
    evaluator.report.to_csv(os.path.join('data', 'AMPdsR1_report_%i_appliances_for_train_%i_days_test_36_days.csv' % (n_appliance, days)))
    sol_times.loc[n_appliance, 'sol_time'] = solver.sol_time
    sol_times.to_csv(os.path.join('data', 'sol_time_vs_number_appliances.csv'))
    # evaluator.show()

sol_times.to_csv(os.path.join('data', 'sol_time_vs_number_appliances.csv'))