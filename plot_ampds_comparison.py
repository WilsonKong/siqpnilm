"""
DESCRIPTION
Copyright (C) Weicong Kong, the University of Sydney, 2016
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import os

pd.set_option('display.width', 200)

kolter_cmp = pd.read_csv(os.path.join('data', 'ampds_kolter_cmp.csv'), index_col=0)
f1_cmp = pd.read_csv(os.path.join('data', 'ampds_f1_cmp.csv'), index_col=0)

n_groups = len(kolter_cmp)
index = kolter_cmp.index.values

bar_width = 0.35
opacity = 0.4
font = {'font.size': 14}
fontsize = 18
lagend_font = 10
text_loc = -35

fig = plt.figure()

for i in range(1, 5):
    ax = fig.add_subplot(2, 2, i)
    siqp_scenario = list(kolter_cmp)[2 * (i - 1)]
    sparsenilm_scenario = list(kolter_cmp)[2 * (i - 1) + 1]
    rects1 = plt.bar(
        index - bar_width, kolter_cmp[siqp_scenario] * 100, bar_width, alpha=opacity, color='b', label=siqp_scenario
    )
    rects2 = plt.bar(
        index, kolter_cmp[sparsenilm_scenario] * 100, bar_width,
        alpha=opacity, color='r', label=sparsenilm_scenario
    )
    plt.xlabel('Number of Appliances', fontsize=fontsize)
    plt.ylabel('Estimate Accuracy %', fontsize=fontsize)
    plt.text(10, text_loc, ['(a)', '(b)', '(c)', '(d)'][i - 1], ha='center', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.legend(fontsize=lagend_font)
    # plt.grid()
    plt.ylim(0, 110)
    ax.yaxis.grid(True)

plt.tight_layout(w_pad=-1, h_pad=-0.5)

fig = plt.figure()

for i in range(1, 5):
    ax = fig.add_subplot(2, 2, i)
    siqp_scenario = list(f1_cmp)[2 * (i - 1)]
    sparsenilm_scenario = list(f1_cmp)[2 * (i - 1) + 1]
    rects1 = plt.bar(
        index - bar_width, f1_cmp[siqp_scenario] * 100, bar_width, alpha=opacity, color='b', label=siqp_scenario
    )
    rects2 = plt.bar(
        index, f1_cmp[sparsenilm_scenario] * 100, bar_width,
        alpha=opacity, color='r', label=sparsenilm_scenario
    )
    plt.xlabel('Number of Appliances', fontsize=fontsize)
    plt.ylabel('f1-score %', fontsize=fontsize)
    plt.text(10, text_loc+5, ['(a)', '(b)', '(c)', '(d)'][i - 1], ha='center', fontsize=fontsize)
    plt.legend(fontsize=lagend_font)
    # plt.grid()
    plt.ylim(0, 117)
    ax.yaxis.grid(True)

plt.tight_layout(w_pad=-1)

# solve time versus # appliances
fig = plt.figure()
soltime = pd.read_csv(os.path.join('data', 'sol_time_vs_number_appliances.csv'), index_col=0)
plt.plot(soltime, marker='o')
plt.xlabel('Number of Appliances', fontsize=fontsize)
plt.ylabel('Solving Time [s]', fontsize=fontsize)
plt.grid()
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.tight_layout()

plt.show()