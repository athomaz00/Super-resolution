# -*- coding: utf-8 -*-
"""
Spyder Editor

Este é um arquivo de script temporário.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns
sns.set(color_codes=True)

mybins  = np.arange(-500,500,50)
xlim = (-500,500)

plt.close("all")

#################AMPAR LTD######################
HAL1 = pd.read_excel('dist-change-homer-ampar-LTD-2.xlsx', names= ['before', 'after', 'dist_h-h'])
HAL2 = pd.read_excel('dist-change-homer-ampar-LTD-2.xlsx', names= ['before', 'after', 'dist_h-h'])
HAL3 = pd.read_excel('dist-change-homer-ampar-LTD-2.xlsx', names= ['before', 'after', 'dist_h-h'])
HAL1['dist'] = HAL1['after'] - HAL1['before']
HAL2['dist'] = HAL2['after'] - HAL2['before']
HAL3['dist'] = HAL3['after'] - HAL3['before']






plt.figure()
sns.distplot(HAL1['dist'], bins=mybins,  label='1', kde=False ,hist_kws=dict(edgecolor="k", linewidth=1,  normed=True))
sns.distplot(HAL2['dist'], bins=mybins, label='2', kde=False, hist_kws=dict(edgecolor="k", linewidth=1,  normed=True))
sns.distplot(HAL3['dist'], bins=mybins, label='3', kde=False, hist_kws=dict(edgecolor="k", linewidth=1,  normed=True))
#plt.plot(xline, yline)
#plt.plot(xlinep, yline, 'r')
plt.xlim(xlim)
plt.legend()





#########Not dist change
bins2 = np.arange(0,500,15)
plt.figure()
sns.distplot(HAL1['before'], bins=bins2, label='AMPAR Before', kde=False, hist_kws=dict(edgecolor="k", linewidth=1, normed=True))
plt.legend()
plt.figure()
sns.distplot(HAL1['after'], bins=bins2, label='AMPAR After', kde=False, hist_kws=dict(edgecolor="k", linewidth=1, normed=True))

plt.legend()


plt.figure()
sns.distplot(HAL2['before'], bins=bins2, label='NMDAR Before', kde=False, hist_kws=dict(edgecolor="k", linewidth=1, normed=True))
plt.legend()
plt.figure()
sns.distplot(HAL2['after'],  bins=bins2, label='NMDARR After', kde=False, hist_kws=dict(edgecolor="k", linewidth=1, normed=True))
plt.legend()

plt.show()