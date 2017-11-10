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

plt.close("all")
##########Ampar Close to soma##############
HAC = pd.read_excel('dist-change-homer-ampar-close4000.xlsx', names= ['before', 'after'])
#HA2 = pd.read_excel('dist-change-homer-ampar-ctr-2.xlsx', names= ['before', 'after'])
#HA = pd.concat([HA1,HA2],ignore_index=True)
synNumbHACtr = HAC.shape[0]
HAC['dist'] = HAC['after'] - HAC['before']
labelAMPARClose= 'LTD AMPAR close'


###############NMDAR Control####################
HAF = pd.read_excel('dist-change-homer-ampar-far4000.xlsx', names= ['before', 'after'])
#HN2 = pd.read_excel('dist-change-homer-nmdar-ctr-2.xlsx', names= ['before', 'after'])
#HN = pd.concat([HN1,HN2],ignore_index=True)
synNumbHNCtr = HAF.shape[0]
HAF['dist'] = HAF['after'] - HAF['before']
labelAMPARFar = 'LTD AMPAR far'


plt.figure()
sns.distplot(HAC['dist'], bins=mybins,  label=labelAMPARClose, kde=False ,hist_kws=dict(edgecolor="k", linewidth=1,  normed=True))
sns.distplot(HAF['dist'], bins=mybins, label=labelAMPARFar, kde=False, hist_kws=dict(edgecolor="k", linewidth=1,  normed=True))
#plt.plot(xline, yline)
#plt.plot(xlinep, yline, 'r')
plt.legend()

##########NMDAR LTD Close to soma##############
HNC = pd.read_excel('dist-change-homer-nmdar-close4000.xlsx', names= ['before', 'after'])
#HA2 = pd.read_excel('dist-change-homer-ampar-ctr-2.xlsx', names= ['before', 'after'])
#HA = pd.concat([HA1,HA2],ignore_index=True)
synNumbHACtr = HNC.shape[0]
HNC['dist'] = HNC['after'] - HNC['before']
labelNMDARClose= 'LTD NMDAR close'


###############NMDAR LTD far from soma####################
HNF = pd.read_excel('dist-change-homer-nmdar-far4000.xlsx', names= ['before', 'after'])
#HN2 = pd.read_excel('dist-change-homer-nmdar-ctr-2.xlsx', names= ['before', 'after'])
#HN = pd.concat([HN1,HN2],ignore_index=True)
synNumbHNCtr = HNF.shape[0]
HNF['dist'] = HNF['after'] - HNF['before']
labelNMDARFar = 'LTD NMDAR far'


plt.figure()
sns.distplot(HNC['dist'], bins=mybins,  label=labelNMDARClose, kde=False ,hist_kws=dict(edgecolor="k", linewidth=1,  normed=True))
sns.distplot(HNF['dist'], bins=mybins, label=labelNMDARFar, kde=False, hist_kws=dict(edgecolor="k", linewidth=1,  normed=True))
#plt.plot(xline, yline)
#plt.plot(xlinep, yline, 'r')
plt.legend()


plt.show()