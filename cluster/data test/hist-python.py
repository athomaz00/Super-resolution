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
max_dist = 500.00



plt.close("all")
##########Ampar Control##############
HA1 = pd.read_excel('dist-change-homer-ampar-ctr-1.xlsx', names= ['before', 'after', 'dist_h-h'])
#HA2 = pd.read_excel('dist-change-homer-ampar-ctr-2.xlsx', names= ['before', 'after', 'dist_h-h'])
#HA3 = pd.read_excel('dist-change-homer-ampar-ctr-3.xlsx', names= ['before', 'after', 'dist_h-h'])
HA4 = pd.read_excel('dist-change-homer-ampar-ctr-4.xlsx', names= ['before', 'after', 'dist_h-h'])
HA = pd.concat([HA1, HA4],ignore_index=True)
mskHA = HA['dist_h-h']<=max_dist
HA['dist'] = HA['after'] - HA['before']
HA = HA[mskHA]
synNumbHACtr = HA.shape[0]
labelAMPARCtr = 'Control AMPAR ' + str(synNumbHACtr) +' synapses'

#################AMPAR LTD######################
#HAL1 = pd.read_excel('dist-change-homer-ampar-LTD-1.xlsx', names= ['before', 'after', 'dist_h-h'])
#HAL2 = pd.read_excel('dist-change-homer-ampar-LTD-2.xlsx', names= ['before', 'after', 'dist_h-h'])
HAL3 = pd.read_excel('dist-change-homer-ampar-LTD-3.xlsx', names= ['before', 'after', 'dist_h-h'])
HAL4 = pd.read_excel('dist-change-homer-ampar-LTD-4.xlsx', names= ['before', 'after', 'dist_h-h'])
#HAL5 = pd.read_excel('dist-change-homer-ampar-LTD-5.xlsx', names= ['before', 'after', 'dist_h-h'])
HAL = pd.concat([HAL3, HAL4], ignore_index=True)
mskHAL = HAL['dist_h-h']<=max_dist
HAL['dist'] = HAL['after'] - HAL['before']
HAL = HAL[mskHAL]
synNumbHALTD = HAL.shape[0]

labelAMPARLTD = 'LTD AMPAR ' + str(synNumbHALTD) +' synapses'


plt.figure()
sns.distplot(HAL['dist'], bins=mybins,  label=labelAMPARLTD, kde=False ,hist_kws=dict(edgecolor="k", linewidth=1,  normed=True))
sns.distplot(HA['dist'], bins=mybins, label=labelAMPARCtr, kde=False, hist_kws=dict(edgecolor="k", linewidth=1,  normed=True))
plt.xlim(xlim)
#plt.plot(xline, yline)
#plt.plot(xlinep, yline, 'r')
plt.legend()



###############NMDAR Control####################
HN1 = pd.read_excel('dist-change-homer-nmdar-ctr-1.xlsx', names= ['before', 'after', 'dist_h-h'])
HN2 = pd.read_excel('dist-change-homer-nmdar-ctr-2.xlsx', names= ['before', 'after', 'dist_h-h'])
HN = pd.concat([HN1,HN2],ignore_index=True)
mskHN = HN['dist_h-h']<=max_dist
HN['dist'] = HN['after'] - HN['before']
HN = HN[mskHN]
synNumbHNCtr = HN.shape[0]
labelNMDARCtr = 'Control NMDAR ' + str(synNumbHNCtr) +' synapses'


###########NMDAR LTD########################
#HNL = pd.read_excel('dist-change-homer-nmdar-LTD-1.xlsx', names= ['before', 'after', 'dist_h-h'])
#HNL = pd.read_excel('dist-change-homer-nmdar-LTD-2.xlsx', names= ['before', 'after', 'dist_h-h'])
HNL = pd.read_excel('dist-change-homer-nmdar-LTD-3.xlsx', names= ['before', 'after', 'dist_h-h'])
#HNL = pd.concat([HNL2, HNL3],ignore_index=True)
mskHNL = HNL['dist_h-h']<=max_dist
HNL['dist'] = HNL['after'] - HNL['before']
HNL = HNL[mskHNL]
synNumbHNLTD = HNL.shape[0]

labelNMDARLTD = 'LTD NMDAR ' + str(synNumbHNLTD) +' synapses'

#plot NMDAR+Control
plt.figure()
sns.distplot(HNL['dist'], bins=mybins, label=labelNMDARLTD, kde=False, hist_kws=dict(edgecolor="k", linewidth=1, normed=True))
sns.distplot(HN['dist'], bins=mybins, label=labelNMDARCtr, kde=False, hist_kws=dict(edgecolor="k", linewidth=1, normed=True))
#plt.plot(xline, yline)
#plt.plot(xlinep, yline, 'r')
plt.legend()





plt.figure()
sns.distplot(HAL['dist'], bins=mybins, label=labelAMPARLTD, kde=False,hist_kws=dict(edgecolor="k", linewidth=1, normed=True))
sns.distplot(HNL['dist'], bins=mybins, label=labelNMDARLTD, kde=False,hist_kws=dict(edgecolor="k", linewidth=1, normed=True))
plt.legend()


plt.figure()
sns.distplot(HA['dist'], bins=mybins, label=labelAMPARCtr, kde=False,hist_kws=dict(edgecolor="k", linewidth=1, normed=True))
sns.distplot(HN['dist'], bins=mybins, label=labelNMDARCtr, kde=False,hist_kws=dict(edgecolor="k", linewidth=1, normed=True))
plt.legend()



#########Not dist change
bins2 = np.arange(0,500,10)
plt.figure()
sns.distplot(HAL['before'], bins=bins2, label='AMPAR Before', kde=False, hist_kws=dict(edgecolor="k", linewidth=1))
plt.legend()
plt.xlim(0,500)
plt.figure()
sns.distplot(HAL['after'], bins=bins2, label='AMPAR After', kde=False, hist_kws=dict(edgecolor="k", linewidth=1 ))
plt.legend()
plt.xlim(0,500)
plt.figure()
sns.distplot(HA['before'], bins=bins2, label='AMPAR Ctr Before', kde=False, hist_kws=dict(edgecolor="k", linewidth=1 ))
plt.xlim(0,500)
plt.figure()
sns.distplot(HA['after'], bins=bins2, label='AMPAR Ctr After', kde=False, hist_kws=dict(edgecolor="k", linewidth=1 ))
plt.xlim(0,500)

plt.legend()


plt.figure()
sns.distplot(HNL['before'], bins=bins2, label='NMDAR Before', kde=False, hist_kws=dict(edgecolor="k", linewidth=1, normed=True))
plt.legend()
plt.figure()
sns.distplot(HNL['after'],  bins=bins2, label='NMDARR After', kde=False, hist_kws=dict(edgecolor="k", linewidth=1, normed=True))
plt.legend()
plt.figure()
sns.distplot(HN['before'], bins=bins2, label='NMDAR CTr Before', kde=False, hist_kws=dict(edgecolor="k", linewidth=1, normed=True))
plt.legend()
plt.figure()
sns.distplot(HN['after'],  bins=bins2, label='NMDARR Ctr After', kde=False, hist_kws=dict(edgecolor="k", linewidth=1, normed=True))
plt.legend()
plt.show()