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
##########Ampar Control##############
HA1 = pd.read_excel('dist-change-homer-ampar-ctr-1.xlsx', names= ['before', 'after'])
HA2 = pd.read_excel('dist-change-homer-ampar-ctr-2.xlsx', names= ['before', 'after'])
HA = pd.concat([HA1,HA2],ignore_index=True)
synNumbHACtr = HA.shape[0]
HA['dist'] = HA['after'] - HA['before']
HA['distclip'] = HA['dist'].copy()
#HA[HA.distclip<-500] = 0
#HA[HA.distclip>500] = 0
labelAMPARCtr = 'Control AMPAR ' + str(synNumbHACtr) +' synapses'


###############NMDAR Control####################
HN1 = pd.read_excel('dist-change-homer-nmdar-ctr-1.xlsx', names= ['before', 'after'])
HN2 = pd.read_excel('dist-change-homer-nmdar-ctr-2.xlsx', names= ['before', 'after'])
HN = pd.concat([HN1,HN2],ignore_index=True)
synNumbHNCtr = HN.shape[0]
HN['dist'] = HN['after'] - HN['before']
HN['distclip'] = HN['dist'].copy()
#HN[HN.distclip<-500] = 0
#HN[HN.distclip>500] = 0
labelNMDARCtr = 'Control NMDAR ' + str(synNumbHNCtr) +' synapses'
#sns.distplot(HA['distclip'], bins=25, rug=True, label='AMPAR')
#sns.distplot(HN['distclip'], bins=25, rug=True, label='NMDAR')

###########NMDAR LTD########################
HNL1 = pd.read_excel('dist-change-homer-nmdar-LTD-1.xlsx', names= ['before', 'after'])
HNL2 = pd.read_excel('dist-change-homer-nmdar-LTD-2.xlsx', names= ['before', 'after'])
HNL3 = pd.read_excel('dist-change-homer-nmdar-LTD-3.xlsx', names= ['before', 'after'])
HNL = pd.concat([HNL1, HNL2, HNL3],ignore_index=True)
synNumbHNLTD = HNL.shape[0]
HNL['dist'] = HNL['after'] - HNL['before']
HNL['distclip'] = HNL['dist'].copy()
#HNL[HNL.distclip<-500] = 0
#HNL[HNL.distclip>500] = 0
labelNMDARLTD = 'LTD NMDAR ' + str(synNumbHNLTD) +' synapses'

#plot NMDAR+Control
plt.figure()
sns.distplot(HNL['distclip'], bins=mybins, label=labelNMDARLTD, kde=False, hist_kws=dict(edgecolor="k", linewidth=1, normed=True))
sns.distplot(HN['distclip'], bins=mybins, label=labelNMDARCtr, kde=False, hist_kws=dict(edgecolor="k", linewidth=1, normed=True))
#plt.plot(xline, yline)
#plt.plot(xlinep, yline, 'r')
plt.legend()

#################AMPAR LTD######################
HAL = pd.read_excel('dist-change-homer-ampar-LTD-3.xlsx', names= ['before', 'after'])
synNumbHALTD = HAL.shape[0]
HAL['dist'] = HAL['after'] - HAL['before']
HAL['distclip'] = HAL['dist'].copy()
#HAL[HAL.distclip<-500] = 0
#HAL[HAL.distclip>500] = 0
labelAMPARLTD = 'LTD AMPAR ' + str(synNumbHALTD) +' synapses'






plt.figure()
sns.distplot(HAL['distclip'], bins=mybins,  label=labelAMPARLTD, kde=False ,hist_kws=dict(edgecolor="k", linewidth=1,  normed=True))
sns.distplot(HA['distclip'], bins=mybins, label=labelAMPARCtr, kde=False, hist_kws=dict(edgecolor="k", linewidth=1,  normed=True))
#plt.plot(xline, yline)
#plt.plot(xlinep, yline, 'r')
plt.legend()

plt.figure()
sns.distplot(HAL['distclip'], bins=mybins, label=labelAMPARLTD, kde=False,hist_kws=dict(edgecolor="k", linewidth=1, normed=True))
sns.distplot(HNL['distclip'], bins=mybins, label=labelNMDARLTD, kde=False,hist_kws=dict(edgecolor="k", linewidth=1, normed=True))
plt.legend()


plt.figure()
sns.distplot(HA['distclip'], bins=mybins, label=labelAMPARCtr, kde=False,hist_kws=dict(edgecolor="k", linewidth=1, normed=True))
sns.distplot(HN['distclip'], bins=mybins, label=labelNMDARCtr, kde=False,hist_kws=dict(edgecolor="k", linewidth=1, normed=True))
plt.legend()



#########Not dist change
bins2 = np.arange(0,500,5)
plt.figure()
sns.distplot(HAL['before'], bins=bins2, label='AMPAR Before', kde=False, hist_kws=dict(edgecolor="k", linewidth=1, normed=True))
plt.figure()
sns.distplot(HAL['after'], bins=bins2, label='AMPAR After', kde=False, hist_kws=dict(edgecolor="k", linewidth=1, normed=True))
plt.legend()


plt.figure()
sns.distplot(HNL['before'], bins=bins2, label='NMDAR Before', kde=False, hist_kws=dict(edgecolor="k", linewidth=1, normed=True))
#sns.distplot(HNL['after'],  bins=bins2, label='NMDARR After', kde=False, hist_kws=dict(edgecolor="k", linewidth=1, normed=True))
plt.legend()

plt.show()