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

#############AMPAR Control Close to Soma
HCC = pd.read_excel('dist-change-homer-ampar-close4000-CTR.xlsx', names= ['before', 'after'])
#HA2 = pd.read_excel('dist-change-homer-ampar-ctr-2.xlsx', names= ['before', 'after'])
#HA = pd.concat([HA1,HA2],ignore_index=True)
synNumbHACtr = HCC.shape[0]
HCC['dist'] = HCC['after'] - HCC['before']
labelAMPARCloseCTR= 'CTR AMPAR close'




plt.figure()
sns.distplot(HAC['dist'], bins=mybins,  label=labelAMPARClose, kde=False ,hist_kws=dict(edgecolor="k", linewidth=1,  normed=True))
sns.distplot(HCC['dist'], bins=mybins, label=labelAMPARCloseCTR, kde=False, hist_kws=dict(edgecolor="k", linewidth=1,  normed=True))
#plt.plot(xline, yline)
#plt.plot(xlinep, yline, 'r')
plt.legend()

##########Ampar Far from soma##############
HAF = pd.read_excel('dist-change-homer-ampar-far4000.xlsx', names= ['before', 'after'])
#HA2 = pd.read_excel('dist-change-homer-ampar-ctr-2.xlsx', names= ['before', 'after'])
#HA = pd.concat([HA1,HA2],ignore_index=True)
synNumbHACtr = HAF.shape[0]
HAF['dist'] = HAF['after'] - HAF['before']
labelAMPARFar= 'LTD AMPAR far'

##########Ampar CTR Far from soma##############
HCF = pd.read_excel('dist-change-homer-ampar-far4000-CTR.xlsx', names= ['before', 'after'])
#HA2 = pd.read_excel('dist-change-homer-ampar-ctr-2.xlsx', names= ['before', 'after'])
#HA = pd.concat([HA1,HA2],ignore_index=True)
synNumbHACtr = HCF.shape[0]
HCF['dist'] = HCF['after'] - HCF['before']
labelAMPARFarCTR= 'CTR AMPAR far'

plt.figure()
sns.distplot(HAF['dist'], bins=mybins,  label=labelAMPARFar, kde=False ,hist_kws=dict(edgecolor="k", linewidth=1,  normed=True))
sns.distplot(HCF['dist'], bins=mybins, label=labelAMPARFarCTR, kde=False, hist_kws=dict(edgecolor="k", linewidth=1,  normed=True))
#plt.plot(xline, yline)
#plt.plot(xlinep, yline, 'r')
plt.legend()



###############NMDAR LTD Close to Soma####################
HNC = pd.read_excel('dist-change-homer-nmdar-close4000.xlsx', names= ['before', 'after'])
#HN2 = pd.read_excel('dist-change-homer-nmdar-ctr-2.xlsx', names= ['before', 'after'])
#HN = pd.concat([HN1,HN2],ignore_index=True)
synNumbHNCtr = HNC.shape[0]
HNC['dist'] = HNC['after'] - HNC['before']
labelNMDARClose = 'LTD NMDAR Close'




##########NMDAR CTR Close to soma##############
HNCT = pd.read_excel('dist-change-homer-nmdar-close4000-CTR.xlsx', names= ['before', 'after'])
#HA2 = pd.read_excel('dist-change-homer-ampar-ctr-2.xlsx', names= ['before', 'after'])
#HA = pd.concat([HA1,HA2],ignore_index=True)
synNumbHACtr = HNCT.shape[0]
HNCT['dist'] = HNCT['after'] - HNCT['before']
labelNMDARCloseCTR= 'CTR NMDAR close'


plt.figure()
sns.distplot(HNC['dist'], bins=mybins,  label=labelNMDARClose, kde=False ,hist_kws=dict(edgecolor="k", linewidth=1,  normed=True))
sns.distplot(HNCT['dist'], bins=mybins, label=labelNMDARCloseCTR, kde=False, hist_kws=dict(edgecolor="k", linewidth=1,  normed=True))
#plt.plot(xline, yline)
#plt.plot(xlinep, yline, 'r')
plt.legend()

###############NMDAR LTD FAR to Soma####################
HNF = pd.read_excel('dist-change-homer-nmdar-far4000.xlsx', names= ['before', 'after'])
#HN2 = pd.read_excel('dist-change-homer-nmdar-ctr-2.xlsx', names= ['before', 'after'])
#HN = pd.concat([HN1,HN2],ignore_index=True)
synNumbHNCtr = HNF.shape[0]
HNF['dist'] = HNF['after'] - HNF['before']
labelNMDARFar = 'LTD NMDAR FAR'

###############NMDAR LTD FAR to Soma####################
HNFC = pd.read_excel('dist-change-homer-nmdar-far4000-CTR.xlsx', names= ['before', 'after'])
#HN2 = pd.read_excel('dist-change-homer-nmdar-ctr-2.xlsx', names= ['before', 'after'])
#HN = pd.concat([HN1,HN2],ignore_index=True)
synNumbHNCtr = HNFC.shape[0]
HNFC['dist'] = HNFC['after'] - HNFC['before']
labelNMDARFarCTR = 'CTR NMDAR FAR'


plt.figure()
sns.distplot(HNF['dist'], bins=mybins,  label=labelNMDARFar, kde=False ,hist_kws=dict(edgecolor="k", linewidth=1,  normed=True))
sns.distplot(HNFC['dist'], bins=mybins, label=labelNMDARFarCTR, kde=False, hist_kws=dict(edgecolor="k", linewidth=1,  normed=True))
#plt.plot(xline, yline)
#plt.plot(xlinep, yline, 'r')
plt.legend()



###LTD AMPAR VS LTD NMDAR close to soma#
plt.figure()
sns.distplot(HAF['dist'], bins=mybins,  label=labelAMPARClose, kde=False ,hist_kws=dict(edgecolor="k", linewidth=1,  normed=True))
sns.distplot(HNC['dist'], bins=mybins, label=labelNMDARClose, kde=False, hist_kws=dict(edgecolor="k", linewidth=1,  normed=True))
#plt.plot(xline, yline)
#plt.plot(xlinep, yline, 'r')
plt.legend()

###LTD AMPAR VS LTD NMDAR far from soma#
plt.figure()
sns.distplot(HAC['dist'], bins=mybins,  label=labelAMPARFar, kde=False ,hist_kws=dict(edgecolor="k", linewidth=1,  normed=True))
sns.distplot(HNF ['dist'], bins=mybins, label=labelNMDARFar, kde=False, hist_kws=dict(edgecolor="k", linewidth=1,  normed=True))
#plt.plot(xline, yline)
#plt.plot(xlinep, yline, 'r')
plt.legend()

#########Not dist change
bins2 = np.arange(0,500,15)
plt.figure()
sns.distplot(HAC['before'], bins=bins2, label='Ampar Close Before', kde=False, hist_kws=dict(edgecolor="k", linewidth=1))
sns.distplot(HAC['after'], bins=bins2, label='Ampar Close After', kde=False, hist_kws=dict(edgecolor="k", linewidth=1))
plt.legend()


plt.figure()
sns.distplot(HNC['before'], bins=bins2, label='NMDAR Close Before', kde=False, hist_kws=dict(edgecolor="k", linewidth=1))
sns.distplot(HNC['after'],  bins=bins2, label='NMDAR Close After', kde=False, hist_kws=dict(edgecolor="k", linewidth=1))
plt.legend()


plt.show()