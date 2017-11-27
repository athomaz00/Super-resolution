# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 12:34:42 2017

@author: athomaz
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns
sns.set()

mybins  = np.arange(-500,500,50)
xlim = (-500,500)
ylim = (0,0.006)


#File names
fileName_CTR1 = 'homer-ampar-before-after-diff-trace-dist-CTR-1.xlsx'
fileName_CTR2 = 'homer-ampar-before-after-diff-trace-dist-CTR-2.xlsx'


###Only for AMPAR for now
fileName_CTR3 = 'homer-ampar-before-after-diff-trace-dist-CTR-3.xlsx'
fileName_CTR4 = 'homer-ampar-before-after-diff-trace-dist-CTR-4.xlsx'


data_CTR1 = pd.read_excel(fileName_CTR1)
data_CTR2 = pd.read_excel(fileName_CTR2)
data_CTR3 = pd.read_excel(fileName_CTR3)
data_CTR4 = pd.read_excel(fileName_CTR4)

distChange_CTR1 = data_CTR1['Distance_afe'] - data_CTR1['Distance_bef']
distChange_CTR2 = data_CTR2['Distance_afe'] - data_CTR2['Distance_bef']
distChange_CTR3 = data_CTR3['Distance_afe'] - data_CTR3['Distance_bef']
distChange_CTR4 = data_CTR4['Distance_afe'] - data_CTR4['Distance_bef']

distChange_CTR =  np.concatenate((distChange_CTR1, distChange_CTR2, distChange_CTR4))

fileName_LTD1  = 'homer-ampar-before-after-diff-trace-dist-LTD-1.xlsx'
fileName_LTD2  = 'homer-ampar-before-after-diff-trace-dist-LTD-2.xlsx'
fileName_LTD3  = 'homer-ampar-before-after-diff-trace-dist-LTD-3.xlsx'
fileName_LTD4  = 'homer-ampar-before-after-diff-trace-dist-LTD-4.xlsx'
fileName_LTD5  = 'homer-ampar-before-after-diff-trace-dist-LTD-5.xlsx'

data_LTD1 = pd.read_excel(fileName_LTD1)
data_LTD2 = pd.read_excel(fileName_LTD2)
data_LTD3 = pd.read_excel(fileName_LTD3)
data_LTD4 = pd.read_excel(fileName_LTD4)
data_LTD5 = pd.read_excel(fileName_LTD5)


distChange_LTD1 =  data_LTD1['Distance_afe'] - data_LTD1['Distance_bef']
distChange_LTD2 =  data_LTD2['Distance_afe'] - data_LTD2['Distance_bef']
distChange_LTD3 =  data_LTD3['Distance_afe'] - data_LTD3['Distance_bef']
distChange_LTD4 =  data_LTD4['Distance_afe'] - data_LTD4['Distance_bef']
distChange_LTD5 =  data_LTD5['Distance_afe'] - data_LTD5['Distance_bef']

distChange_LTD = np.concatenate((distChange_LTD1, distChange_LTD3)) #, distChange_LTD3))


plt.figure()
sns.distplot(distChange_LTD, bins=mybins, kde=False , hist_kws=dict(edgecolor="k", linewidth=1, normed=True))
sns.distplot(distChange_CTR, bins=mybins, kde=False, hist_kws=dict(edgecolor="k", linewidth=1, normed=True))


plt.xlim(xlim)
plt.ylim(ylim)
#plt.plot(xline, yline)
#plt.plot(xlinep, yline, 'r')
plt.legend()
plt.xlabel('Relative Distance (nm)')
plt.ylabel('Relative Frequency')
plt.title('Relative Distance Homer-Ampar')
plt.xticks(np.arange(-500,600,100))