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
fileName_CTR1 = 'homer-nmdar-before-after-diff-trace-dist-CTR-1.xlsx'
fileName_CTR2 = 'homer-nmdar-before-after-diff-trace-dist-CTR-2.xlsx'




data_CTR1 = pd.read_excel(fileName_CTR1)
data_CTR2 = pd.read_excel(fileName_CTR2)


distChange_CTR1 = data_CTR1['Distance_afe'] - data_CTR1['Distance_bef']
distChange_CTR2 = data_CTR2['Distance_afe'] - data_CTR2['Distance_bef']


distChange_CTR =  np.concatenate((distChange_CTR1, distChange_CTR2))

fileName_LTD1  = 'homer-nmdar-before-after-diff-trace-dist-LTD-1.xlsx'
fileName_LTD3  = 'homer-nmdar-before-after-diff-trace-dist-LTD-3.xlsx'


data_LTD1 = pd.read_excel(fileName_LTD1)
data_LTD3 = pd.read_excel(fileName_LTD3)


distChange_LTD1 =  data_LTD1['Distance_afe'] - data_LTD1['Distance_bef']
distChange_LTD3 =  data_LTD3['Distance_afe'] - data_LTD3['Distance_bef']


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