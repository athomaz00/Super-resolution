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

mybins  = np.arange(-500,500,25)
xlim = (-500,500)
ylim = (0,0.006)


#File names
fileName_CTR5 = 'homer-nmdar-before-after-distance-CTR-5.xlsx'
fileName_CTR6 = 'homer-nmdar-before-after-distance-CTR-6.xlsx'




data_CTR5 = pd.read_excel(fileName_CTR5)
data_CTR6 = pd.read_excel(fileName_CTR6)
#data_CTR3 = pd.read_excel(fileName_CTR3)
#data_CTR4 = pd.read_excel(fileName_CTR4)

distChange_CTR5 = data_CTR5['Distance_afe'] - data_CTR5['Distance_bef']
distChange_CTR6 = data_CTR6['Distance_afe'] - data_CTR6['Distance_bef']
#distChange_CTR3 = data_CTR3['Distance_afe'] - data_CTR3['Distance_bef']
#distChange_CTR4 = data_CTR4['Distance_afe'] - data_CTR4['Distance_bef']

distChange_CTR = np.concatenate((distChange_CTR5, distChange_CTR6)) #, distChange_CTR4))

#File names
#fileName_CTR1 = 'homer-ampar-before-after-diff-trace-dist-CTR-1.xlsx'
#fileName_CTR2 = 'homer-ampar-before-after-diff-trace-dist-CTR-2.xlsx'
##fileName_CTR4 = 'homer-ampar-before-after-diff-trace-dist-CTR-4.xlsx'
#
#
#data_CTR1 = pd.read_excel(fileName_CTR1)
#data_CTR2 = pd.read_excel(fileName_CTR2)
#
##data_CTR4 = pd.read_excel(fileName_CTR4)
#
#distChange_CTR1 = data_CTR1['Distance_afe'] - data_CTR1['Distance_bef']
#distChange_CTR2 = data_CTR2['Distance_afe'] - data_CTR2['Distance_bef']
##distChange_CTR4 = data_CTR4['Distance_afe'] - data_CTR4['Distance_bef']
#
#distChange_CTR_nottx =  np.concatenate((distChange_CTR1, distChange_CTR2)) #, distChange_CTR4))

fileName_LTD6  = 'homer-nmdar-before-after-distance-LTD-6.xlsx'
fileName_LTD7  = 'homer-nmdar-before-after-distance-LTD-7.xlsx'


data_LTD6 = pd.read_excel(fileName_LTD6)
data_LTD7 = pd.read_excel(fileName_LTD7)
#data_LTD3 = pd.read_excel(fileName_LTD3)
#data_LTD4 = pd.read_excel(fileName_LTD4)
#data_LTD5 = pd.read_excel(fileName_LTD5)
#
#
distChange_LTD6 =  data_LTD6['Distance_afe'] - data_LTD6['Distance_bef']
distChange_LTD7 =  data_LTD7['Distance_afe'] - data_LTD7['Distance_bef']
#distChange_LTD3 =  data_LTD3['Distance_afe'] - data_LTD3['Distance_bef']
#distChange_LTD4 =  data_LTD4['Distance_afe'] - data_LTD4['Distance_bef']
#distChange_LTD5 =  data_LTD5['Distance_afe'] - data_LTD5['Distance_bef']
#
distChange_LTD = np.concatenate((distChange_LTD6, distChange_LTD7)) #, distChange_LTD3))
#distChange_LTD = distChange_LTD6

plt.figure()
sns.distplot(distChange_LTD, bins=mybins, kde=False , hist_kws=dict(edgecolor="k", linewidth=1, normed=True))
sns.distplot(distChange_CTR5, bins=mybins, kde=False, hist_kws=dict(edgecolor="k", linewidth=1, normed=True))


plt.xlim(xlim)
#plt.ylim(ylim)
#plt.plot(xline, yline)
#plt.plot(xlinep, yline, 'r')
plt.legend()
plt.xlabel('Relative Distance (nm)')
plt.ylabel('Counts')
plt.title('Relative Distance Between Cluster Centers Homer-NMDAR')
plt.xticks(np.arange(-500,600,100))


#diff_bef_ctr = np.concatenate((data_CTR5['Diff_Coeff_bef'], data_CTR5['Diff_Coeff_bef']))
#diff_afe_ctr = np.concatenate((data_CTR5['Diff_Coeff_afe'], data_CTR6['Diff_Coeff_afe']))
#
#bins = np.arange(-6,0,0.2)
#plt.figure()
#sns.distplot(np.log10(diff_bef_ctr), bins=bins, label='bef', kde=False , hist_kws=dict(edgecolor="k", linewidth=1, normed=True))
#sns.distplot(np.log10(diff_afe_ctr), bins=bins, label='aft', kde=False, hist_kws=dict(edgecolor="k", linewidth=1, normed=True))
#plt.xlabel('D ' + r'($\mu m^2/s, log)$')
#plt.ylabel('Counts')
#plt.title('Diffusion Coefficient of NMDAR before and after CTR')
#plt.legend()
#
#
#
#
#diff_bef = np.concatenate((data_LTD6['Diff_Coeff_bef'], data_LTD7['Diff_Coeff_bef']))
#diff_afe = np.concatenate((data_LTD6['Diff_Coeff_afe'], data_LTD7['Diff_Coeff_afe']))
#
#bins = np.arange(-6,0,0.2)
#plt.figure()
#sns.distplot(np.log10(diff_bef), bins=bins, label='bef', kde=False , hist_kws=dict(edgecolor="k", linewidth=1, normed=True))
#sns.distplot(np.log10(diff_afe), bins=bins, label='aft', kde=False, hist_kws=dict(edgecolor="k", linewidth=1, normed=True))
#plt.xlabel('D ' + r'($\mu m^2/s, log)$')
#plt.ylabel('Counts')
#plt.title('Diffusion Coefficient of NMDAR before and after LTD')
#plt.legend()
#
#
#trace_bef_ctr = np.concatenate((data_CTR5['Trace_Range_bef'], data_CTR6['Trace_Range_bef']))
#trace_afe_ctr = np.concatenate((data_CTR5['Trace_Range_afe'], data_CTR6['Trace_Range_afe']))
#
#bins = np.arange(0.0,3.0,0.05)
#plt.figure()
#sns.distplot((trace_bef_ctr/1000.0),bins=bins, label='bef', kde=False , hist_kws=dict(edgecolor="k", linewidth=1, ))
#sns.distplot((trace_afe_ctr/1000.0),bins=bins, label='aft', kde=False, hist_kws=dict(edgecolor="k", linewidth=1, ))
#plt.xlabel('Trace Range ' + r'($\mu m)$')
#plt.ylabel('Counts')
#plt.title('Trace Range NMDAR Before and After CTR')
#plt.legend()
#
#
#
#
#
#trace_bef = np.concatenate((data_LTD6['Trace_Range_bef'], data_LTD7['Trace_Range_bef']))
#trace_afe = np.concatenate((data_LTD6['Trace_Range_afe'], data_LTD7['Trace_Range_afe']))
#
#bins = np.arange(0.0,3.0,0.05)
#plt.figure()
#sns.distplot((trace_bef/1000.0),bins=bins, label='bef', kde=False , hist_kws=dict(edgecolor="k", linewidth=1, ))
#sns.distplot((trace_afe/1000.0),bins=bins, label='aft', kde=False, hist_kws=dict(edgecolor="k", linewidth=1, ))
#plt.xlabel('Trace Range ' + r'($\mu m)$')
#plt.ylabel('Counts')
#plt.title('Trace Range NMDAR Before and After LTD')
#plt.legend()

