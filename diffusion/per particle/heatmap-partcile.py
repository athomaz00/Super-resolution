# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 15:00:10 2017

@author: Andre Thomaz
"""
##############################################################################
#This code calculates the heatmap (contour) of Diffusion Coeffecient and Trace Range
#of tracked particles.
#Input: diff xlsx file outputed by msd-diffsion.py
#Output heatmap for input file
##############################################################################

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import numpy as np
import pandas as pd
from scipy.integrate import trapz
from scipy.optimize import curve_fit


######Before
fileName_before_QD1 = 'nmdar-before-diff-traj-LTD-1.xlsx'
data_before_QD1 = pd.read_excel(fileName_before_QD1)

fileName_before_QD2 = 'nmdar-before-diff-traj-CTR-2.xlsx'
data_before_QD2 = pd.read_excel(fileName_before_QD2)

fileName_before_QD3 = 'nmdar-before-diff-traj-LTD-3.xlsx'
data_before_QD3 = pd.read_excel(fileName_before_QD3)

fileName_before_QD4 = 'ampar-before-diff-traj-LTD-5.xlsx'
data_before_QD4 = pd.read_excel(fileName_before_QD4 )

Dif_pd_before = np.concatenate((data_before_QD1['Diff Coeff'].values, data_before_QD3['Diff Coeff'].values)) # data_before_QD4['Diff Coeff'].values))
Trace_before = np.concatenate((data_before_QD1['Trace Range'].values, data_before_QD3['Trace Range'].values)) #, data_before_QD4['Trace Range'].values))


############After
fileName_after_QD1 = 'nmdar-after-diff-traj-CTR-1.xlsx'
data_after_QD1 = pd.read_excel(fileName_after_QD1 )

fileName_after_QD2 = 'nmdar-after-diff-traj-CTR-2.xlsx'
data_after_QD2 = pd.read_excel(fileName_after_QD2)

fileName_after_QD3 = 'nmdar-after-diff-traj-LTD-3.xlsx'
data_after_QD3 = pd.read_excel(fileName_after_QD3)

fileName_after_QD4 = 'ampar-after-diff-traj-LTD-5.xlsx'
data_after_QD4 = pd.read_excel(fileName_after_QD4)

Dif_pd_after = np.concatenate((data_after_QD1['Diff Coeff'].values, data_after_QD3['Diff Coeff'].values)) #, data_after_QD4['Diff Coeff'].values))
Trace_after = np.concatenate((data_after_QD1['Trace Range'].values, data_after_QD3['Trace Range'].values)) #, data_after_QD4['Trace Range'].values))



plt.figure()
sns.distplot(np.log10(Dif_pd_before ), label='LTD ', bins=25, kde=False, hist_kws=dict(edgecolor="k", linewidth=1,  normed=True))

sns.distplot(np.log10(Dif_pd_after), label='LTD ', bins=25, kde=False, hist_kws=dict(edgecolor="k", linewidth=1,  normed=True))








#
#
#
fig, ax = plt.subplots(figsize=(8, 6))
xlabel = 'Diff Coeff (um2/s, log)'
ylabel = 'Trajectory Range (um, log)'

sns.kdeplot(np.log10(Dif_pd_before), np.log10(Trace_before/1000), cmap="gray", shade=True, ax=ax, n_levels=10,  shade_lowest=True )
sns.kdeplot(np.log10(Dif_pd_after), np.log10(Trace_after/1000), cmap="Reds", ax=ax,  n_levels=10,  shade_lowest=False )
ax.set(xlabel=xlabel, ylabel=ylabel)



ax.set_xlim(-5.0,0)
ax.set_ylim(-1.15,0.50)

ax.plot([-6,0],[np.log10(0.79),np.log10(0.79)], 'w--')
ax.plot([np.log10(0.018),np.log10(0.018)],[-1.2,0.8],'w--' )
##fig.savefig(fileSave_QD1[0]+'-heatmap-'+filseSave_QD_Sample[0])
#
#
#g = sns.jointplot(np.log10(data_QD['Diff Coeff']), np.log10(data_QD['Trace Range']/1000), color='k', kind='kde', stat_func=None, size=6, xlim=(-6,0), ylim=(-1.15, 0.55))
#
#g.ax_joint.plot([-5.89,-0.12],[np.log10(0.79),np.log10(0.79)], 'k--')
#g.ax_joint.plot([np.log10(0.018),np.log10(0.018)],[-1.2,0.58],'k--' )
#g.set_axis_labels(xlabel=xlabel, ylabel=ylabel)
#
#plt.tight_layout()
#
##g.savefig(fileSave_QD[0]+'-heatmap-'+filseSave_QD_Sample[0])