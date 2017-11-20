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


#Load file
fileName_QD = 'ampar-after-diff-LTD-3-4.xlsx'
data_QD = pd.read_excel(fileName_QD)
fileSave_QD = fileName_QD.split('-diff-')
filseSave_QD_Sample = fileSave_QD[1].split('.')


fileName_QD1 = 'nmdar-after-diff-LTD-2-3.xlsx'
data_QD1 = pd.read_excel(fileName_QD1)
fileSave_QD1 = fileName_QD1.split('-after-diff-')

fig, ax = plt.subplots(figsize=(8, 6))
xlabel = 'Diff Coeff (um2/s, log)'
ylabel = 'Trajectory Range (um, log)'

sns.kdeplot(np.log10(data_QD['Diff Coeff']), np.log10(data_QD['Trace Range']/1000), cmap="gray", shade=True, ax=ax, n_levels=10,  shade_lowest=True )
sns.kdeplot(np.log10(data_QD1['Diff Coeff']), np.log10(data_QD1['Trace Range']/1000), cmap="Reds", ax=ax,  n_levels=10,  shade_lowest=False )
ax.set(xlabel=xlabel, ylabel=ylabel)



ax.set_xlim(-5.0,0)
ax.set_ylim(-1.15,0.50)

ax.plot([-6,0],[np.log10(0.79),np.log10(0.79)], 'w--')
ax.plot([np.log10(0.018),np.log10(0.018)],[-1.2,0.8],'w--' )
fig.savefig(fileSave_QD1[0]+'-heatmap-'+filseSave_QD_Sample[0])


g = sns.jointplot(np.log10(data_QD['Diff Coeff']), np.log10(data_QD['Trace Range']/1000), color='k', kind='kde', stat_func=None, size=6, xlim=(-6,0), ylim=(-1.15, 0.55))

g.ax_joint.plot([-5.89,-0.12],[np.log10(0.79),np.log10(0.79)], 'k--')
g.ax_joint.plot([np.log10(0.018),np.log10(0.018)],[-1.2,0.58],'k--' )
g.set_axis_labels(xlabel=xlabel, ylabel=ylabel)

plt.tight_layout()

g.savefig(fileSave_QD[0]+'-heatmap-'+filseSave_QD_Sample[0])