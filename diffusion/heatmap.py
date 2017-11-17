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
fileName_QD = 'nmdar-before-diff-LTD-2-3.xlsx'
data_QD = pd.read_excel(fileName_QD)

fileName_QD1 = 'ampar-before-diff-LTD-3-4.xlsx'
data_QD1 = pd.read_excel(fileName_QD1)

fig, ax = plt.subplots(figsize=(8, 6))
with sns.axes_style('white'):

    sns.kdeplot(np.log10(data_QD['Diff Coeff']), np.log10(data_QD['Trace Range']/1000), cmap="binary", shade=True, ax=ax, n_levels=10,  shade_lowest=False )
    sns.kdeplot(np.log10(data_QD1['Diff Coeff']), np.log10(data_QD1['Trace Range']/1000), cmap="Blues", ax=ax,  n_levels=10,  shade_lowest=False )


ax.set_xlim(-6,0)
ax.set_ylim(-1.2,0.8)

ax.plot([-6,0],[np.log10(0.79),np.log10(0.79)], 'k--')
ax.plot([np.log10(0.018),np.log10(0.018)],[-1.2,0.8],'k--' )



g = sns.jointplot(np.log10(data_QD['Diff Coeff']), np.log10(data_QD['Trace Range']/1000), kind='kde', stat_func=None, size=7)

g.ax_joint.plot([-6.1,0.2],[np.log10(0.79),np.log10(0.79)], 'k--')
g.ax_joint.plot([np.log10(0.018),np.log10(0.018)],[-1.10,0.60],'k--' )

