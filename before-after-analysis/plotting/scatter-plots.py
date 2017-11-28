# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 16:24:35 2017

@author: athomaz
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

import seaborn as sns
sns.set()

fileName_AMPAR_LTD3  = 'homer-ampar-before-after-diff-trace-dist-LTD-3.xlsx'
fileName_NMDAR_LTD3  = 'homer-nmdar-before-after-diff-trace-dist-LTD-3.xlsx'

AMPAR_LTD = pd.read_excel(fileName_AMPAR_LTD3)
NMDAR_LTD = pd.read_excel(fileName_NMDAR_LTD3)


distChange_AMPAR = AMPAR_LTD['Distance_afe'] - AMPAR_LTD['Distance_bef']
coeffChange_AMPAR = np.log10(AMPAR_LTD['Diff_Coeff_afe']) - np.log10(AMPAR_LTD['Diff_Coeff_bef'])

distChange_NMDAR = NMDAR_LTD['Distance_afe'] - NMDAR_LTD['Distance_bef']


sns.set(rc={'axes.facecolor':'#32353d'})
cmap = plt.cm.get_cmap('seismic', 4) 
normalize = mpl.colors.Normalize(vmin=-200, vmax=200)
plt.figure()
plt.scatter(AMPAR_LTD['Homer_x_afe'], AMPAR_LTD['Homer_y_afe'], c=distChange_AMPAR, s=25,cmap=cmap, norm=normalize )
plt.grid(False)
plt.colorbar()


cmap = plt.cm.get_cmap('seismic', 4) 
normalize = mpl.colors.Normalize(vmin=-200, vmax=200)
plt.figure()
plt.scatter(NMDAR_LTD['Homer_x_afe'], NMDAR_LTD['Homer_y_afe'], c=distChange_NMDAR, s=25,cmap=cmap, norm=normalize )
plt.grid(False)
plt.colorbar()

cmap = plt.cm.get_cmap('seismic',8) 
normalize = mpl.colors.Normalize(vmin=-2, vmax=2)
plt.figure()
plt.scatter(AMPAR_LTD['Homer_x_afe'], AMPAR_LTD['Homer_y_afe']+500, c=-1*coeffChange_AMPAR, s=25, cmap=cmap, norm=normalize )
#plt.grid(False)
plt.grid(False)
plt.colorbar()#ticks=[-5.75, -4.75, -3.75, -2.75, -1.75, -0.75, 0])
