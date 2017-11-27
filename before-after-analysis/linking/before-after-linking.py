# -*- coding: utf-8 -*-
# =============================================================================
# This code links the Homer Cluster Center from the before file to the after file.
# If the cluster in the after file is less than 500 um from the cluster in the before
# file the code links them.
# input: xlsx file from the msd-diffusion-trace-centers.py code for before at variable fileName_Before
# input: xlsx file from the msd-diffusion-trace-centers.py code for after at variable fileName_After
# output: ????
# Code written by Andre Thomaz in 11/27/2017
####TO Do#####
# 
# =============================================================================

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd

from scipy import spatial



#File names
fileName_Before = 'homer-ampar-before-diff-trace-dist-LTD-3.xlsx'
fileName_After  = 'homer-ampar-after-diff-trace-dist-LTD-3.xlsx'

data_Before = pd.read_excel(fileName_Before)
data_After = pd.read_excel(fileName_After)

plt.figure()
plt.scatter(data_Before['Homer_x'], data_Before['Homer_y'])
plt.scatter(data_After['Homer_x'], data_After['Homer_y'])
#homerPoints_Before = data_Before.loc[:,'Homer_x':'Homer_z']
homerPoints_After = data_After.loc[:,'Homer_x':'Homer_z']


pairTable = []
for i, row in data_Before.iterrows():
    homerPoints_Before = [row['Homer_x'], row['Homer_y'], row['Homer_z']]
    distance,ind = spatial.KDTree(homerPoints_After.values).query(homerPoints_Before)
    if distance <=500:
        if (data_After.iloc[ind]['Homer_x']==homerPoints_After.iloc[ind]['Homer_x']) & (data_After.iloc[ind]['Homer_y']==homerPoints_After.iloc[ind]['Homer_y']) & (data_After.iloc[ind]['Homer_z']==homerPoints_After.iloc[ind]['Homer_z']):
            tempTable = np.concatenate((row.values.tolist(), data_After.iloc[ind].values.tolist()))
            pairTable.append(tempTable)
    
pairTable = np.array(pairTable)
pairPd = pd.DataFrame(pairTable, columns=['Homer_Number_bef', 'Homer_x_bef', 'Homer_y_bef', 'Homer_z_bef', 'Receptor_Number_bef', 'Receptor_x_bef', 'Receptor_y_bef', 'Receptor_z_bef', 'Diff_Coeff_bef', 'Trace_Range_bef', 'Distance_bef', 'Homer_Number_afe', 'Homer_x_afe', 'Homer_y_afe', 'Homer_z_afe', 'Receptor_Number_afe', 'Receptor_x_afe', 'Receptor_y_afe', 'Receptor_z_afe', 'Diff_Coeff_afe', 'Trace_Range_afe', 'Distance_afe'])
plt.figure()
plt.scatter(pairPd['Homer_x_bef'],pairPd['Homer_y_bef'], s=10)
plt.scatter(pairPd['Homer_x_afe'],pairPd['Homer_y_afe'], alpha=0.6, s=10)
        
dist_Change = pairPd['Distance_afe'] - pairPd['Distance_bef']


sns.set(rc={'axes.facecolor':'#32353d'})
cmap = plt.cm.get_cmap('seismic', 4) 
normalize = mpl.colors.Normalize(vmin=-200, vmax=200)
plt.figure()
plt.scatter(pairPd['Homer_x_afe'],pairPd['Homer_y_afe'], alpha=0.6, c=dist_Change, s=25, cmap=cmap, norm=normalize )
plt.colorbar()
plt.grid(False)


sns.set()
mybins  = np.arange(-500,500,50)
xlim = (-500,500)
ylim = (0,0.0045)

plt.figure()
sns.distplot(dist_Change, bins=mybins, kde=False, hist_kws=dict(edgecolor="k", linewidth=1,  normed=True))
plt.xlim(xlim)
plt.ylim(ylim)


coeff_Change = (pairPd['Diff_Coeff_afe']) - (pairPd['Diff_Coeff_bef'])

sns.set(rc={'axes.facecolor':'#32353d'})
cmap = plt.cm.get_cmap('seismic', 6) 
normalize = mpl.colors.Normalize(vmin=-0.15, vmax=0.15)
plt.figure()
plt.scatter(pairPd['Homer_x_afe'],pairPd['Homer_y_afe'], alpha=0.6, c=coeff_Change, s=25, cmap=cmap, norm=normalize)
plt.colorbar()
plt.grid(False)

trace_Change = (pairPd['Trace_Range_afe']) - (pairPd['Trace_Range_bef'])

sns.set(rc={'axes.facecolor':'#32353d'})
cmap = plt.cm.get_cmap('seismic',4) 
normalize = mpl.colors.Normalize(vmin=-1000, vmax=1000)
plt.figure()
plt.scatter(pairPd['Homer_x_afe'],pairPd['Homer_y_afe'], alpha=0.6, c=trace_Change, s=25, cmap=cmap, norm=normalize)
plt.colorbar()
plt.grid(False)


