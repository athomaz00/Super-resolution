# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 00:01:16 2017

@author: athomaz
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

import seaborn as sns






sns.set(rc={'axes.facecolor':'#32353d'})



mybins  = np.arange(-500,500,50)
xlim = (0,80000)
ylim = (0,80000)

HAb = pd.read_excel('dist-homer-ampar-before-LTD-3.xlsx')
HAa = pd.read_excel('dist-homer-ampar-after-LTD-3.xlsx')
centerBA = np.load('homer-ampar-BfAf-LTD-3.npy')
Diff_APb = pd.read_excel('ampar-before-diff-traj-LTD-3.xlsx')
Diff_APa = pd.read_excel('ampar-after-diff-traj-LTD-3.xlsx')

dist_AP_LTD1 = []
for part1, part2 in centerBA:
     temp = HAa[HAa['Homer Number']==part2].Distance.values - HAb[HAb['Homer Number']==part1].Distance.values
     dist_AP_LTD1.append(temp[0])
dist_AP_LTD1 = np.array(dist_AP_LTD1)



dist_500_AP = np.abs(dist_AP_LTD1)<500



#plt.figure()
#sns.distplot(HAa['Distance'], bins=mybins, kde=False, hist_kws=dict(edgecolor="k", linewidth=1,  normed=True))
#plt.grid(False)

#cmap = plt.cm.get_cmap('Greens', 8) 
#
#
#plt.figure(figsize=(12,8))
#plt.scatter(HAb['Homer Center x'], HAb['Homer Center y'] , c='g', s=5)
#plt.scatter(HAb['QD Center x'],HAb['QD Center y'], c=HAb['Distance'], s=25, cmap=cmap )
#plt.colorbar()
#plt.grid(False)

cmap = plt.cm.get_cmap('seismic', 64) 
normalize = mpl.colors.Normalize(vmin=-200, vmax=200)
plt.figure(figsize=(12,8))
plt.scatter(HAb['Homer Center x'], HAb['Homer Center y'] , c='r', s=5)
plt.scatter(HAb[dist_500_AP]['QD Center x'],HAb[dist_500_AP]['QD Center y'], c=dist_AP_LTD1[dist_500_AP], s=25, cmap=cmap, norm=normalize )
plt.colorbar()
plt.grid(False)
plt.xlim(xlim)
plt.ylim(ylim)
#circle = plt.Circle((80000, 40000), radius=50000, color='w', fill=False)
#ax = plt.gca()

#ax.add_artist(circle)

HNLb = pd.read_excel('dist-homer-nmdar-before-LTD-3.xlsx')
HNLa = pd.read_excel('dist-homer-nmdar-after-LTD-3.xlsx')
centerBA_LTD = np.load('homer-nmdar-BfAf-LTD-3.npy')

dist_HN_LTD1 = []
for part1, part2 in centerBA_LTD:
     temp = HNLa[HNLa['Homer Number']==part2].Distance.values - HNLb[HNLb['Homer Number']==part1].Distance.values
     dist_HN_LTD1.append(temp[0])
dist_HN_LTD1 = np.array(dist_HN_LTD1)





dist_500_NM = np.abs(dist_HN_LTD1)<500

cmap = plt.cm.get_cmap('viridis', 8) 
plt.figure(figsize=(12,8))
plt.scatter(HNLb['Homer Center x'],HNLb['Homer Center y'] , c='r', s=5)
plt.scatter(HNLb[dist_500_NM]['QD Center x'],HNLb[dist_500_NM]['QD Center y'], c=dist_HN_LTD1[dist_500_NM], s=25, cmap=cmap, norm=normalize)
plt.colorbar()
plt.grid(False)
plt.xlim(xlim)
plt.ylim(ylim)
#circle = plt.Circle((80000, 40000), radius=50000, color='w', fill=False)
#ax = plt.gca()
#
#ax.add_artist(circle)


new_table = []

for part in Diff_APb['Particle'].tolist():
    if part in HAb['QD Number'].tolist():

        temp = [part, HAb[HAb['QD Number']==part].Distance.values[0], Diff_APb[Diff_APb['Particle']==part]['Diff Coeff'].values[0],  Diff_APb[Diff_APb['Particle']==part]['Trace Range'].values[0]]
        new_table.append(temp)
        
newPDb = pd.DataFrame(new_table, columns=['Particle', 'DistanceH', 'Diff_Coeff', 'Trace'])

new_table = []

for part in Diff_APa['Particle'].tolist():
    if part in HAa['QD Number'].tolist():

        temp = [part, HAa[HAa['QD Number']==part].Distance.values[0], Diff_APa[Diff_APa['Particle']==part]['Diff Coeff'].values[0],  Diff_APa[Diff_APa['Particle']==part]['Trace Range'].values[0]]
        new_table.append(temp)
        
newPDa = pd.DataFrame(new_table, columns=['Particle', 'DistanceH', 'Diff_Coeff', 'Trace'])

plt.figure()

for index, row in newPDb.iterrows():
    temp = HAb[HAb['QD Number']==row.Particle]['QD Center x']
    newPDb.loc[index, 'x'] = temp.values[0]
    temp = HAb[HAb['QD Number']==row.Particle]['QD Center y']
    newPDb.loc[index, 'y'] = temp.values[0]
    temp = HAb[HAb['QD Number']==row.Particle]['QD Center z']
    newPDb.loc[index, 'z'] = temp.values[0]

for index, row in newPDa.iterrows():
    temp = HAa[HAa['QD Number']==row.Particle]['QD Center x']
    newPDa.loc[index, 'x'] = temp.values[0]
    temp = HAa[HAa['QD Number']==row.Particle]['QD Center y']
    newPDa.loc[index, 'y'] = temp.values[0]
    temp = HAa[HAa['QD Number']==row.Particle]['QD Center z']
    newPDa.loc[index, 'z'] = temp.values[0]


