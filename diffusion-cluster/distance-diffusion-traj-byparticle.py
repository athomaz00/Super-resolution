# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 23:33:26 2017

@author: athomaz
"""
# =============================================================================
# This code calculates the distance from the Homer cluster to the first QD traj
#ectory center calculated by diffusion-cluster.py
#input: dictionary output from diffusion-cluster.py
#output: spreadsheet with 'Homer Number', 'Homer Center', 'QD Number', 'QD Center', 'Distance'
# =============================================================================

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import numpy as np
import pandas as pd 
from scipy import spatial



fileHomer = np.load('Homer-before-clusters-centers.npy').flat[0]
fileQD = np.load('Ampar-before-particles-centers.npy').flat[0]

#fig = plt.figure()
#ax = fig.add_subplot()

homerClusters = []
for i in fileHomer.keys():
    homerClusters.append(fileHomer[i][1])
    
receptorClusters = []
for i in fileQD.keys():
    receptorClusters.append(fileQD[i][1])

distTable =[]
for i, hc in enumerate(homerClusters):
    distance,index = spatial.KDTree(receptorClusters).query(hc)
    
    for part, center in fileQD.items():
        if receptorClusters[index] == center[1] and distance<=500.0:
            tempTable = [i, hc[0], hc[1], hc[2], part, receptorClusters[index][0],receptorClusters[index][1],receptorClusters[index][2],  distance]
            distTable.append(tempTable)

distPdframe = pd.DataFrame(distTable, columns=['Homer Number', 'Homer Center x', 'Homer Center y', 'Homer Center z',  'QD Number', 'QD Center x', 'QD Center y', 'QD Center z', 'Distance'])

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(distPdframe['Homer Center x'],distPdframe['Homer Center y'], s=7.5 )
ax.scatter(distPdframe['QD Center x'],distPdframe['QD Center y'],s=7.5)


########################################################################################################


fileHomer = np.load('Homer-before-clusters-centers.npy').flat[0]
fileQD = np.load('Ampar-before-particles-centers.npy').flat[0]



homerClusters = []
for i in fileHomer.keys():
    homerClusters.append(fileHomer[i][1])
    
receptorClusters = []
for i in fileQD.keys():
    receptorClusters.append(fileQD[i][1])

distTable =[]
for i, hc in enumerate(homerClusters):
    distance,index = spatial.KDTree(receptorClusters).query(hc)
    
    for part, center in fileQD.items():
        if receptorClusters[index] == center[1] and distance<=500.0:
            tempTable = [i, hc[0], hc[1], hc[2], part, receptorClusters[index][0],receptorClusters[index][1],receptorClusters[index][2],  distance]
            distTable.append(tempTable)

distPdframe = pd.DataFrame(distTable, columns=['Homer Number', 'Homer Center x', 'Homer Center y', 'Homer Center z',  'QD Number', 'QD Center x', 'QD Center y', 'QD Center z', 'Distance'])

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(distPdframe['Homer Center x'],distPdframe['Homer Center y'], s=7.5 )
ax.scatter(distPdframe['QD Center x'],distPdframe['QD Center y'],s=7.5)
