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



fileHomer1 = np.load('Homer-before-clusters-centers-CTR-1.npy').flat[0]
fileQD1 = np.load('Ampar-before-particles-centers-CTR-1.npy').flat[0]


fileHomer2 = np.load('Homer-after-clusters-centers-CTR-1.npy').flat[0]
fileQD2 = np.load('Ampar-after-particles-centers-CTR-1.npy').flat[0]



#fig = plt.figure()
#ax = fig.add_subplot()

homerClusters1 = []
for i in fileHomer1.keys():
    homerClusters1.append(fileHomer1[i][1])
    
receptorClusters1 = []
for i in fileQD1.keys():
    receptorClusters1.append(fileQD1[i][1])
    
#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.scatter(hc1[:][:,0],hc1[:][:,1], s=7.5 )
#ax.scatter(ac1[:][:,0],ac1[:][:,1],s=7.5, alpha=0.3)


distTable1 =[]
for i, hc in enumerate(homerClusters1):
    distance,index = spatial.KDTree(receptorClusters1).query(hc)
    
    for part, center in fileQD1.items():
        if receptorClusters1[index] == center[1] and distance<=2000.0:
            tempTable = [i, hc[0], hc[1], hc[2], part, receptorClusters1[index][0],receptorClusters1[index][1],receptorClusters1[index][2],  distance]
            distTable1.append(tempTable)

distPdframe_before = pd.DataFrame(distTable1, columns=['Homer Number', 'Homer Center x', 'Homer Center y', 'Homer Center z',  'QD Number', 'QD Center x', 'QD Center y', 'QD Center z', 'Distance'])

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(distPdframe_before['Homer Center x'],distPdframe_before['Homer Center y'], s=7.5 )
ax.scatter(distPdframe_before['QD Center x'],distPdframe_before['QD Center y'],s=7.5)


########################################################################################################





homerClusters2 = []
for i in fileHomer2.keys():
    homerClusters2.append(fileHomer2[i][1])
    
receptorClusters2 = []
for i in fileQD2.keys():
    receptorClusters2.append(fileQD2[i][1])

distTable2 =[]
for i, hc in enumerate(homerClusters2):
    distance,index = spatial.KDTree(receptorClusters2).query(hc)
    
    for part, center in fileQD2.items():
        if receptorClusters2[index] == center[1] and distance<=2000.0:
            tempTable2 = [i, hc[0], hc[1], hc[2], part, receptorClusters2[index][0],receptorClusters2[index][1],receptorClusters2[index][2],  distance]
            distTable2.append(tempTable2)

distPdframe_after = pd.DataFrame(distTable2, columns=['Homer Number', 'Homer Center x', 'Homer Center y', 'Homer Center z',  'QD Number', 'QD Center x', 'QD Center y', 'QD Center z', 'Distance'])

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(distPdframe_after['Homer Center x'],distPdframe_after['Homer Center y'], s=7.5 )
ax.scatter(distPdframe_after['QD Center x'],distPdframe_after['QD Center y'],s=7.5)


#Keep Homer that appear only before and after
homerBfAf = []
hafter = distPdframe_after.loc[:,'Homer Center x':'Homer Center z'].values.tolist()
for ind, row in distPdframe_before.iterrows():
    hbefore = row['Homer Center x':'Homer Center z']
    distance, index = spatial.KDTree(hafter).query(hbefore)
    if distance<=500.00:
        tempTable = [row['Homer Number'], distPdframe_after.iloc[index]['Homer Number']]
        homerBfAf.append(tempTable)
        
homerBfAf = np.array(homerBfAf)
np.save('homer-ampar-BfAf-CTR-1.npy', homerBfAf)

hb_mask = distPdframe_before['Homer Number'].isin(homerBfAf[:][:,0])
hf_mask = distPdframe_after['Homer Number'].isin(homerBfAf[:][:,1])
            

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(distPdframe_before[hb_mask].loc[:,'Homer Center x'],distPdframe_before[hb_mask].loc[:,'Homer Center y'], c='b', s=5.5 )
ax.scatter(distPdframe_after[hf_mask].loc[:,'Homer Center x'],distPdframe_after[hf_mask].loc[:,'Homer Center y'],c='r', s=10.5, alpha=0.3)      
#ax.scatter(distPdframe_after[~hf_mask].loc[:,'Homer Center x'],distPdframe_after[~hf_mask].loc[:,'Homer Center y'],c='g', s=7.5, alpha=0.2) 



writer = pd.ExcelWriter('dist-homer-ampar-before-CTR-1.xlsx')
distPdframe_before[hb_mask].to_excel(writer, 'sheet1')
writer.save()

writer = pd.ExcelWriter('dist-homer-ampar-after-CTR-1.xlsx')
distPdframe_after[hf_mask].to_excel(writer, 'sheet1')
writer.save()

print('homerBfAf = ', homerBfAf.shape)
print('Before = ', distPdframe_before[hb_mask].shape)
print('After = ', distPdframe_after[hf_mask].shape)