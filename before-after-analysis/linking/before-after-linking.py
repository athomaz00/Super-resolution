# -*- coding: utf-8 -*-
# =============================================================================
# This code links the Homer Cluster Center from the before file to the after file.
# If the cluster in the after file is less than 500 nm from the cluster in the before
# file the code links them.
# input: xlsx file from the msd-diffusion-trace-centers.py code for before at variable fileName_Before
# input: xlsx file from the msd-diffusion-trace-centers.py code for after at variable fileName_After
# output: ????
# Code written by Andre Thomaz in 11/27/2017
####TO Do#####
# 
# =============================================================================



import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import numpy as np
import pandas as pd

from scipy import spatial

plt.close('all')

#File names
fileName_Before = 'homer-nmdar-before-distance-LTD-7.xlsx'
fileName_After  = 'homer-nmdar-after-distance-LTD-7.xlsx'

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
pairPd = pd.DataFrame(pairTable, columns=['Homer_Number_bef', 'Homer_x_bef', 'Homer_y_bef', 'Homer_z_bef', 'Receptor_Number_bef', 'Receptor_x_bef', 'Receptor_y_bef', 'Receptor_z_bef', 'Distance_bef', 'Homer_Number_afe', 'Homer_x_afe', 'Homer_y_afe', 'Homer_z_afe', 'Receptor_Number_afe', 'Receptor_x_afe', 'Receptor_y_afe', 'Receptor_z_afe', 'Distance_afe'])
plt.figure()
plt.scatter(pairPd['Homer_x_bef'],pairPd['Homer_y_bef'], s=10)
plt.scatter(pairPd['Homer_x_afe'],pairPd['Homer_y_afe'], alpha=0.6, s=10)
   

fileName_Split = fileName_Before.split('-distance-')

fileName_Save = fileName_Split[0]+'-after'+'-distance-'+fileName_Split[1]

     
writer = pd.ExcelWriter(fileName_Save)
pairPd.to_excel(writer, 'sheet1')
writer.save()

print('=================================================================')
print(str(pairTable.shape[0]) + ' receptors/synapses after linking saved')
