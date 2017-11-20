# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 23:33:26 2017

@author: athomaz
"""


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

cc =[]
for i, hc in enumerate(homerClusters):
    distance,index = spatial.KDTree(receptorClusters).query(hc)
    
    for part, center in fileQD.items():
        if receptorClusters[index] == center[1] and distance<=500.0:
            dd = [part, receptorClusters[index], i]
            cc.append(dd)