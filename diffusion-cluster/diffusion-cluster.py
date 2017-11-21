# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 17:57:43 2017

@author: Andre Thomaz

"""
# =============================================================================
# This code calculates cluster based positions points detected using DBSCAN 
# and K-means methods to calculate centers, both from sklearn
#input = super resolution file after all corrections
#output = matrix for each cluster detected and its center
#################To Do################################
#1.Check if it is better to save as dictionary os dataframe
# =============================================================================

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans 
from scipy import spatial



#Load super resolution file and plot the 2D points
file1 = pd.read_excel('homer_before.xlsx', names=['x', 'y', 'z'])

#file1 = file1.drop(file1.columns[[0, 1, 2, 3, 4, 8, 9, 10, 11, 12, 13, 14, 15]], axis=1) 
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(file1['x'], file1['y'], c='k', marker='o', s=1)




#Get values to be calculated
dataPoints = file1.values

#call DBSCAN, eps =  Distance of Influence and Min_Samples number of neighbours
db = DBSCAN(eps=500, min_samples=50).fit(dataPoints)
labels = db.labels_   #how many labes (clusters) detected, -1 noise
unique_labels = set(labels)
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
clusters = [file1[labels == j] for j in np.arange(n_clusters_)] #separate points according to clusters

for i in range(n_clusters_):
    ax.scatter(clusters[i].x, clusters[i].y, c='r', marker='o', s=5, alpha=0.2)
    
#clusters_centers is a dictionatry with keys as the number of calculated cluster and values the 
#cluster points and cluster center
clusters_centers = {} 
for j in range(n_clusters_):
    kCluster = clusters[j]
    kmeans = KMeans(n_clusters=1).fit(kCluster)
    kCenter = [kmeans.cluster_centers_[0][0], kmeans.cluster_centers_[0][1], kmeans.cluster_centers_[0][2]]
    clusters_centers[j] = [kCluster, kCenter ]
    ax.scatter(kCenter[0], kCenter[1], s=10, c='b')

np.save("Homer-before-clusters-centers.npy", clusters_centers)

#Create a dictionary for the QD data, key particle and values x,y,z position and center 

file2 = pd.read_excel('ampar-before-tracking-CTR-1.xlsx')

file2 = file2.drop(file2.columns[[0, 1, 2, 3, 4, 8, 9, 10, 11, 12, 13, 14, 15]], axis=1) 

nParticles = file2.particle.nunique()


receptor_centers = {}
for i, part in enumerate(file2.particle.unique()):
    sample = file2[file2['particle']==part]
    tt = sample.loc[:,'x':'z']
    kmeans2 = KMeans(n_clusters=1).fit(tt)
    rCenter = [kmeans2.cluster_centers_[0][0], kmeans2.cluster_centers_[0][1], kmeans2.cluster_centers_[0][2]]
    receptor_centers[part] = [tt, rCenter]
    #ax.scatter(tt.x, tt.y, s=5, c='g', alpha=0.2)
    #ax.scatter(rCenter[0], rCenter[1], s=10, c='g')
    
np.save("Ampar-before-particles-centers.npy", receptor_centers)



#file3 = pd.read_excel('ampar-before-tracking-CTR-1.xlsx')
#file3 = file3.drop(file3.columns[[0, 1, 2, 3, 4, 8, 9, 10, 11, 12, 13, 14, 15]], axis=1) 
#
#for k 
#sample = file3[file3['particle']==0]

#kmeans = KMeans(n_clusters=1, random_state=0).fit(sample)
#
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(sample['x'], sample['y'], sample['z'], c='r', marker='o')
#ax.scatter(kmeans.cluster_centers_[0][0], kmeans.cluster_centers_[0][1], kmeans.cluster_centers_[0][2], s=100, c='b')