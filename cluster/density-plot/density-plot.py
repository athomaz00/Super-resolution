# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 13:13:56 2017

@author: athomaz
"""

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import numpy as np
import pandas as pd

from sklearn.cluster import DBSCAN, KMeans 
from scipy import spatial
import scipy.stats as st


###########################Cluster Center Calculation
#Load super resolution file and plot the 2D points
fileName_Homer = 'homer_before-CTR-4.xlsx'
file1 = pd.read_excel(fileName_Homer, names=['x', 'y', 'z'])



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
    ax.scatter(clusters[i].x, clusters[i].y, c='b', marker='o', s=5, alpha=0.2)
    
#clusters_centers is a dictionatry with keys as the number of calculated cluster and values the 
#cluster points and cluster center
centersHomer = []
for j in range(n_clusters_):
    kCluster = clusters[j]
    kmeans = KMeans(n_clusters=1).fit(kCluster)
    #kCenter = [kmeans.cluster_centers_[0][0], kmeans.cluster_centers_[0][1], kmeans.cluster_centers_[0][2]]
    centersHomer.append([j, kmeans.cluster_centers_[0][0], kmeans.cluster_centers_[0][1], kmeans.cluster_centers_[0][2] ])
    ax.scatter(kmeans.cluster_centers_[0][0], kmeans.cluster_centers_[0][1], s=10, c='g')
    
data_test = clusters[20] #.iloc[:, 0:2]
xmin, xmax = data_test['x'].min(), data_test['x'].max()
ymin, ymax = data_test['y'].min(), data_test['y'].max()
tree = spatial.KDTree(data_test.values)
xi = np.linspace(xmin, xmax)
yi = np.linspace(ymin, ymax)


#plt.contour(xi, yi, zi, cmap='jet')

distTable = []
for i, row in data_test.iterrows():
    homerPoints = [row['x'],row['y'] , row['z'] ]
    distance, ind = tree.query(homerPoints, k=2)
    distTable.append(distance[1])

distTable = np.array(distTable)
radius = 5*distTable.mean()

neighbors = tree.query_ball_tree(tree, radius)
frequency = np.array(list(map(len, neighbors)))
print(frequency)
# [2 3 2 1]
densi = frequency

#Z = np.reshape(frequency, X.shape)



plt.figure()
#sns.kdeplot(data_test['x'], data_test['y'], cmap='jet')
plt.scatter(data_test['x'], data_test['y'], c=frequency, s=20, cmap='jet')




kde = st.gaussian_kde([ data_test['x'],  data_test['y'],  data_test['z']])



density = kde([ data_test['x'],  data_test['y'],  data_test['z']])



import matplotlib.mlab as mlab
zi = mlab.griddata(data_test['x'].values, data_test['y'].values, density, xi, yi, interp='linear')
X, Y = np.mgrid[xmin:xmax:50j, ymin:ymax:50j]
#plt.contour(xi,yi,zi, cmap='jet')
d10 = 150


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data_test['x'], data_test['y'], data_test['z'], c=density, s=10)
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.contour(X, Y,zi)  #, data_test['z'], c=densi, s=10)
ax.set_xlim(data_test['x'].min(), data_test['x'].max())
ax.set_ylim(data_test['y'].min(), data_test['y'].max())
ax.set_zlim(data_test['z'].min(), data_test['z'].max())


data_test2 = data_test.copy()
data_test2['densi'] = densi
msk = data_test2['densi']>=d10
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data_test[msk]['x'], data_test[msk]['y'], data_test[msk]['z'], c='r', s=10)
ax.scatter(data_test[~msk]['x'], data_test[~msk]['y'], data_test[~msk]['z'], c='g', s=10)
ax.set_xlim(data_test['x'].min(), data_test['x'].max())
ax.set_ylim(data_test['y'].min(), data_test['y'].max())
ax.set_zlim(data_test['z'].min(), data_test['z'].max())




