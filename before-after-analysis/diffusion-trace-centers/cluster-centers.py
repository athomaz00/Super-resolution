# =============================================================================
# 
# This code calculates the Diffusion Coefficient, Trajectory Range of moving particles.
# Next step is to calculate the clusters centers both for Homer Clusters and Receptor Cluters.
# Then the code searches for receptor center that are less than 2000 um away from a given synapse 
# and assign that receptor to the specific Homer cluster. 
# Input: position of particles after all corrections. Txt file in data_QD variable
# Input: postition of Homer particles after all corrections. Xlsx file in file1 variable
# Output: xlsx file with Homer Cluster Number/Center Position and QD parameters
# To calculate the Trajectory Range the convex_hull is calculated and the vertices with the highest distance 
# between them are considered the trajectory range(distance between them)
# Diffusion coefficient is calculated using mean square displacement
#  D = MSD/(2*d*dt) where d is the dimensionality (1,2,3)
# MSD is the displacement in the position of the particles for various lag times. 
# Code by Andre Thomaz 11/27/2017, adapted from Matlab version from Selvin Lab
# 
# ############TO DO LIST#######################################################################################
# Write a function py file for fMSD
# Comment the steps
# Rewrite for clarity
# =============================================================================



import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import numpy as np
import pandas as pd

from sklearn.cluster import DBSCAN, KMeans 
from scipy import spatial


plt.close('all')


#File names
fileName_QD1 = 'nmdar-after-LTD-7.xlsx'
fileName_Homer = 'homer-after-LTD-7.xlsx'

##############################################################################





data_QD1 = pd.read_excel(fileName_QD1, names=['x', 'y', 'z'])
data_QD1 = data_QD1.dropna()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(data_QD1['x'], data_QD1['y'], c='k', marker='o', s=1)

#data_QD1 = pd.read_csv(fileName_QD1, sep="\t")

data_QD = data_QD1

#Get values to be calculated
data_QD_Points = data_QD.values

#call DBSCAN, eps =  Distance of Influence and Min_Samples number of neighbours
db_QD = DBSCAN(eps=500, min_samples=50).fit(data_QD_Points)
labels_QD = db_QD.labels_   #how many labes (clusters) detected, -1 noise
unique_labels_QD = set(labels_QD)
n_clusters_QD = len(set(labels_QD)) - (1 if -1 in labels_QD else 0)
clusters_QD = [data_QD_Points[labels_QD == j] for j in np.arange(n_clusters_QD)] #separate points according to clusters

centersQD = []
for j in range(n_clusters_QD):
    kCluster = clusters_QD[j]
    kmeans = KMeans(n_clusters=1).fit(kCluster)
    #kCenter = [kmeans.cluster_centers_[0][0], kmeans.cluster_centers_[0][1], kmeans.cluster_centers_[0][2]]
    centersQD.append([j, kmeans.cluster_centers_[0][0], kmeans.cluster_centers_[0][1], kmeans.cluster_centers_[0][2] ])
    ax.scatter(kmeans.cluster_centers_[0][0], kmeans.cluster_centers_[0][1], s=10, c='g')




###########################Cluster Center Calculation
#Load super resolution file and plot the 2D points
file1 = pd.read_excel(fileName_Homer, names=['x', 'y', 'z'])
file1 = file1.dropna()

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
    




distTable1 =[]
center_QD_table = []

centersQD = np.array(centersQD)
for i, hc in enumerate(centersHomer):
    homerPoints = [hc[1], hc[2], hc[3]]
    distance,ind = spatial.KDTree(centersQD[:,1:4]).query(homerPoints)
    if distance <=400:
        tempTable = [i, hc[1], hc[2], hc[3], centersQD[ind,0], centersQD[ind,1], centersQD[ind,2], centersQD[ind,3], distance]
        distTable1.append(tempTable)
    

           
           
           
           
      
    

    
finalPd = pd.DataFrame(data=distTable1, columns=['Homer_Number', 'Homer_x', 'Homer_y', 'Homer_z', 'Receptor_Number', 'Receptor_x', 'Receptor_y', 'Receptor_z', 'distance'])

fileSave = fileName_Homer.split('homer')
if 'before' in fileName_QD1:
    split_QD = '-before-'
else:
    split_QD = '-after-'
fileSave_QD = fileName_QD1.split(split_QD)
fileSave = 'homer-'+fileSave_QD[0] + split_QD + 'distance-'+fileSave_QD[1]


writer = pd.ExcelWriter(fileSave)
finalPd.to_excel(writer, 'sheet1')
writer.save()
print(str(finalPd.shape[0]) + ' receptors/synapses saved')


