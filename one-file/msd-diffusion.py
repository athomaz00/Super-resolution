# =============================================================================
# 
# This code calculates the Diffusion Coefficient and Trajectory Range of moving particles.
# Input: position of particles after all corrections. Txt file in data_QD variable
# Plots the histogram of Diffusion Coefficients and Trajectory Range
# To calculate the Trajectory Range the convex_hull is calculated and the vertices with the highest distance 
# between them are considered the trajectory range(distance between them)
# Diffusion coefficient is calculated using mean square displacement
#  D = MSD/(2*d*dt) where d is the dimensionality (1,2,3)
# MSD is the displacement in the position of the particles for various lag times. 
# Code by Andre Thomaz 11/15/2017, adapted from Matlab version from Selvin Lab
# 
# ############TO DO LIST#######################################################################################
# Write a function py file for fMSD
# Comment the steps
#
# =============================================================================


import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import numpy as np
import pandas as pd

from sklearn.cluster import DBSCAN, KMeans 

import scipy.spatial.distance as spd
import scipy.spatial as sps
import trackpy as tp

#########################################################MSD######################

def fMSD_vect(x,y,z, dpmax, dpmin,tSteps):
    x,y,z = x,y,z
    pNumb = dpmax-dpmin #number of positions
    msd0=np.zeros((1,pNumb))
    d2r=np.zeros((1,pNumb))
    counts=np.zeros((1,pNumb))
    Bc = np.arange(dpmin+1,dpmax+1)
    Ac = np.arange(dpmin,dpmax)
    if tSteps<=(pNumb):
        d2r=np.zeros((pNumb, tSteps))
        for stp in np.arange(0,tSteps):
            [B, A] = np.meshgrid(Bc,Ac)
            [a, b] = np.where((B - A)==stp+1)
            b += 1
            d2r[a,stp]=(x[b]-x[a])**2+(y[b]-y[a])**2+(z[b]-z[a])**2
            msd0[0,stp]=np.mean(d2r[np.nonzero(d2r[:,stp]),stp])
            counts[0,stp]=np.size(np.nonzero(d2r[:,stp]))
    return msd0[0], d2r[0], counts[0]
    

##############################################################################
fileName_QD1 = 'ampar-before-for_diffusion-CTR-1.txt'

data_QD1 = pd.read_csv(fileName_QD1, sep="\t");

data_QD = data_QD1

data_QD = data_QD.rename(columns={'X (nm)': 'x', 'Y (nm)':'y', 'Z (nm)':'z', 'Frame Number':'frame'})
data_QD = data_QD[np.abs(data_QD['z']<600)]
data_QD['z'] = data_QD['z']*0.79


result_tracking = tp.link_df(data_QD, 1000.0, memory=10, pos_columns=['x', 'y', 'z'])

t1 = tp.filter_stubs(result_tracking,10)



#Construct filename output for Tracking positions
spt = fileName_QD1.split("for_diffusion")
fileNameTrack = spt[0] + 'tracking' + spt[1].split('.txt')[0]
for j in range(2,6):
    if ('fileName_QD' + str(j)) in vars(): # search through all vars for fileName_Qd
        fileNameTrack = fileNameTrack +'-' + list(filter(str.isdigit, vars()['fileName_QD' + str(j)]))[0] #add the number at fileName QD
        
    


#writer = pd.ExcelWriter(fileNameTrack+'.xlsx')
#t1.to_excel(writer, 'sheet1')
#writer.save()


##loop through the particles to calculate msd for 10 steps
tSteps = 10;
nParticles = t1.particle.nunique()
Trace_range = np.zeros((nParticles,1))
Dif = np.zeros((nParticles,4))
for i, part in enumerate(t1.particle.unique()):
    dp = np.where(t1.particle==part) #range of rows of the particle in the results tracking
    dplength = len(dp)
    dpmax = np.max(dp)
    dpmin = np.min(dp)
    x = np.copy(t1[t1['particle']==part].x) # get the x positions of the particle
    y = np.copy(t1[t1['particle']==part].y)
    z = np.copy(t1[t1['particle']==part].z)
    frames = np.copy(t1[t1['particle']==part].frame)
    particle = np.copy(t1[t1['particle']==part].particle)
    # TraceAll add data frame for each particle
    TraceXYZ = np.column_stack((x, y, z))
    #Calculating trajectory range
    TRI_tr = sps.ConvexHull(TraceXYZ) #Make a polygon with the points
    Trace_range[i] = np.max(spd.pdist(TraceXYZ[TRI_tr.vertices]))
    msd, d2r, counts = fMSD_vect(x,y,z, dpmax, dpmin, tSteps)
    cutoff = 10.0#cutoff for data points (not sure exactly why)
    ind = np.where(counts>=cutoff)
    msdcut = msd[ind]
    indlength = np.size(ind)
    if indlength>=4:
        ind1=ind[0][0:4]
        msd1 = msdcut[0:4]
        d_temp = np.polyfit(ind1,msd1,1)
        Dif[i,0:2] = np.polyfit(ind1,msd1,1)
        Dif[i,2] = part
    else:
        Dif[i,0:2] = 0.0
        Dif[i,2] = part
        ind1 = ind
        msd1 = msd
    
    
    
    
    
    plt.plot(msd1)
    plt.xlim(0,3)
    


    
#Diffusion coefficient calculation
Dif_pd = pd.DataFrame(data=Dif, columns=['Coeff', 'Intercept', 'Particle', 'Diff-Coeff' ])
Dif_pd['Trace-Range'] = Trace_range[:]
dt = 0.05 #Time step between frames
Dif_pd = Dif_pd[Dif_pd['Coeff']>0.0]
Dif_pd['Diff-Coeff'] = Dif_pd['Coeff']/(dt*2*3*1E6)







#save diff track
fileNameDiff = spt[0] + 'diff-trace' + spt[1].split('.txt')[0]
for j in range(2,6):
    if ('fileName_QD' + str(j)) in vars(): # search through all vars for fileName_Qd
        fileNameDiff = fileNameDiff +'-' + list(filter(str.isdigit, vars()['fileName_QD' + str(j)]))[0] #add the number at fileName QD
        
#writer = pd.ExcelWriter(fileNameDiff+'.xlsx')
#Dif_pd.to_excel(writer, 'sheet1')
#writer.save()

###########################Cluster Center Calculation
#Load super resolution file and plot the 2D points
file1 = pd.read_excel('homer_before-CTR-1.xlsx', names=['x', 'y', 'z'])

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
    
#fileTracking = pd.read_excel('ampar-before-tracking-CTR-1.xlsx') #output iof tracking
#fileTracking = fileTracking.drop(fileTracking.columns[[0, 1, 2, 3, 4, 8, 9, 10, 11, 12, 13, 14, 15]], axis=1)  #drop columns

fileTracking = t1

#fileDiff = pd.read_excel('ampar-before-diff-trace-CTR-1.xlsx')
fileDiff = Dif_pd
fileDiff['QD_x'] = 0
fileDiff['QD_y'] = 0
fileDiff['QD_z'] = 0

nParticles = fileDiff.Particle.nunique()


receptor_centers = {}
for i, part in enumerate(fileDiff.Particle.unique()):
    sample = fileTracking[fileTracking['particle']==part]
    tt = sample.loc[:,'x':'z']
    kmeans2 = KMeans(n_clusters=1).fit(tt)
    rCenter = [kmeans2.cluster_centers_[0][0], kmeans2.cluster_centers_[0][1], kmeans2.cluster_centers_[0][2]]
    receptor_centers[part] = [tt, rCenter]
    index = fileDiff[fileDiff['Particle']==part].index
    fileDiff.loc[index[0], 'QD_x'] = rCenter[0]
    fileDiff.loc[index[0], 'QD_y'] = rCenter[1]
    fileDiff.loc[index[0], 'QD_z'] = rCenter[2]
    ax.scatter(tt.x, tt.y, s=5, c='g', alpha=0.2)
    ax.scatter(rCenter[0], rCenter[1], s=10, c='g')




