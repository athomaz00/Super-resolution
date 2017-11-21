# =============================================================================
# ###########################################################################################################
# #This code calculates the Diffusion Coefficient and Trajectory Range of moving particles.
# #Input: position of particles after all corrections. Txt file in data_QD variable
# #Plots the histogram of Diffusion Coefficients and Trajectory Range
# #To calculate the Trajectory Range the convex_hull is calculated and the vertices with the highest distance 
# #between them are considered the trajectory range(distance between them)
# #Diffusion coefficient is calculated using mean square displacement
# # D = MSD/(2*d*dt) where d is the dimensionality (1,2,3)
# #MSD is the displacement in the position of the particles for various lag times. 
# #Code by Andre Thomaz 11/15/2017, adapted from Matlab version from Selvin Lab
# #############################################################################################################
# ############TO DO LIST#######################################################################################
# #Write a function py file for fMSD
# #Comment the steps
#
# =============================================================================


import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import lmfit as lm
import numpy as np
import pandas as pd
from scipy.integrate import trapz
from scipy.optimize import curve_fit
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
fileName_QD1 = 'nmdar-after-for_diffusion-LTD-3.txt'
#fileName_QD2 = 'nmdar-before-for_diffusion-CTR-2.txt'
#fileName_QD3 = 'nmdar-after-for_diffusion-LTD-3.txt'
data_QD1 = pd.read_csv(fileName_QD1, sep="\t");
#data_QD2 = pd.read_csv(fileName_QD2, sep="\t");
#data_QD3 = pd.read_csv(fileName_QD3, sep="\t");
data_QD = data_QD1
#data_QD = pd.concat([data_QD1, data_QD2, data_QD2])
data_QD = data_QD.rename(columns={'X (nm)': 'x', 'Y (nm)':'y', 'Z (nm)':'z', 'Frame Number':'frame'})
data_QD = data_QD[np.abs(data_QD['z']<600)]
data_QD['z'] = data_QD['z']*0.79


result_tracking = tp.link_df(data_QD, 1000.0, memory=10, pos_columns=['x', 'y', 'z'])

t1 = tp.filter_stubs(result_tracking,10)
#print(t1.particle.nunique())


#Construct filename output for Tracking positions
spt = fileName_QD1.split("for_diffusion")
fileNameTrack = spt[0] + 'tracking' + spt[1].split('.txt')[0]
for j in range(2,6):
    if ('fileName_QD' + str(j)) in vars(): # search through all vars for fileName_Qd
        fileNameTrack = fileNameTrack +'-' + list(filter(str.isdigit, vars()['fileName_QD' + str(j)]))[0] #add the number at fileName QD
        
    


writer = pd.ExcelWriter(fileNameTrack+'.xlsx')
t1.to_excel(writer, 'sheet1')
writer.save()

#t1[(t1['x']>59930) & (t1['x']<59940)];
#t1[t1['particle']==76] #FIRST PARTICLE IN MATLAB;

#t1 = pd.read_excel('result_tracking_py.xlsx')
#t1 = pd.read_excel('ampar-before-tracking-CTR-1-4.xlsx')

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
Dif_pd = pd.DataFrame(data=Dif, columns=['Coeff', 'Intercept', 'Particle', 'Diff Coeff' ])
Dif_pd['Trace Range'] = Trace_range[:]
dt = 0.05 #Time step between frames
Dif_pd = Dif_pd[Dif_pd['Coeff']>0.0]
Dif_pd['Diff Coeff'] = Dif_pd/(dt*2*3*1E6)







#save diff track
fileNameDiff = spt[0] + 'diff' + spt[1].split('.txt')[0]
for j in range(2,6):
    if ('fileName_QD' + str(j)) in vars(): # search through all vars for fileName_Qd
        fileNameDiff = fileNameDiff +'-' + list(filter(str.isdigit, vars()['fileName_QD' + str(j)]))[0] #add the number at fileName QD
        
writer = pd.ExcelWriter(fileNameDiff+'.xlsx')
Dif_pd.to_excel(writer, 'sheet1')
writer.save()









#x = np.copy(t1[t1['particle']== 76].x) # get the x positions of the particle
#y = np.copy(t1[t1['particle']==76].y)
#z = np.copy(t1[t1['particle']==76].z)
#TraceXYZ = np.column_stack((x, y, z))
#
#from scipy.spatial import Delaunay, ConvexHull
#tri = Delaunay(TraceXYZ)

#writer = pd.ExcelWriter('diff_coeff.xlsx')
#.to_excel(writer, 'sheet1')
##writer.save()