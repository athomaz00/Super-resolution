###########################################################################################################
#This code calculates the Diffusion Coefficient and Trajectory Range of moving particles.
#Input: position of particles after all corrections. Txt file in data_QD variable
#Plots the histogram of Diffusion Coefficients and Trajectory Range
#To calculate the Trajectory Range the convex_hull is calculated and the vertices with the highest distance 
#between them are considered the trajectory range(distance between them)
#Diffusion coefficient is calculated using mean square displacement
# D = MSD/(2*d*dt) where d is the dimensionality (1,2,3)
#MSD is the displacement in the position of the particles for various lag times. 
#Code by Andre Thomaz 11/15/2017, adapted from Matlab version from Selvin Lab
#############################################################################################################
############TO DO LIST#######################################################################################
#Write a function py file for fMSD
#Comment the steps


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
    
def gauss(x,mu,sigma,A):
    return A*np.exp(-(x-mu)**2/2/sigma**2)

def bimodal(x,mu1,sigma1,A1,mu2,sigma2,A2):
    return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)

def integ(bins, params1, params2):
    A1 = trapz(gauss(bins,*params1), bins)
    A2 = trapz(gauss(bins,*params2), bins)
    AT = A1+A2
    A1 = A1*100/AT
    print('Percentage of 1 int', A1)
    A2 = A2*100/AT
    print('Percentage of 2 int', A2)
    return A1, A2
##############################################################################
fileName_QD1 = 'nmdar-before-for_diffusion-CTR-1.txt'
fileName_QD2 = 'nmdar-before-for_diffusion-CTR-2.txt'
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


######Load saved diffusion data###############
#Dif_pd = pd.read_excel('ampar-after-diff-LTD-3-4.xlsx')



####Plotting Dif

plt.figure()
sns.distplot(np.log10(Dif_pd['Diff Coeff']), bins=30,  label='Dif', kde=False ,hist_kws=dict(edgecolor="k", linewidth=1))

    
#####fitting Dif
y_d, bins_d = np.histogram(np.log10(Dif_pd['Diff Coeff']), bins=30)
bins_d=bins_d[:-1]
bins_plot_d = np.linspace(-6,0,500)
#bins_d=(bins_d[1:]+bins_d[:-1])/2
#
###two peaks at once
#expected_d = [-3.02, 0.2, 100, -0.21629379, 0.15312294, 20 ] # first peak, second peak
#params_d,cov_d = curve_fit(bimodal,bins_d,y_d,expected_d, maxfev=5000)
#plt.plot(bins_plot_d,bimodal(bins_plot_d,*params_d),color='k',lw=2,label='model')

#If the first peak is small, fit the peaks separated
msk_d = bins_d>-1.18
expected1 =  [-1.121629379, 0.05312294, 4.0 ]
expected2 =  [-4.52, 0.2, 50, ]
params_d1,cov_d1 = curve_fit(gauss,bins_d[msk_d],y_d[msk_d],expected1, maxfev=50000)
params_d2,cov_d2 = curve_fit(gauss,bins_d[~msk_d],y_d[~msk_d],expected2, maxfev=5000)

A1_d, A2_d = integ(bins_d, params_d1, params_d2)

g1 = gauss(bins_plot_d,*params_d1)
g2 = gauss(bins_plot_d,*params_d2)     
#plt.plot(bins_plot_d,g1,color='g',lw=2,label='model')
plt.fill(bins_plot_d,g1, color='r', alpha=0.2)
#plt.plot(bins_plot_d,g2 ,color='r',lw=2,label='model')
plt.fill(bins_plot_d,g2, color='g', alpha=0.2)
plt.plot(bins_plot_d,g2+g1, color='b' ,lw=2,label='model')
plt.annotate('mean: '+"{0:.2f}".format(params_d1[0]), xy=(-6, 70), color='r')
plt.annotate('sigma: '+"{0:.2f}".format(params_d1[1]), xy=(-6, 65), color='r')
plt.annotate('mean: '+"{0:.2f}".format(params_d2[0]), xy=(-6, 60), color='g')
plt.annotate('sigma: '+"{0:.2f}".format(params_d2[1]), xy=(-6, 55), color='g')
plt.annotate("{0:.0f}".format(A1_d)+"%", xy=(params_d1[0]-0.07, params_d1[2]/2.5), fontsize=17, fontweight='bold')
plt.annotate("{0:.0f}".format(A2_d)+"%", xy=(params_d2[0]-0.09, params_d2[2]/3.0), fontsize=17, fontweight='bold')
plt.xlabel('log D (um^2/s)')
plt.ylabel('Count')
plt.title('Diffusion coefficient')




#Skewed gaussian fitting
#skg = lm.models.SkewedGaussianModel()
#params_d_skg = skg.make_params(gamma=5, sigma=0.6, center=-2.3, amplitute=140)
#results_skg = skg.fit(y_d,params_d_skg,x=bins_d )
#params_skg_best = skg.make_params(gamma=results_skg.best_values['gamma'], sigma=results_skg.best_values['sigma'], center=results_skg.best_values['center'], amplitude=results_skg.best_values['amplitude'])
#y_skg = skg.eval(params_skg_best, x=bins_plot_d )


#####Bimodal Lmfit
#bmm = lm.Model(bimodal)
#params_d_bmm = bmm.make_params(mu1=-3.02, sigma1=0.2, A1=100, mu2=0.0121629379, sigma2=20.15312294, A2=20 )
#results_bmm = bmm.fit(y_d, params_d_bmm, x=bins_d)
#
#print(results_bmm.best_values)
#
#
#plt.figure()
#sns.distplot(np.log10(Dif_track), bins=30,  label='Dif', kde=False ,hist_kws=dict(edgecolor="k", linewidth=1))
#plt.plot(bins_plot_d, y_skg, lw=2)
#plt.xlabel('log D (um^2/s)')
#plt.ylabel('Count')
#plt.title('Diffusion coefficient')
#plt.plot(bins_d, results_bmm.best_fit)
#print(results_skg.best_values)



#Plotting Trace
plt.figure()
sns.distplot(np.log10(Dif_pd['Trace Range']/1000.0), bins=48,  label='Trace', kde=False ,hist_kws=dict(edgecolor="k", linewidth=1, alpha=0.3))
plt.xlabel('log Trajectory Range (um)')
plt.ylabel('Count')
plt.title('Trajectory Range')

#plt.figure()
#sns.distplot(Dif_pd['Trace Range']/1000, bins=100,  label='Dif', kde=False ,hist_kws=dict(edgecolor="k",linewidth=1))
#plt.xticks(np.arange(0,2.5,0.2))
#plt.xlabel('Trajectory Range (um)')
#plt.ylabel('Count')
#plt.title('Trajectory Range')

#####fitting Trace
y_t, bins_t = np.histogram(np.log10(Dif_pd['Trace Range']/1000.0), bins=48)
bins_t = bins_t[:-1]

bmm = lm.Model(bimodal)
params_t_bmm = bmm.make_params(mu1=-0.6, sigma1=0.2, A1=50, mu2=0.017121629379, sigma2=0.15312294, A2=10 )
results_t_bmm = bmm.fit(y_t, params_t_bmm, x=bins_t)
params_t_1 = [results_t_bmm.best_values['mu1'], results_t_bmm.best_values['sigma1'], results_t_bmm.best_values['A1']]
params_t_2 = [results_t_bmm.best_values['mu2'], results_t_bmm.best_values['sigma2'], results_t_bmm.best_values['A2']]

A1_t, A2_t = integ(bins_t, params_t_1, params_t_2)

#plt.plot(bins_t, gauss(bins_t,*params_t_1 ), color='r')
plt.fill(bins_t,  gauss(bins_t,*params_t_1 ), color='g', alpha=0.2)
#plt.plot(bins_t, gauss(bins_t,*params_t_2 ), color='g')
plt.fill(bins_t,  gauss(bins_t,*params_t_2 ), color='r', alpha=0.2)
plt.annotate('mean: '+"{0:.2f}".format(params_t_1[0]), xy=(-1.0, 35), color='g')
plt.annotate('sigma: '+"{0:.2f}".format(params_t_1[1]), xy=(-1.0, 30), color='g')
plt.annotate('mean: '+"{0:.2f}".format(params_t_2[0]), xy=(-1.0, 25), color='r')
plt.annotate('sigma: '+"{0:.2f}".format(params_t_2[1]), xy=(-1.0, 20), color='r')
plt.annotate("{0:.0f}".format(A1_t)+"%", xy=(params_t_1[0]-.05, params_t_1[2]/2.5), fontsize=17, fontweight='bold')
plt.annotate("{0:.0f}".format(A2_t)+"%", xy=(params_t_2[0]-0.07, params_t_2[2]/2.5), fontsize=17, fontweight='bold')
plt.plot(bins_t, results_t_bmm.best_fit, color='b', lw=2, label='model')


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