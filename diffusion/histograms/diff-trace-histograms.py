# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 22:38:10 2017

@author: athomaz
"""

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



#################################################
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
#############################################


######Load saved diffusion data###############
Dif_pd = pd.read_excel('ampar-after-diff-LTD-3-4.xlsx')



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