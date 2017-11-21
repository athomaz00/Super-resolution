# -*- coding: utf-8 -*-
"""
Spyder Editor

Este é um arquivo de script temporário.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns
sns.set(color_codes=True)

mybins  = np.arange(-500,500,50)
xlim = (-500,500)
ylim = (0,0.0045)




#plt.close("all")
##########Ampar Control##############
HAb = pd.read_excel('dist-homer-ampar-before-ctr-1.xlsx')
HAa = pd.read_excel('dist-homer-ampar-after-ctr-1.xlsx')
centerBA = np.load('homer-ampar-BfAf-CTR-1.npy')
dist_AP_CTR1 = HAa[HAa['Homer Number']==np.sort(centerBA[:,1])].Distance.values - HAb[HAb['Homer Number']==centerBA[:,0]].Distance.values

HAb = pd.read_excel('dist-homer-ampar-before-ctr-2.xlsx')
HAa = pd.read_excel('dist-homer-ampar-after-ctr-2.xlsx')
centerBA = np.load('homer-ampar-BfAf-CTR-2.npy')
dist_AP_CTR2 = HAa[HAa['Homer Number']==np.sort(centerBA[:,1])].Distance.values - HAb[HAb['Homer Number']==centerBA[:,0]].Distance.values

HAb = pd.read_excel('dist-homer-ampar-before-ctr-3.xlsx')
HAa = pd.read_excel('dist-homer-ampar-after-ctr-3.xlsx')
centerBA = np.load('homer-ampar-BfAf-CTR-3.npy')
dist_AP_CTR3 = HAa[HAa['Homer Number']==np.sort(centerBA[:,1])].Distance.values - HAb[HAb['Homer Number']==centerBA[:,0]].Distance.values


HAb = pd.read_excel('dist-homer-ampar-before-ctr-4.xlsx')
HAa = pd.read_excel('dist-homer-ampar-after-ctr-4.xlsx')
centerBA = np.load('homer-ampar-BfAf-CTR-4.npy')
dist_AP_CTR4 = HAa[HAa['Homer Number']==np.sort(centerBA[:,1])].Distance.values - HAb[HAb['Homer Number']==centerBA[:,0]].Distance.values
    
dist_AP_CTR = np.concatenate((dist_AP_CTR1, dist_AP_CTR2))
label_AP_CTR = len(dist_AP_CTR)



#################AMPAR LTD######################
HALb = pd.read_excel('dist-homer-ampar-before-LTD-1.xlsx')
HALa = pd.read_excel('dist-homer-ampar-after-LTD-1.xlsx')
centerBA_LTD = np.load('homer-ampar-BfAf-LTD-1.npy')
dist_AP_LTD1 = HALa[HALa['Homer Number']==np.sort(centerBA_LTD[:,1])].Distance.values - HALb[HALb['Homer Number']==centerBA_LTD[:,0]].Distance.values

HALb = pd.read_excel('dist-homer-ampar-before-LTD-2.xlsx')
HALa = pd.read_excel('dist-homer-ampar-after-LTD-2.xlsx')
centerBA_LTD = np.load('homer-ampar-BfAf-LTD-2.npy')
dist_AP_LTD2 = HALa[HALa['Homer Number']==np.sort(centerBA_LTD[:,1])].Distance.values - HALb[HALb['Homer Number']==centerBA_LTD[:,0]].Distance.values

HALb = pd.read_excel('dist-homer-ampar-before-LTD-3.xlsx')
HALa = pd.read_excel('dist-homer-ampar-after-LTD-3.xlsx')
centerBA_LTD = np.load('homer-ampar-BfAf-LTD-3.npy')
dist_AP_LTD3 = HALa[HALa['Homer Number']==np.sort(centerBA_LTD[:,1])].Distance.values - HALb[HALb['Homer Number']==centerBA_LTD[:,0]].Distance.values

HALb = pd.read_excel('dist-homer-ampar-before-LTD-4.xlsx')
HALa = pd.read_excel('dist-homer-ampar-after-LTD-4.xlsx')
centerBA_LTD = np.load('homer-ampar-BfAf-LTD-4.npy')
dist_AP_LTD4 = HALa[HALa['Homer Number']==np.sort(centerBA_LTD[:,1])].Distance.values - HALb[HALb['Homer Number']==centerBA_LTD[:,0]].Distance.values

HALb = pd.read_excel('dist-homer-ampar-before-LTD-5.xlsx')
HALa = pd.read_excel('dist-homer-ampar-after-LTD-5.xlsx')
centerBA_LTD = np.load('homer-ampar-BfAf-LTD-5.npy')
dist_AP_LTD5 = HALa[HALa['Homer Number']==np.sort(centerBA_LTD[:,1])].Distance.values - HALb[HALb['Homer Number']==centerBA_LTD[:,0]].Distance.values


dist_AP_LTD = np.concatenate((dist_AP_LTD1, dist_AP_LTD3,dist_AP_LTD5 ))
label_AP_LTD = len(dist_AP_LTD)

plt.figure()
sns.distplot(dist_AP_LTD, label='LTD '+str(label_AP_LTD), bins=mybins, kde=False, hist_kws=dict(edgecolor="k", linewidth=1,  normed=True))
sns.distplot(dist_AP_CTR, label='Control ' +str(label_AP_CTR) , bins=mybins, kde=False ,hist_kws=dict(edgecolor="k", linewidth=1,  normed=True))

plt.xlim(xlim)
plt.ylim(ylim)
#plt.plot(xline, yline)
#plt.plot(xlinep, yline, 'r')
plt.legend()
plt.xlabel('Relative Distance (nm)')
plt.ylabel('Relative Frequency')
plt.title('Relative Distance Homer-Ampar')
plt.xticks(np.arange(-500,600,100))



###############NMDAR Control####################
HNb = pd.read_excel('dist-homer-nmdar-before-CTR-1.xlsx')
HNa = pd.read_excel('dist-homer-nmdar-after-CTR-1.xlsx')
centerBA_CTR = np.load('homer-nmdar-BfAf-CTR-1.npy')
dist_NM_CTR1 = HNa[HNa['Homer Number']==np.sort(centerBA_CTR[:,1])].Distance.values - HNb[HNb['Homer Number']==centerBA_CTR[:,0]].Distance.values

HNb = pd.read_excel('dist-homer-nmdar-before-CTR-2.xlsx')
HNa = pd.read_excel('dist-homer-nmdar-after-CTR-2.xlsx')
centerBA_CTR = np.load('homer-nmdar-BfAf-CTR-2.npy')
dist_NM_CTR2 = HNa[HNa['Homer Number']==np.sort(centerBA_CTR[:,1])].Distance.values - HNb[HNb['Homer Number']==centerBA_CTR[:,0]].Distance.values



dist_NM_CTR = np.concatenate((dist_NM_CTR1,dist_NM_CTR2 ))

label_NM_CTR = len(dist_NM_CTR)


##############NMDAR LTD###############################
HNLb = pd.read_excel('dist-homer-nmdar-before-LTD-1.xlsx')
HNLa = pd.read_excel('dist-homer-nmdar-after-LTD-1.xlsx')
centerBA_LTD = np.load('homer-nmdar-BfAf-LTD-1.npy')
dist_NM_LTD1 = HNLa[HNLa['Homer Number']==np.sort(centerBA_LTD[:,1])].Distance.values - HNLb[HNLb['Homer Number']==centerBA_LTD[:,0]].Distance.values

HNLb = pd.read_excel('dist-homer-nmdar-before-LTD-2.xlsx')
HNLa = pd.read_excel('dist-homer-nmdar-after-LTD-2.xlsx')
centerBA_LTD = np.load('homer-nmdar-BfAf-LTD-2.npy')
dist_NM_LTD2 = HNLa[HNLa['Homer Number']==np.sort(centerBA_LTD[:,1])].Distance.values - HNLb[HNLb['Homer Number']==centerBA_LTD[:,0]].Distance.values

HNLb = pd.read_excel('dist-homer-nmdar-before-LTD-3.xlsx')
HNLa = pd.read_excel('dist-homer-nmdar-after-LTD-3.xlsx')
centerBA_LTD = np.load('homer-nmdar-BfAf-LTD-3.npy')
dist_NM_LTD3 = HNLa[HNLa['Homer Number']==np.sort(centerBA_LTD[:,1])].Distance.values - HNLb[HNLb['Homer Number']==centerBA_LTD[:,0]].Distance.values

dist_NM_LTD = np.concatenate((dist_NM_LTD1, dist_NM_LTD3 ))

label_NM_LTD = len(dist_NM_LTD)





plt.figure()
sns.distplot(dist_NM_LTD, label='LTD '+str(label_NM_LTD), bins=mybins, kde=False, hist_kws=dict(edgecolor="k", linewidth=1,  normed=True))
sns.distplot(dist_NM_CTR, label='Control ' +str(label_NM_CTR) , bins=mybins, kde=False ,hist_kws=dict(edgecolor="k", linewidth=1,  normed=True))
plt.xlim(xlim)
plt.ylim(ylim)
#plt.plot(xline, yline)
#plt.plot(xlinep, yline, 'r')
plt.legend()
plt.xlabel('Relative Distance (nm)')
plt.ylabel('Relative Frequency')
plt.title('Relative Distance Homer-Nmdar')
plt.xticks(np.arange(-500,600,100))





