# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

from sklearn.metrics.pairwise import euclidean_distances

import seaborn as sns
sns.set()

fileName_AMPAR_LTD3  = 'homer-ampar-before-after-diff-trace-dist-CTR-5.xlsx'
#fileName_NMDAR_LTD3  = 'homer-nmdar-before-after-diff-trace-dist-LTD-3.xlsx'

fileName_AMPAR_Tracking_before = 'ampar-before-tracking-CTR-5.xlsx'
fileName_AMPAR_Tracking_after = 'ampar-after-tracking-CTR-5.xlsx'

AMPAR_Tracking_before = pd.read_excel(fileName_AMPAR_Tracking_before)
AMPAR_Tracking_after = pd.read_excel(fileName_AMPAR_Tracking_after)


AMPAR_HOMER = pd.read_excel(fileName_AMPAR_LTD3)
#NMDAR_HOMER = pd.read_excel(fileName_NMDAR_LTD3)

bins  = np.arange(0, 2200, 50)

AMPAR_Dist_before =[]
for i, row in AMPAR_HOMER.iterrows():
    ampar_number = row['Receptor_Number_bef']
    ampar_points = AMPAR_Tracking_before[AMPAR_Tracking_before['particle'] == ampar_number]
    ampar_points_bef = ampar_points.iloc[:,5:8].values
    homer_bef = row['Homer_x_bef':'Homer_z_bef']
    distance_bef = euclidean_distances(homer_bef.values.reshape(1,-1), ampar_points_bef)
    ampar_number = row['Receptor_Number_afe']
    ampar_points = AMPAR_Tracking_after[AMPAR_Tracking_after['particle'] == ampar_number]
    ampar_points_aft = ampar_points.iloc[:,5:8].values
    homer_aft = row['Homer_x_afe':'Homer_z_afe']
    distance_aft = euclidean_distances(homer_aft.values.reshape(1,-1), ampar_points_aft)
    
    
    
    #plt.figure()
    #sns.distplot(distance_aft, bins=bins, label='aft', kde=True, hist_kws=dict(edgecolor="k", linewidth=1, normed=True))
    #sns.distplot(distance_bef, bins=bins, label='bef', kde=True , hist_kws=dict(edgecolor="k", linewidth=1, normed=True))
   
    plt.legend()
    