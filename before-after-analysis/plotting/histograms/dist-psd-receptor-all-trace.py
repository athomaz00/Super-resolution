# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 17:14:25 2017

@author: athomaz
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

import seaborn as sns
sns.set()

fileName_AMPAR_LTD3  = 'homer-ampar-before-after-diff-trace-dist-LTD-3.xlsx'
fileName_NMDAR_LTD3  = 'homer-nmdar-before-after-diff-trace-dist-LTD-3.xlsx'

fileName_AMPAR_Tracking_before = 'ampar-before-tracking-LTD-3.xlsx'
fileName_AMPAR_Tracking_after = 'ampar-after-tracking-LTD-3.xlsx'

AMPAR_Tracking_before = pd.read_excel(fileName_AMPAR_Tracking_before)
AMPAR_Tracking_after = pd.read_excel(fileName_AMPAR_Tracking_after)


AMPAR_HOMER = pd.read_excel(fileName_AMPAR_LTD3)
NMDAR_HOMER = pd.read_excel(fileName_NMDAR_LTD3)

AMPAR_Dist_before =[]
for i, row in AMPAR_HOMER.iterrows():
    ampar_number = row['Receptor_Numer_bef']
    ampar_points = AMPAR_Tracking_before['particle'] == ampar_number