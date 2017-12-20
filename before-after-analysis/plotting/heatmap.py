# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 10:49:42 2017

@author: athomaz
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns
sns.set()


fileName_AMPAR1  = 'homer-ampar-before-after-diff-trace-dist-LTD-6.xlsx'
fileName_AMPAR3  = 'homer-ampar-before-after-diff-trace-dist-LTD-7.xlsx'

AMPAR1 = pd.read_excel(fileName_AMPAR1)
AMPAR3 = pd.read_excel(fileName_AMPAR3)

AMPAR = pd.concat([AMPAR1, AMPAR3], ignore_index=True, axis=0)

sns.set()
plt.figure()
sns.kdeplot(np.log10(AMPAR['Diff_Coeff_bef']), np.log10(AMPAR['Trace_Range_bef']/1000), cmap="gray", shade=True, n_levels=10)
sns.kdeplot(np.log10(AMPAR['Diff_Coeff_afe']), np.log10(AMPAR['Trace_Range_afe']/1000), cmap="Reds", shade=False, n_levels=10,  shade_lowest=False )




fileName_NMDAR1  = 'homer-nmdar-before-after-diff-trace-dist-LTD-6.xlsx'
fileName_NMDAR3  = 'homer-nmdar-before-after-diff-trace-dist-LTD-7.xlsx'

NMDAR1 = pd.read_excel(fileName_NMDAR1)
NMDAR3 = pd.read_excel(fileName_NMDAR3)

NMDAR = pd.concat([NMDAR1, NMDAR3], ignore_index=True, axis=0)


sns.set()
plt.figure()
sns.kdeplot(np.log10(NMDAR['Diff_Coeff_bef']), np.log10(NMDAR['Trace_Range_bef']/1000), cmap="gray", shade=True, n_levels=10)
sns.kdeplot(np.log10(NMDAR['Diff_Coeff_afe']), np.log10(NMDAR['Trace_Range_afe']/1000), cmap="Reds", shade=False, n_levels=10,  shade_lowest=False )

