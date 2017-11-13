# -*- coding: utf-8 -*-
"""
Spyder Editor

Este é um arquivo de script temporário.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns
sns.set(color_codes=True)

########################### Window to ask for files
import tkinter as tk
from tkinter import filedialog
def select_file(data_dir):
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes('-topmost', 1)
    root.focus_force()
    file = filedialog.askopenfilename(parent=root, initialdir=data_dir)
    

    return file
########################################
dirPath = os.path.normpath(r'C:\Users\athomaz\Google Drive\superresolution\analysis\mycodes\superres-matlab\cluster\data test\plot by distance')



mybins  = np.arange(-500,500,50)

plt.close("all")
##########Ampar Close to soma##############
fileName1 = select_file(dirPath)
fileName2 = select_file(dirPath)
fileName3 = select_file(dirPath)
HAC1 = pd.read_excel(fileName1, names= ['before', 'after'])
HAC2 = pd.read_excel(fileName2, names= ['before', 'after'])
HAC3 = pd.read_excel(fileName3, names= ['before', 'after'])
#HA2 = pd.read_excel('dist-change-homer-ampar-ctr-2.xlsx', names= ['before', 'after'])
HA = pd.concat([HAC1,HAC2,HAC3],ignore_index=True)
synNumbHACtr = HAC.shape[0]
HAC['dist'] = HAC['after'] - HAC['before']
labelAMPARClose= 'LTD NMDAR close'

#############AMPAR Control Close to Soma
fileName4 = select_file(dirPath)
fileName5 = select_file(dirPath)
#fileName6 = select_file(dirPath)
HCC1 = pd.read_excel(fileName2, names= ['before', 'after'])
HCC2 = pd.read_excel(fileName3, names= ['before', 'after'])
#HCC3 = pd.read_excel(fileName6, names= ['before', 'after'])
#HA2 = pd.read_excel('dist-change-homer-ampar-ctr-2.xlsx', names= ['before', 'after'])
HCC = pd.concat([HCC1,HCC2],ignore_index=True)
synNumbHACtr = HCC.shape[0]
HCC['dist'] = HCC['after'] - HCC['before']
labelAMPARCloseCTR= 'CTR NMDAR close'


plt.figure()
sns.distplot(HAC['dist'], bins=mybins, label='NMDAR LTD Far', kde=False, hist_kws=dict(edgecolor="k", linewidth=1, normed=True))
sns.distplot(HCC['dist'],  bins=mybins, label='NMDAR CTR Far', kde=False, hist_kws=dict(edgecolor="k", linewidth=1, normed=True))
plt.xlabel('Homer-NMDAR Distance (nm)')
plt.legend()


plt.show()