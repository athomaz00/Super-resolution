# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 13:37:10 2017

@author: athomaz
"""

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import numpy as np
import pandas as pd
import trackpy as tp

##############################################################################
#File names
fileName_QD = 'ampar-before-tracking-LTD-7.xlsx'


##############################################################################

data_QD = pd.read_excel(fileName_QD)