# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 16:51:01 2021

@author: asnag
"""

import pandas as pd
import numpy as np
import wfdb
import seaborn as sns
from scipy import stats
import neurokit2 as nk

#%%

def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data

#%%
path='../processed_data/'

M=pd.read_csv(path+'male.csv', index_col='ecg_id')
F=pd.read_csv(path+'female.csv',index_col='ecg_id')

#%%
#path='C:/Users/asnag/OneDrive - Politecnico di Milano/NECSTCamp/Research Project/GIIIIIT/FDAxECG/data/'
x='../data/'

sampling_rate=500

ecgM=load_raw_data(M, sampling_rate, x)
ecgF=load_raw_data(F, sampling_rate, x)

#%%

col=['age','bmi','height','weight']

for x in col:
    # unpaired t-test
    print(x,stats.ttest_ind(F[x],M[x]))
    
# SIGNIFICATIVI: BMI, Weight, Height

#%%
import matplotlib.pyplot as plt 

Subject=pd.concat([M,F],axis=0)
xl='Sex'
yl=['Age (yr.)','BMI','Height (cm)','Weight (kg)']

fig, axs = plt.subplots(1,4)
fig.suptitle('Sex difference', fontsize=24,y=0.93)
i=0

for x in col:
    sns.set_theme(palette='viridis', font_scale=0.95)
    sns.violinplot(data=Subject,y=x,x='sex',split=False,ax=axs[i])
    axs[i].set(xlabel=xl, ylabel=yl[i])
    i=i+1
# Viridis: per daltonici e b/n
    
#%%
# Trovare Picchi R
Rf=[]
Rm=[]

for i in range(ecgF.shape[0]):
    _, rpeaks = nk.ecg_peaks(ecgF[i,:,1], sampling_rate=500)
    Rf.append(rpeaks.get('ECG_R_Peaks'))
    
for i in range(ecgM.shape[0]):
    _, rpeaks = nk.ecg_peaks(ecgM[i,:,1], sampling_rate=500)
    Rm.append(rpeaks.get('ECG_R_Peaks'))


# %%
    