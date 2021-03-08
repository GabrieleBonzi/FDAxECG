import pandas as pd
import numpy as np
import wfdb
import seaborn as sns
from scipy import stats
import neurokit2 as nk
import matplotlib.pyplot as plt 
import sys
sys.path.append("..")
from fda import *

#%%

def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data

#%%

M = pd.read_csv(data_processed+'male.csv', index_col='ecg_id')
F = pd.read_csv(data_processed+'female.csv', index_col='ecg_id')

#%%

sampling_rate = 500

ecgM = load_raw_data(M, sampling_rate, data_raw)
ecgF = load_raw_data(F, sampling_rate, data_raw)

#%%

col = ['age', 'bmi', 'height', 'weight']

for x in col:
    # unpaired t-test
    print(x, stats.ttest_ind(F[x], M[x]))
    
# significant features: BMI, Weight, Height

#%%

Subject = pd.concat([M, F], axis=0)
xl = 'Sex'
yl = ['Age (yr.)', 'BMI', 'Height (cm)', 'Weight (kg)']

fig, axs = plt.subplots(1, 4)
fig.suptitle('Sex difference', fontsize=24, y=0.93)
i = 0

for x in col:
    sns.set_theme(palette='viridis', font_scale=0.95)
    sns.violinplot(data=Subject, y=x, x='sex', split=False, ax=axs[i])
    axs[i].set(xlabel=xl, ylabel=yl[i])
    i = i + 1
# Viridis palette for color blinds and b/w prints
    
#%%
# R peaks (Pan-Tompkins)
# Extraction of the R peaks from the ECG signals
Rf = []
Rm = []

for i in range(ecgF.shape[0]):
    _, rpeaks = nk.ecg_peaks(ecgF[i,:,1], sampling_rate=sampling_rate)
    Rf.append(rpeaks.get('ECG_R_Peaks'))
    
for i in range(ecgM.shape[0]):
    _, rpeaks = nk.ecg_peaks(ecgM[i,:,1], sampling_rate=sampling_rate)
    Rm.append(rpeaks.get('ECG_R_Peaks'))

# %%
# RR interval

RRf = []
RRm = []

for i in range(ecgF.shape[0]):
    x = np.ediff1d(Rf[i]) * (1/sampling_rate) #RR interval calculation (difference between R peaks in time)
    RRf.append(x)

for i in range(ecgM.shape[0]):
    x = np.ediff1d(Rm[i]) * (1/sampling_rate) #RR interval calculation (difference between R peaks in time)
    RRm.append(x)

# %%
# t-test on RR(s) for both groups

RRmeanf = []
RRmeanm = []

for i in range(ecgF.shape[0]):
    x = np.mean(RRf[i]) #mean of the RR interval for every subjects
    RRmeanf.append(x)
    
for i in range(ecgM.shape[0]):
    x = np.mean(RRm[i]) #mean of the RR interval for every subjects
    RRmeanm.append(x)


print('RR Mean Interval Female', np.mean(RRmeanf))
print('RR Mean Interval Male', np.mean(RRmeanm))

print(stats.ttest_ind(RRmeanf, RRmeanm, axis=0))

# %%

RRstdf = []
RRstdm = []

for i in range(ecgF.shape[0]):
    x = np.std(RRf[i]) #mean of the RR interval for every subjects
    RRstdf.append(x)
    
for i in range(ecgM.shape[0]):
    x = np.std(RRm[i]) #mean of the RR interval for every subjects
    RRstdm.append(x)

print('RR Mean of Std Interval Female', np.mean(RRstdf))
print('RR Mean of Std Interval Male', np.mean(RRstdm))

print(stats.ttest_ind(RRmeanf, RRmeanm, axis=0))


