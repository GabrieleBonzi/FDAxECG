#%%
from datetime import datetime
import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
from texttable import Texttable
import wfdb
import sys

sys.path.append("..")
from fda import *


#%%

M = pd.read_csv(data_processed + "male_ages.csv", index_col="ecg_id")
F = pd.read_csv(data_processed + "female_ages.csv", index_col="ecg_id")

F = F.drop([F.index[530], F.index[1196]])
#%%

sampling_rate = 500

ecgM = load_raw_data(M, sampling_rate, data_raw)
ecgF = load_raw_data(F, sampling_rate, data_raw)

# R peaks (Pan-Tompkins)
# Extraction of the R peaks from the ECG signals
Rf = []
Rm = []

for i in range(ecgF.shape[0]):
    _, rpeaks = nk.ecg_peaks(ecgF[i, :, 1], sampling_rate=sampling_rate)
    Rf.append(rpeaks.get("ECG_R_Peaks"))

for i in range(ecgM.shape[0]):
    _, rpeaks = nk.ecg_peaks(ecgM[i, :, 1], sampling_rate=sampling_rate)
    Rm.append(rpeaks.get("ECG_R_Peaks"))

# %%
# RR interval

RRf = []
RRm = []

for i in range(ecgF.shape[0]):
    # RR interval calculation (difference between R peaks in time)
    x = np.ediff1d(Rf[i]) * (1 / sampling_rate)
    RRf.append(x)

for i in range(ecgM.shape[0]):
    # RR interval calculation (difference between R peaks in time)
    x = np.ediff1d(Rm[i]) * (1 / sampling_rate)
    RRm.append(x)

# %%
# t-test on RR(s) for both groups

RRmeanf = []
RRmeanm = []

for i in range(ecgF.shape[0]):
    x = np.mean(RRf[i])  # mean of the RR interval for every subjects
    RRmeanf.append(x)

for i in range(ecgM.shape[0]):
    x = np.mean(RRm[i])  # mean of the RR interval for every subjects
    RRmeanm.append(x)


# print("RR Mean Interval Female", np.mean(RRmeanf))
# print("RR Mean Interval Male", np.mean(RRmeanm))
# print(stats.ttest_ind(RRmeanf, RRmeanm, axis=0))

# %%

RRstdf = []
RRstdm = []

for i in range(ecgF.shape[0]):
    x = np.std(RRf[i])  # mean of the RR interval for every subjects
    RRstdf.append(x)

for i in range(ecgM.shape[0]):
    x = np.std(RRm[i])  # mean of the RR interval for every subjects
    RRstdm.append(x)

# print("RR Mean of Std Interval Female", np.mean(RRstdf))
# print("RR Mean of Std Interval Male", np.mean(RRstdm))
# print(stats.ttest_ind(RRmeanf, RRmeanm, axis=0))

# separation by age
# clusters: [0-1], (1-5], (5-16], (16-30], (30-50], (50-70], (70-90]
RRmeanf = np.array(RRmeanf)
RRmeanm = np.array(RRmeanm)
RRstdf = np.array(RRstdf)
RRstdm = np.array(RRstdm)

ages = [0, 1, 5, 16, 30, 50, 70, 90]

TAges = Texttable()
TAges.set_cols_dtype(["t", "f", "f", "e"])
for i in range(len(ages) - 1):
    TAges.add_rows(
        [
            ["age", "F", "M", "p-value"],
            [
                str(ages[i]) + "-" + str(ages[i + 1]) + " RR mean",
                np.mean(RRmeanf[(F.age >= ages[i]) & (F.age <= ages[i + 1])]),
                np.mean(RRmeanm[(M.age >= ages[i]) & (M.age <= ages[i + 1])]),
                stats.ttest_ind(
                    RRmeanf[(F.age >= ages[i]) & (F.age <= ages[i + 1])],
                    RRmeanm[(M.age >= ages[i]) & (M.age <= ages[i + 1])],
                    axis=0,
                )[1],
            ],
            [
                str(ages[i]) + "-" + str(ages[i + 1]) + " RR std mean",
                np.mean(RRstdf[(F.age >= ages[i]) & (F.age <= ages[i + 1])]),
                np.mean(RRstdm[(M.age >= ages[i]) & (M.age <= ages[i + 1])]),
                stats.ttest_ind(
                    RRstdf[(F.age >= ages[i]) & (F.age <= ages[i + 1])],
                    RRstdm[(M.age >= ages[i]) & (M.age <= ages[i + 1])],
                    axis=0,
                )[1],
            ],
        ]
    )

print(TAges.draw())
# %%
fig, axs = plt.subplots(1, 1)

sns.set_theme(palette="viridis")
axs.set(xlabel="Years Old", ylabel="Count")
sns.histplot(F.age, bins=ages)
sns.histplot(M.age, bins=ages)
