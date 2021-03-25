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

M = pd.read_csv(data_processed + "male.csv", index_col="ecg_id")
F = pd.read_csv(data_processed + "female.csv", index_col="ecg_id")

#%%

sampling_rate = 500

ecgM = load_raw_data(M, sampling_rate, data_raw)
ecgF = load_raw_data(F, sampling_rate, data_raw)

#%%

col = ["age", "bmi", "height", "weight"]

for x in col:
    # unpaired t-test
    print(x, stats.ttest_ind(F[x], M[x]))

# significant features: BMI, Weight, Height

#%%

Subject = pd.concat([M, F], axis=0)
xl = "Sex"
yl = ["Age (yr.)", "BMI", "Height (cm)", "Weight (kg)"]

fig, axs = plt.subplots(1, 4)
fig.suptitle("Sex difference", fontsize=24, y=0.93)
i = 0

for x in col:
    sns.set_theme(palette="viridis", font_scale=0.95)
    sns.violinplot(data=Subject, y=x, x="sex", split=False, ax=axs[i])
    axs[i].set(xlabel=xl, ylabel=yl[i])
    i = i + 1
# Viridis palette for color blinds and b/w prints

#%% BOXPLOT

Subject = pd.concat([M, F], axis=0)
xl = "Sex"
yl = ["Age (yr.)", "BMI", "Height (cm)", "Weight (kg)"]

fig, axs = plt.subplots(1, 4)
fig.suptitle("Sex difference", fontsize=24, y=0.93)
i = 0

for x in col:
    sns.set_theme(palette="viridis", font_scale=0.95)
    sns.boxplot(data=Subject, y=x, x="sex", ax=axs[i])
    sns.swarmplot(data=Subject, y=x, x="sex", ax=axs[i], size=1.5, color="black")
    axs[i].set(xlabel=xl, ylabel=yl[i])
    i = i + 1

fig, axs = plt.subplots(1, 4)
fig.suptitle("Sex difference", fontsize=24, y=0.93)
i = 0

for x in col:
    sns.set_theme(palette="viridis", font_scale=0.95)
    sns.boxplot(data=Subject, y=x, x="sex", ax=axs[i])
    axs[i].set(xlabel=xl, ylabel=yl[i])
    i = i + 1

#%%
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
    x = np.ediff1d(Rf[i]) * (
        1 / sampling_rate
    )  # RR interval calculation (difference between R peaks in time)
    RRf.append(x)

for i in range(ecgM.shape[0]):
    x = np.ediff1d(Rm[i]) * (
        1 / sampling_rate
    )  # RR interval calculation (difference between R peaks in time)
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


print("RR Mean Interval Female", np.mean(RRmeanf))
print("RR Mean Interval Male", np.mean(RRmeanm))

print(stats.ttest_ind(RRmeanf, RRmeanm, axis=0))

# %%

RRstdf = []
RRstdm = []

for i in range(ecgF.shape[0]):
    x = np.std(RRf[i])  # mean of the RR interval for every subjects
    RRstdf.append(x)

for i in range(ecgM.shape[0]):
    x = np.std(RRm[i])  # mean of the RR interval for every subjects
    RRstdm.append(x)

print("RR Mean of Std Interval Female", np.mean(RRstdf))
print("RR Mean of Std Interval Male", np.mean(RRstdm))

print(stats.ttest_ind(RRmeanf, RRmeanm, axis=0))

# %%
# Separation by part of the day: early morning, late morning, evening

date = F.recording_date
x = date[283]

date_obj = datetime.strptime(x, "%Y-%m-%d %H:%M:%S")

# %% Conversion from string to datetime

Hf = []
Hm = []

for i in F.index:
    x = datetime.strptime(F.recording_date[i], "%Y-%m-%d %H:%M:%S")
    Hf.append(x.hour)

for i in M.index:
    x = datetime.strptime(M.recording_date[i], "%Y-%m-%d %H:%M:%S")
    Hm.append(x.hour)

Hf = np.array(Hf)
Hm = np.array(Hm)

# %% Choose bins

# 3 bins: early_morning (4-10), late_morning (11-15), afternoon_evening (16-23)

fig, axs = plt.subplots(1, 1)

bins = [4, 10, 15, 23]
sns.set_theme(palette="viridis")
axs.set(xlabel="Hours of the day", ylabel="Count")
sns.histplot(Hf, bins=bins)
sns.histplot(Hm, bins=bins)

# %% Clustering by datetime

RRmeanf = np.array(RRmeanf)
RRmeanm = np.array(RRmeanm)
RRstdf = np.array(RRstdf)
RRstdm = np.array(RRstdm)

table = Texttable()
table.set_cols_dtype(["t", "f", "f", "e"])
table.add_rows(
    [
        ["stat", "F", "M", "p-value"],
        [
            "(EM) RR mean",
            np.mean(RRmeanf[(Hf >= 4) & (Hf < 10)]),
            np.mean(RRmeanm[(Hm >= 4) & (Hm < 10)]),
            stats.ttest_ind(
                RRmeanf[(Hf >= 4) & (Hf < 10)], RRmeanm[(Hm >= 4) & (Hm < 10)], axis=0
            )[1],
        ],
        [
            "(EM) RR std mean",
            np.mean(RRstdf[(Hf >= 4) & (Hf < 10)]),
            np.mean(RRstdm[(Hm >= 4) & (Hm < 10)]),
            stats.ttest_ind(
                RRstdf[(Hf >= 4) & (Hf < 10)], RRstdm[(Hm >= 4) & (Hm < 10)], axis=0
            )[1],
        ],
        [
            "(LM) RR mean",
            np.mean(RRmeanf[(Hf >= 10) & (Hf < 15)]),
            np.mean(RRmeanm[(Hm >= 10) & (Hm < 15)]),
            stats.ttest_ind(
                RRmeanf[(Hf >= 10) & (Hf < 15)], RRmeanm[(Hm >= 10) & (Hm < 15)], axis=0
            )[1],
        ],
        [
            "(LM) RR std mean",
            np.mean(RRstdf[(Hf >= 10) & (Hf < 15)]),
            np.mean(RRstdm[(Hm >= 10) & (Hm < 15)]),
            stats.ttest_ind(
                RRstdf[(Hf >= 10) & (Hf < 15)], RRstdm[(Hm >= 10) & (Hm < 15)], axis=0
            )[1],
        ],
        [
            "(AE) RR mean",
            np.mean(RRmeanf[(Hf >= 15) & (Hf <= 23)]),
            np.mean(RRmeanm[(Hm >= 15) & (Hm <= 23)]),
            stats.ttest_ind(
                RRmeanf[(Hf >= 15) & (Hf <= 23)],
                RRmeanm[(Hm >= 15) & (Hm <= 23)],
                axis=0,
            )[1],
        ],
        [
            "(AE) RR std mean",
            np.mean(RRstdf[(Hf >= 15) & (Hf <= 23)]),
            np.mean(RRstdm[(Hm >= 15) & (Hm <= 23)]),
            stats.ttest_ind(
                RRstdf[(Hf >= 15) & (Hf <= 23)], RRstdm[(Hm >= 15) & (Hm <= 23)], axis=0
            )[1],
        ],
    ]
)

print(table.draw())

# %% test normality
from scipy.stats import shapiro  # Shapiro-Wilk Test
from scipy.stats import normaltest  # D'Agostino test


# normality test (F)
# stat, p = shapiro(RRmeanf)
stat, p = normaltest(RRmeanf)
print("Statistics=%.3f, p=%.3f" % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
    print("Sample looks Gaussian (fail to reject H0)")
else:
    print("Sample does not look Gaussian (reject H0)")

    # normality test (M)
# stat, p = shapiro(RRmeanm)
stat, p = normaltest(RRmeanm)
print("Statistics=%.3f, p=%.3f" % (stat, p))
# interpret
if p > alpha:
    print("Sample looks Gaussian (fail to reject H0)")
else:
    print("Sample does not look Gaussian (reject H0)")

# %% ANOVA test
from scipy.stats import f_oneway

print("test over Female's RRmean EM/LM/AE")
print(
    f_oneway(
        RRmeanf[(Hf >= 4) & (Hf < 10)],
        RRmeanf[(Hf >= 10) & (Hf < 15)],
        RRmeanf[(Hf >= 15) & (Hf < 23)],
    )
)
print("test over Male's RRmean EM/LM/AE")
print(
    f_oneway(
        RRmeanm[(Hm >= 4) & (Hm < 10)],
        RRmeanm[(Hm >= 10) & (Hm < 15)],
        RRmeanm[(Hm >= 15) & (Hm < 23)],
    )
)

#%% Single File with every feature!

DELTA = 50
DERIVATION = 0
FCuts = []

wavesF = []

for row in np.arange(ecgF.shape[0]):
    _, rpeaks = nk.ecg_peaks(ecgF[row, :, DERIVATION], sampling_rate=sampling_rate)
    _, waves_peak = nk.ecg_delineate(
        ecgF[row, :, DERIVATION], rpeaks, sampling_rate=sampling_rate, show=False
    )

    # convert ECG's dictionaries into array
    # RPeaks = np.array(rpeaks["ECG_R_Peaks"])
    start = np.array(waves_peak["ECG_P_Onsets"]) - DELTA
    waves_peak["ECG_Start"] = start.tolist()
    stop = np.array(waves_peak["ECG_T_Offsets"]) + DELTA
    waves_peak["ECG_Stop"] = stop.tolist()
    waves_peak["ECG_R_Peaks"] = rpeaks["ECG_R_Peaks"].tolist()

    wavesF.append(waves_peak)

# %%

wavesM = []

n = np.arange(ecgM.shape[0])

for row in n:
    _, rpeaks = nk.ecg_peaks(ecgM[row, :, DERIVATION], sampling_rate=sampling_rate)
    _, waves_peak = nk.ecg_delineate(
        ecgM[row, :, DERIVATION], rpeaks, sampling_rate=sampling_rate, show=False
    )

    # convert ECG's dictionaries into array
    # RPeaks = np.array(rpeaks["ECG_R_Peaks"])
    start = np.array(waves_peak["ECG_P_Onsets"]) - DELTA
    waves_peak["ECG_Start"] = start.tolist()
    stop = np.array(waves_peak["ECG_T_Offsets"]) + DELTA
    waves_peak["ECG_Stop"] = stop.tolist()
    waves_peak["ECG_R_Peaks"] = rpeaks["ECG_R_Peaks"].tolist()

    wavesM.append(waves_peak)


# %% features exploration

j = 0

for i in np.arange(len(wavesF[j]["ECG_Start"])):
    start = wavesF[j]["ECG_Start"][i]
    stop = wavesF[j]["ECG_Stop"][i]
    r = wavesF[j]["ECG_R_Peaks"][i]

    x = np.arange(stop - start)
    x = x - r + start

    plt.plot(x, ecgF[j, start:stop, DERIVATION])

# %% export to csv
pd.DataFrame(wavesF).to_csv(data_processed + "waves_F.csv")
pd.DataFrame(wavesM).to_csv(data_processed + "waves_M.csv")
# %%
# 30 has < 0 in ['ECG_P_Onsets'] and NaN in ['ECG_T_Offsets']

# %% from list of dictionaries to 3D list
data_keys = [
    "ECG_Start",
    "ECG_P_Onsets",
    "ECG_P_Peaks",
    "ECG_Q_Peaks",
    "ECG_R_Peaks",
    "ECG_S_Peaks",
    "ECG_T_Peaks",
    "ECG_T_Offsets",
    "ECG_Stop",
]
data_list = []
for row in wavesF:
    pat = []
    for key in data_keys:
        pat.append(list(row[key]))
    data_list.append(pat)

# %% clean 3D list from nan & neg values
for elem in data_list:
    pop_index = []
    for xi in np.arange(len(elem[0])):
        column = []
        for yi in np.arange(len(elem)):
            column.append(elem[yi][xi])
        if has_nan(column).any():
            pop_index.append(xi)
    for xi in reversed(pop_index):
        for yi in np.arange(len(elem)):
            elem[yi].pop(xi)

# %% from 3D list to 3D numpy array
length = max([max(map(len, xi)) for xi in data_list])
data_array = np.array(
    [np.array([yi + [np.nan] * (length - len(yi)) for yi in xi]) for xi in data_list]
)
