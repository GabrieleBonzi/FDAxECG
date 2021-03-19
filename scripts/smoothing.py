#%%
from datetime import datetime
import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import skfda
from texttable import Texttable
import wfdb
import sys

sys.path.append("..")
from fda import *


#%%
F = pd.read_csv(data_processed + "female.csv", index_col="ecg_id")
M = pd.read_csv(data_processed + "male.csv", index_col="ecg_id")
waves_F = pd.read_csv(data_processed + "waves_F.csv", index_col=0)
waves_M = pd.read_csv(data_processed + "waves_M.csv", index_col=0)

# %%
SAMPLING_RATE = 500
PATIENT = 2
LEAD = 0
BEAT = 0

ecgM = load_raw_data(M, SAMPLING_RATE, data_raw)
ecgF = load_raw_data(F, SAMPLING_RATE, data_raw)
# %%
start = list(eval(waves_F.ECG_Start[PATIENT]))[BEAT]
stop = list(eval(waves_F.ECG_Stop[PATIENT]))[BEAT]
POnset = list(eval(waves_F.ECG_P_Onsets[PATIENT]))[BEAT]
PPeak = list(eval(waves_F.ECG_P_Peaks[PATIENT]))[BEAT]
QPeak = list(eval(waves_F.ECG_Q_Peaks[PATIENT]))[BEAT]
RPeak = list(eval(waves_F.ECG_R_Peaks[PATIENT]))[BEAT]
SPeak = list(eval(waves_F.ECG_S_Peaks[PATIENT]))[BEAT]
TPeak = list(eval(waves_F.ECG_T_Peaks[PATIENT]))[BEAT]
TOffset = list(eval(waves_F.ECG_T_Offsets[PATIENT]))[BEAT]
# knots = [start, (start+POnset)*0.5, POnset, (POnset+PPeak)*0.5, PPeak, (PPeak+QPeak)*0.5, QPeak, (QPeak+RPeak)*0.5, RPeak, (RPeak+SPeak)*0.5, SPeak, (SPeak+TPeak)*0.5, TPeak, (TPeak+TOffset)*0.5, TOffset, (TOffset+stop)*0.5, stop]
t_knots = [start, POnset, PPeak, QPeak, RPeak, SPeak, TPeak, TOffset, stop]

for d in range(2):
    knots = []
    knots.append(t_knots[0])
    for i in range(len(t_knots) - 1):
        knots.append((t_knots[i] + t_knots[i + 1]) * 0.5)
        knots.append(t_knots[i + 1])
    t_knots = knots
knots = (np.array(knots) - start) / SAMPLING_RATE
#
t_points = np.linspace(0, (stop - start) / SAMPLING_RATE, (stop - start))
len(t_points)
# knots = np.linspace(0, (stop - start)/SAMPLING_RATE, 35)
# %%

fd_F = skfda.FDataGrid(
    ecgF[PATIENT, start:stop, LEAD],
    t_points,
    dataset_name="ECG leads",
    argument_names=["time"],
    coordinate_names=["mV"],
)
# %% basis
# fd_F_basis = fd_F.to_basis(skfda.representation.basis.BSpline(n_basis=4))
N_BASIS = 2 + len(knots)
print(N_BASIS)
basis = skfda.representation.basis.BSpline(
    n_basis=N_BASIS, domain_range=[0, (stop - start) / SAMPLING_RATE], knots=knots
)
basis.plot()
# %% smoother
smoother = skfda.preprocessing.smoothing.BasisSmoother(basis, method="cholesky")
# %% smooothato
fd_F_smoothed = smoother.fit_transform(fd_F)
# %%
# fd_smoothed.data_matrix.round(2)
fig1 = fd_F.plot()
fig2 = fd_F_smoothed.plot()
plt.show()

# %%
fd_axis = []
fd_s_axis = []
for b in range(317):
    fd_axis.append(fd_F.data_matrix[0, b, 0])
    fd_s_axis.append(fd_F_smoothed.data_matrix[0, b, 0])
fd_axis = np.array(fd_axis)
fd_s_axis = np.array(fd_s_axis)
print(len(fd_axis), len(fd_axis))
# %%
plt.figure()
plt.plot(fd_axis, label="ECG raw")
plt.plot(fd_s_axis, label="ECG smoothed")
plt.legend()
plt.show()
# %%
plt.figure()
plt.plot(abs(fd_axis - fd_s_axis), label="Raw - Smoothed")
plt.legend()
plt.show()
# %%
print(np.sum(abs(fd_axis - fd_s_axis)))
print(np.sum((fd_axis - fd_s_axis) * (fd_axis - fd_s_axis)))
