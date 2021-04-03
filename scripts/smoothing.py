#%%
from scipy.interpolate import BSpline, make_interp_spline
import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import skfda
from texttable import Texttable
import sys
import random

sys.path.append("..")
from fda import *

# %% Define constant variables
ENV_FIGURES = False
SAMPLING_RATE = 500
N_CHEB = 21
LEAD = 0
BEAT = 0

# %% Load data
# load datasets
F = pd.read_csv(data_processed + "femaleH.csv", index_col="ecg_id")
M = pd.read_csv(data_processed + "maleH.csv", index_col="ecg_id")

# load ECG interval data
waves_F = np.load(data_processed + "waves_F.npy")
waves_M = np.load(data_processed + "waves_M.npy")

# load ECG signals
ecgM = load_raw_data(M, SAMPLING_RATE, data_raw)
ecgF = load_raw_data(F, SAMPLING_RATE, data_raw)

# %% Bootstrap patients
PATIENT_F = random.choices(range(waves_F.shape[0]), k=5)
PATIENT_M = random.choices(range(waves_M.shape[0]), k=5)

bins = [7, 12, 19, 24]
PATIENT_F_7_12 = [i for i, v in enumerate(F.Hour) if (v >= 7) & (v <= 12)]
PATIENT_M_7_12 = [i for i, v in enumerate(M.Hour) if (v >= 7) & (v <= 12)]
# %% smoothing function
def smoothedECG(
    ECG,
    intervals,
    show_figures=False,
    sampling_rate=SAMPLING_RATE,
    _beat=BEAT,
    _n_cheb=N_CHEB,
):
    # cut heartbeat from patient's ECG
    peakList = [el[_beat] for el in intervals]
    start = int(peakList[0])
    stop = int(peakList[-1])
    knots, t_points = compute_knots(peakList, _n_cheb)

    # create skfda 's FDataGrid data
    heartbeatRaw = skfda.FDataGrid(
        ECG[start:stop, LEAD],
        t_points,
        dataset_name="ECG lead " + str(_beat + 1),
        argument_names=["time"],
        coordinate_names=["mV"],
    )
    # compute basis
    N_BASIS = len(knots) + 2
    basis = skfda.representation.basis.BSpline(
        n_basis=N_BASIS, domain_range=[knots[0], knots[-1]], knots=knots
    )
    # basis.plot()

    # compute smoother
    smoother = skfda.preprocessing.smoothing.BasisSmoother(basis, method="cholesky")

    # compute smoothed curve
    heartbeatSmoothed = smoother.fit_transform(heartbeatRaw)

    y = heartbeatSmoothed.data_matrix[0, :, 0]

    # interpolation to obtain the same number of samples
    f = make_interp_spline(t_points, y)
    new_t = np.linspace(0, t_points[-1], sampling_rate)
    y_new = f(new_t).T

    # plot raw + smoothed ECG
    if show_figures:
        plt.figure()
        plt.plot(t_points, heartbeatRaw.data_matrix[0, :, 0], label="ECG raw")
        # plt.plot(t_points, heartbeatSmoothed.data_matrix[0, :, 0], label="ECG smoothed")
        plt.plot(new_t, y_new)
        plt.legend()
        plt.show()

    return skfda.FDataGrid(y_new)


# %%
def getLandmarks(waveList, patientList, sampling_rate=SAMPLING_RATE):
    peak = waveList[patientList, :, 0]
    sub = peak[:, 0]

    peak = [(peak[:, i] - peak[:, 0]) / sampling_rate for i in range(peak.shape[1])]
    peak = np.transpose(np.array(peak))
    peak = peak[:, 1:8]

    return np.array(peak)


# %%
smoothedF = [
    smoothedECG(ecgF[p], waves_F[p], show_figures=False) for p in PATIENT_F_7_12
]

fd_F = smoothedF[0]
for fi in range(len(smoothedF)):
    if fi > 0:
        fd_F = fd_F.concatenate(smoothedF[fi])

# %% Alignment LANDMARK FEATURE
landF = getLandmarks(waves_F, PATIENT_F_7_12, sampling_rate=SAMPLING_RATE)

#%%
warpingF = skfda.preprocessing.registration.landmark_registration_warping(
    fd_F, landF, location=np.mean(landF, axis=0)
)

fig = warpingF.plot()

# Plot landmarks
for index, value in enumerate(PATIENT_F_7_12):
    fig.axes[0].scatter(
        np.mean(landF, axis=0), landF[index], label="Patient_" + str(value)
    )
    plt.legend()


# %%
fd_registered_F = fd_F.compose(warpingF)
fig = fd_registered_F.plot()

# %% Males
smoothedM = [
    smoothedECG(ecgM[p], waves_M[p], show_figures=False) for p in PATIENT_M_7_12
]

fd_M = smoothedM[0]
for fi in range(len(smoothedM)):
    if fi > 0:
        fd_M = fd_M.concatenate(smoothedM[fi])

# %% Alignment LANDMARK FEATURE
landM = getLandmarks(waves_M, PATIENT_M_7_12, sampling_rate=SAMPLING_RATE)

#%%
warpingM = skfda.preprocessing.registration.landmark_registration_warping(
    fd_M, landM, location=np.mean(landM, axis=0)
)

fig = warpingM.plot()

# Plot landmarks
for index, value in enumerate(PATIENT_M_7_12):
    fig.axes[0].scatter(
        np.mean(landM, axis=0), landM[index], label="Patient_" + str(value)
    )
    # plt.legend()


# %%
fd_registered_M = fd_M.compose(warpingM)
fig = fd_registered_M.plot()


# %%
fig = plt.figure()
for i in range(warpingF.n_samples):
    if i == 0:
        plt.plot(
            warpingF.data_matrix[i, :, 0], "r", alpha=0.8, linewidth=0.2, label="Female"
        )
    else:
        plt.plot(warpingF.data_matrix[i, :, 0], "r", alpha=0.8, linewidth=0.2)
for i in range(warpingM.n_samples):
    if i == 0:
        plt.plot(
            warpingM.data_matrix[i, :, 0], "b", alpha=0.7, linewidth=0.2, label="Male"
        )
    else:
        plt.plot(warpingM.data_matrix[i, :, 0], "b", alpha=0.7, linewidth=0.2)
plt.legend()
