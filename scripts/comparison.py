#%%
from scipy import signal
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
import time
import statsmodels.api as sm
from skfda.exploratory.visualization import Boxplot
from skfda.preprocessing.dim_reduction.projection import FPCA
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import LineCollection
from matplotlib import colors as mcolors
from statistics import mean, median

sys.path.append("..")
from fda import *

# %%
def centeredFiniteDistance2D(x, y):
    even = y[:, ::2]
    odd = y[:, 1::2]

    dx = np.diff(x)[0]

    d_odd = np.diff(odd) / (2 * dx)
    d_even = np.diff(even) / (2 * dx)

    z = np.zeros((y.shape[0], y.shape[1] - 2))
    z[:, ::2] = d_even
    z[:, 1::2] = d_odd
    return z


def compute_knots2(peakList=[0.0, 1.0], n_chebyshev=20, sampling_rate=500):
    if len(peakList) < 9:
        raise Exception(
            "Input list too short({}), expected to be 9".format(len(peakList))
        )
    start = peakList[0]
    stop = peakList[-1]
    middle = peakList[4]
    t_points = np.linspace(0, (stop - start) / sampling_rate, int((stop - start)))

    chebyshev = np.polynomial.chebyshev.chebpts1(n_chebyshev * 2)
    cheb_start = chebyshev[int(len(chebyshev) / 2) :]
    cheb_stop = chebyshev[: int(len(chebyshev) / 2)]

    a = np.interp(chebyshev, (cheb_start.min(), cheb_start.max()), (start, middle))
    b = np.interp(chebyshev, (cheb_stop.min(), cheb_stop.max()), (middle, stop))

    knots = np.concatenate((a, b, np.array(peakList)))
    knots = np.unique(knots)
    knots = np.sort(knots)
    knots = (knots - start) / sampling_rate

    return knots, t_points


# %% Define constant variables
ENV_FIGURES = False
SAMPLING_RATE = 500
N_CHEB = 21
LEAD = 0
BEAT = 0
PREFIX = "sttc_"

# %% Load data
# load datasets
F = pd.read_csv(data_processed + "femaleH.csv", index_col="ecg_id")
M = pd.read_csv(data_processed + "maleH.csv", index_col="ecg_id")

PDnorm = pd.concat([F, M])

F_sttc = pd.read_csv(data_processed + PREFIX + "femaleH.csv", index_col="ecg_id")
M_sttc = pd.read_csv(data_processed + PREFIX + "maleH.csv", index_col="ecg_id")

PDsttc = pd.concat([F_sttc, M_sttc])

# load ECG interval data
waves_F = np.load(data_processed + "waves_F.npy")
waves_M = np.load(data_processed + "waves_M.npy")

waves_F_sttc = np.load(data_processed + PREFIX + "waves_F.npy")
waves_M_sttc = np.load(data_processed + PREFIX + "waves_M.npy")
#%%
norm = np.concatenate((waves_F[:, :, 0:5], waves_M[:, :, 0:5]), axis=0)

sttc = np.concatenate((waves_F_sttc[:, :, 0:5], waves_M_sttc[:, :, 0:5]), axis=0)

# load ECG signals
# ecgM = load_raw_data(M, SAMPLING_RATE, data_raw)
# ecgF = load_raw_data(F, SAMPLING_RATE, data_raw)

ecgNORM = load_raw_data(PDnorm, SAMPLING_RATE, data_raw)

# ecgM_sttc = load_raw_data(M_sttc, SAMPLING_RATE, data_raw)
# ecgF_sttc = load_raw_data(F_sttc, SAMPLING_RATE, data_raw)
#       colpa di Gabry!

ecgSTTC = load_raw_data(PDsttc, SAMPLING_RATE, data_raw)

#%%
# MALE = Norm
waves_M = norm
M = PDnorm
ecgM = ecgNORM

# FEMALE = STTc
waves_F = sttc
F = PDsttc
ecgF = ecgSTTC

# %% Bootstrap patients
PATIENT_F = random.choices(range(waves_F.shape[0]), k=5)
PATIENT_M = random.choices(range(waves_M.shape[0]), k=5)

bins = [7, 12, 19, 24]
PATIENT_F_0_7 = [i for i, v in enumerate(F.Hour) if (v < 7)]
PATIENT_M_0_7 = [i for i, v in enumerate(M.Hour) if (v < 7)]
PATIENT_F_7_12 = [i for i, v in enumerate(F.Hour) if (v >= 7) & (v <= 12)]
PATIENT_M_7_12 = [i for i, v in enumerate(M.Hour) if (v >= 7) & (v <= 12)]
PATIENT_F_12_19 = [i for i, v in enumerate(F.Hour) if (v > 12) & (v <= 19)]
PATIENT_M_12_19 = [i for i, v in enumerate(M.Hour) if (v > 12) & (v <= 19)]
PATIENT_F_19_24 = [i for i, v in enumerate(F.Hour) if (v > 19) & (v <= 24)]
PATIENT_M_19_24 = [i for i, v in enumerate(M.Hour) if (v > 19) & (v <= 24)]

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
    knots, t_points = compute_knots2(peakList, _n_cheb)

    # create skfda 's FDataGrid data
    heartbeatRaw = skfda.FDataGrid(
        ECG[start:stop, LEAD] - np.mean(ECG[start:stop, LEAD]),
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
    new_t = np.linspace(0, t_points[-1], y.shape[0])
    y_new = f(new_t).T

    # plot raw + smoothed ECG
    if show_figures:
        plt.figure()
        plt.plot(t_points, heartbeatRaw.data_matrix[0, :, 0], label="ECG raw")
        # plt.plot(t_points, heartbeatSmoothed.data_matrix[0, :, 0], label="ECG smoothed")
        plt.plot(new_t, y_new)
        plt.legend()
        plt.show()

    return y


# %%
def getLandmarks(waveList, patientList, sampling_rate=SAMPLING_RATE):
    peak = waveList[patientList, :, 0]
    sub = peak[:, 0]

    peak = [(peak[:, i] - peak[:, 0]) / sampling_rate for i in range(peak.shape[1])]
    peak = np.transpose(np.array(peak))
    peak = peak[:, 2:8]

    return np.array(peak)


# %%
smoothed_F_0_7 = [
    smoothedECG(ecgF[p], waves_F[p], show_figures=False) for p in PATIENT_F_0_7
]
smoothed_F_7_12 = [
    smoothedECG(ecgF[p], waves_F[p], show_figures=False) for p in PATIENT_F_7_12
]
smoothed_F_12_19 = [
    smoothedECG(ecgF[p], waves_F[p], show_figures=False) for p in PATIENT_F_12_19
]
smoothed_F_19_24 = [
    smoothedECG(ecgF[p], waves_F[p], show_figures=False) for p in PATIENT_F_19_24
]


smoothed_M_0_7 = [
    smoothedECG(ecgM[p], waves_M[p], show_figures=False) for p in PATIENT_M_0_7
]
smoothed_M_7_12 = [
    smoothedECG(ecgM[p], waves_M[p], show_figures=False) for p in PATIENT_M_7_12
]
smoothed_M_12_19 = [
    smoothedECG(ecgM[p], waves_M[p], show_figures=False) for p in PATIENT_M_12_19
]
smoothed_M_19_24 = [
    smoothedECG(ecgM[p], waves_M[p], show_figures=False) for p in PATIENT_M_19_24
]

PATIENT_M_7_12 = [p for p in range(len(M))]
PATIENT_F_7_12 = [p for p in range(len(F))]

smoothed_F_7_12 = [
    smoothedECG(ecgF[p], waves_F[p], show_figures=False) for p in range(len(F))
]

smoothed_M_7_12 = [
    smoothedECG(ecgM[p], waves_M[p], show_figures=False) for p in range(len(M))
]


# max samples are 484
maxSamples_F_0_7 = max(map(len, smoothed_F_0_7))
maxSamples_F_7_12 = max(map(len, smoothed_F_7_12))
maxSamples_F_12_19 = max(map(len, smoothed_F_12_19))
maxSamples_F_19_24 = max(map(len, smoothed_F_19_24))

maxSamples_M_0_7 = max(map(len, smoothed_M_0_7))
maxSamples_M_7_12 = max(map(len, smoothed_M_7_12))
maxSamples_M_12_19 = max(map(len, smoothed_M_12_19))
maxSamples_M_19_24 = max(map(len, smoothed_M_19_24))

maxSamples_0_7 = max([maxSamples_F_0_7, maxSamples_M_0_7])
maxSamples_7_12 = max([maxSamples_F_7_12, maxSamples_M_7_12])
maxSamples_12_19 = max([maxSamples_F_12_19, maxSamples_M_12_19])
maxSamples_19_24 = max([maxSamples_F_19_24, maxSamples_M_19_24])

maxSamples = max([maxSamples_0_7, maxSamples_7_12, maxSamples_12_19, maxSamples_19_24])
t = np.linspace(0, maxSamples * (1 / SAMPLING_RATE), maxSamples)

#%%


def padSamples(sample, length):
    for i in range(len(sample)):
        x = sample[i]
        xc = x[-1] * np.ones(length - x.size)
        sample[i] = np.concatenate((x, xc))
        sample[i] = skfda.representation.grid.FDataGrid(sample[i])
    return sample


# %% pad smoothed signals to maximum length in dataset
smoothed_F_0_7 = padSamples(smoothed_F_0_7, maxSamples)
smoothed_F_7_12 = padSamples(smoothed_F_7_12, maxSamples)
smoothed_F_12_19 = padSamples(smoothed_F_12_19, maxSamples)
smoothed_F_19_24 = padSamples(smoothed_F_19_24, maxSamples)

smoothed_M_0_7 = padSamples(smoothed_M_0_7, maxSamples)
smoothed_M_7_12 = padSamples(smoothed_M_7_12, maxSamples)
smoothed_M_12_19 = padSamples(smoothed_M_12_19, maxSamples)
smoothed_M_19_24 = padSamples(smoothed_M_19_24, maxSamples)

#%%
def concatenateFDataGrid(a, b):
    if a:
        fd = a[0]
        for fi in range(len(a)):
            if fi > 0:
                fd = fd.concatenate(a[fi])
    if b:
        for mi in range(len(b)):
            fd = fd.concatenate(b[mi])
    return fd


# %%
# Cambiando land e smooth cambiamo orario

# smoothed_F_7_12 = smoothed_F_12_19
# smoothed_M_7_12 = smoothed_M_12_19

# PATIENT_F_7_12 = PATIENT_F_12_19
# PATIENT_M_7_12 = PATIENT_M_12_19

smoothed_F_7_12 = smoothed_F_7_12
smoothed_M_7_12 = smoothed_M_7_12

PATIENT_F_7_12 = PATIENT_F_7_12
PATIENT_M_7_12 = PATIENT_M_7_12


# %% concatenate FDataGrid of the same cluster

fd_7_12 = concatenateFDataGrid(smoothed_F_7_12, smoothed_M_7_12)

# %% Alignment LANDMARK FEATURE

land_F_7_12 = getLandmarks(waves_F, PATIENT_F_7_12, sampling_rate=SAMPLING_RATE)
land_M_7_12 = getLandmarks(waves_M, PATIENT_M_7_12, sampling_rate=SAMPLING_RATE)
land_7_12 = np.concatenate([land_F_7_12, land_M_7_12])

#%%
warping_7_12 = skfda.preprocessing.registration.landmark_registration_warping(
    fd_7_12, land_7_12, location=np.mean(land_M_7_12, axis=0)
)

fig = warping_7_12.plot()

for v in np.mean(land_7_12, axis=0):
    plt.axvline(x=v, color="k", lw=0.5)

plt.xticks(np.mean(land_7_12, axis=0), ["P", "Q", "R", "S", "T", "TOff"])

#%%
tw = np.array(warping_7_12.grid_points)
xf = warping_7_12.data_matrix[: len(land_F_7_12), :, 0]
xm = warping_7_12.data_matrix[len(land_F_7_12) :, :, 0]

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.set_title("Warping function: STTC Group")
ax2.set_title("Warping function: CTR Group")

for i in range(len(xf)):
    ax1.plot(tw[0, :], xf[i, :], color="gray", alpha=0.3)
bis = np.linspace(0, 1, 500)
ax1.plot(bis, bis, "k--", alpha=0.7)
ax1.plot(tw[0, :], np.mean(xf, axis=0), color="k")
for v in np.mean(land_7_12, axis=0):
    ax1.axvline(x=v, color="k", lw=0.5)

for i in range(len(xm)):
    ax2.plot(tw[0, :], xm[i, :], color="gray", alpha=0.3)
bis = np.linspace(0, 1, 500)
ax2.plot(bis, bis, "k--", alpha=0.7)
ax2.plot(tw[0, :], np.mean(xm, axis=0), color="k")
for v in np.mean(land_7_12, axis=0):
    ax2.axvline(x=v, color="k", lw=0.5)

# %%
fd_registered_7_12 = fd_7_12.compose(warping_7_12)
fig = fd_registered_7_12.plot()
plt.title("Subjects 07-12")

# %%
fd_F_7_12 = concatenateFDataGrid(smoothed_F_7_12, None)
warping_F_7_12 = skfda.preprocessing.registration.landmark_registration_warping(
    fd_F_7_12, land_F_7_12, location=np.mean(land_F_7_12, axis=0)
)

fig = warping_F_7_12.plot()

for v in np.mean(land_F_7_12, axis=0):
    plt.axvline(x=v, color="k", lw=0.5)

plt.xticks(np.mean(land_F_7_12, axis=0), ["P", "Q", "R", "S", "T", "TOff"])
fd_registered_F_7_12 = fd_F_7_12.compose(warping_F_7_12)

# %%
fd_M_7_12 = concatenateFDataGrid(smoothed_M_7_12, None)
warping_M_7_12 = skfda.preprocessing.registration.landmark_registration_warping(
    fd_M_7_12, land_M_7_12, location=np.mean(land_M_7_12, axis=0)
)

fig = warping_M_7_12.plot()

for v in np.mean(land_M_7_12, axis=0):
    plt.axvline(x=v, color="k", lw=0.5)

plt.xticks(np.mean(land_M_7_12, axis=0), ["P", "Q", "R", "S", "T", "TOff"])
fd_registered_M_7_12 = fd_M_7_12.compose(warping_M_7_12)

#%%

fig, [ax1, ax2] = plt.subplots(2, 1)
ax1.set_title("ST-T Change Subject")
fd_F_7_12.plot(ax1)
ax2.set_title("Control Subject")
fd_M_7_12.plot(ax2)

# %%
fig = plt.figure()
for i in range(warping_7_12.n_samples):
    if i < len(land_F_7_12):
        plt.plot(
            warping_7_12.data_matrix[i, :, 0],
            "r",
            alpha=0.7,
            linewidth=0.2,
            label="Female",
        )
    else:
        plt.plot(
            warping_7_12.data_matrix[i, :, 0],
            "b",
            alpha=0.7,
            linewidth=0.2,
            label="Male",
        )
for v in np.mean(land_7_12, axis=0):
    plt.axvline(x=v * SAMPLING_RATE, color="k", lw=0.5)

plt.xticks(
    np.mean(land_7_12, axis=0) * SAMPLING_RATE, ["P", "Q", "R", "S", "T", "TOff"]
)
# plt.legend()

# %%
mean_F_7_12 = stats.trim_mean(
    warping_7_12.data_matrix[: len(land_F_7_12), :, 0], 0.05, axis=0
)
mean_M_7_12 = stats.trim_mean(
    warping_7_12.data_matrix[len(land_F_7_12) :, :, 0], 0.05, axis=0
)

fig1 = plt.figure()
plt.plot((mean_F_7_12 - mean_M_7_12) * 100, "r")
for v in np.mean(land_7_12, axis=0):
    plt.axvline(x=v * SAMPLING_RATE, color="k", lw=0.5)

plt.xticks(
    np.mean(land_7_12, axis=0) * SAMPLING_RATE, ["P", "Q", "R", "S", "T", "TOff"]
)

# %%
t_mean_F_7_12 = stats.trim_mean(fd_registered_F_7_12.data_matrix[:, :, 0], 0.05, axis=0)
t_mean_M_7_12 = stats.trim_mean(fd_registered_M_7_12.data_matrix[:, :, 0], 0.05, axis=0)
t_median_F_7_12 = np.median(fd_registered_F_7_12.data_matrix[:, :, 0], axis=0)
t_median_M_7_12 = np.median(fd_registered_M_7_12.data_matrix[:, :, 0], axis=0)

fig1 = plt.figure()
plt.title("Trimmed Mean (5%) vs. Median")
plt.plot(t_mean_F_7_12, "r", alpha=0.5, label="STTC Tr.Mean")
plt.plot(t_mean_M_7_12, "b", alpha=0.5, label="CTR Tr.Mean")
plt.plot(t_median_F_7_12, "r--", alpha=0.5, label="STTC Median")
plt.plot(t_median_M_7_12, "b--", alpha=0.5, label="CTR Median")

plt.legend()

for v in np.mean(land_F_7_12, axis=0):
    plt.axvline(x=v * SAMPLING_RATE, color="orange", lw=0.5)

plt.xticks(
    np.mean(land_F_7_12, axis=0) * SAMPLING_RATE, ["P", "Q", "R", "S", "T", "TOff"]
)
for v in np.mean(land_M_7_12, axis=0):
    plt.axvline(x=v * SAMPLING_RATE, color="green", lw=0.5)

plt.xticks(
    np.mean(land_M_7_12, axis=0) * SAMPLING_RATE, ["P", "Q", "R", "S", "T", "TOff"]
)

# %%
fdBoxplot_7_12 = Boxplot(fd_registered_7_12)
fdBoxplot_7_12.plot()
plt.title("Subjects 07-12")
#%%
fdBoxplot_F_7_12 = Boxplot(fd_registered_F_7_12)
fdBoxplot_F_7_12.plot()
plt.title("Female Subjects")

fdBoxplot_M_7_12 = Boxplot(fd_registered_M_7_12)
fdBoxplot_M_7_12.plot()
plt.title("Male Subjects")

#%%
sm.graphics.fboxplot(warping_M_7_12.data_matrix[:, :, 0], wfactor=2.5)
plt.title("Male Subjects")
sm.graphics.fboxplot(warping_F_7_12.data_matrix[:, :, 0], wfactor=2.5)
plt.title("Female Subjects")

sm.graphics.fboxplot(fd_registered_M_7_12.data_matrix[:, :, 0], wfactor=2.5)
plt.title("Male Subjects")
sm.graphics.fboxplot(fd_registered_F_7_12.data_matrix[:, :, 0], wfactor=2.5)
plt.title("Female Subjects")

#%% FUNCTIONAL PCA

from skfda.exploratory.visualization import plot_fpca_perturbation_graphs

n = 4

fpca = FPCA(n_components=n)
fpca.fit(fd_M_7_12)

fpca.components_.plot()
plt.title("STTC: Principal Components")
plt.legend(["PC1", "PC2", "PC3", "PC4"])

pc, axs = plt.subplots(4)
for i in range(4):
    fpca.components_[i].plot(axes=axs[i])
    axs[i].set_title("PC" + str(i + 1))


evr_M_7_12 = fpca.explained_variance_ratio_ * 100

fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.bar(range(n), evr_M_7_12, alpha=0.6, label="CTR")

print("Norm:  " + str(np.sum(evr_M_7_12[:3])))

pert = plt.figure(figsize=(6, 2 * 4))
pert.suptitle("Perturbation Plot: Control", fontsize=16)

plot_fpca_perturbation_graphs(fd_M_7_12.mean(), fpca.components_, 30, fig=pert)

fpca = FPCA(n_components=n)
fpca.fit(fd_F_7_12)

fpca.components_.plot()
plt.title("STTC: Principal Components")
plt.legend(["PC1", "PC2", "PC3", "PC4"])

for i in range(4):
    fpca.components_[i].plot(axes=axs[i])
    axs[i].set_title("PC" + str(i + 1))
    axs[i].legend(["CTR", "STTC"])


evr_F_7_12 = fpca.explained_variance_ratio_ * 100

ax1.bar(range(n), evr_F_7_12, alpha=0.6, label="STTC")
ax1.set_title("FPCA (" + str(n) + ")")
ax1.legend()

print("STTC:  " + str(np.sum(evr_F_7_12[:3])))


ax2.bar(range(n), np.cumsum(evr_M_7_12), alpha=0.6, label="CTR")
ax2.bar(range(n), np.cumsum(evr_F_7_12), alpha=0.6, label="STTC")
ax2.set_title("Cumulative Variance (" + str(n) + ")")
ax2.legend()

pert = plt.figure(figsize=(6, 2 * 4))
pert.suptitle("Perturbation Plot: ST-T Change", fontsize=16)
plot_fpca_perturbation_graphs(fd_F_7_12.mean(), fpca.components_, 30, fig=pert)

#%% DERIVATIVES
# Finite differences: forward approximation
# %%
x = fd_registered_M_7_12.grid_points[0]
y = fd_registered_M_7_12.data_matrix
y = y.reshape(y.shape[0], y.shape[1])

dydx_M = centeredFiniteDistance2D(x, y)

plt.figure()
plt.plot(dydx_M.T)
plt.title("Male Subjects ∂y/∂x")

x = fd_registered_F_7_12.grid_points[0]
y = fd_registered_F_7_12.data_matrix
y = y.reshape(y.shape[0], y.shape[1])

dydx_F = centeredFiniteDistance2D(x, y)

plt.figure()
plt.plot(dydx_F.T)
plt.title("Female Subjects ∂y/∂x")

#%%
df_t_mean_F_7_12 = stats.trim_mean(dydx_F, 0.05, axis=0)
df_t_mean_M_7_12 = stats.trim_mean(dydx_M, 0.05, axis=0)
df_t_median_F_7_12 = np.median(dydx_F, axis=0)
df_t_median_M_7_12 = np.median(dydx_M, axis=0)

plt.figure()
plt.title("∂y/∂x Median vs. Trimmed Mean")
plt.plot(df_t_mean_F_7_12, "r", alpha=0.5, label="STTC Tr.Mean")
plt.plot(df_t_mean_M_7_12, "b", alpha=0.5, label="CTR Tr.Mean")
plt.plot(df_t_median_F_7_12, "r--", alpha=0.5, label="STTC Median")
plt.plot(df_t_median_M_7_12, "b--", alpha=0.5, label="CTR Median")
plt.legend()

# %%
dydx2_M = centeredFiniteDistance2D(x, dydx_M)

plt.figure()
plt.plot(dydx2_M.T)
plt.title("Male Subjects $∂^2y/∂x^2$")

dydx2_F = centeredFiniteDistance2D(x, dydx_F)

plt.figure()
plt.plot(dydx2_F.T)
plt.title("Female Subjects $∂^2y/∂x^2$")

#%%
df2_t_mean_F_7_12 = stats.trim_mean(dydx2_F, 0.05, axis=0)
df2_t_mean_M_7_12 = stats.trim_mean(dydx2_M, 0.05, axis=0)
df2_t_median_F_7_12 = np.median(dydx2_F, axis=0)
df2_t_median_M_7_12 = np.median(dydx2_M, axis=0)

plt.figure()
plt.title("$∂^2y/∂x^2$ Median vs. Trimmed Mean")
plt.plot(df2_t_mean_F_7_12, "r", alpha=0.5, label="STTC Tr.Mean")
plt.plot(df2_t_mean_M_7_12, "b", alpha=0.5, label="CTR Tr.Mean")
plt.plot(df2_t_median_F_7_12, "r--", alpha=0.5, label="STTC Median")
plt.plot(df2_t_median_M_7_12, "b--", alpha=0.5, label="CTR Median")
plt.legend()

#%%
sm.graphics.fboxplot(dydx_M, wfactor=2.5)
plt.title("Male Subjects")
sm.graphics.fboxplot(dydx_F, wfactor=2.5)
plt.title("Female Subjects")

#%%
dydx_M = skfda.FDataGrid(dydx_M)
fdBoxplot_M_7_12 = Boxplot(dydx_M)
# fdBoxplot_M_7_12.show_full_outliers = True
fdBoxplot_M_7_12.plot()
plt.title("Male Subjects")

dydx_F = skfda.FDataGrid(dydx_F)
fdBoxplot_F_7_12 = Boxplot(dydx_F)
# fdBoxplot_M_7_12.show_full_outliers = True
fdBoxplot_F_7_12.plot()
plt.title("Female Subjects")

# %%
plt.close("all")

#%%
f1Mr = []
f2Mr = []
f1Fr = []
f2Fr = []

for i in fd_registered_F_7_12:
    f = make_interp_spline(i.grid_points[0], i.data_matrix[0, :, 0])
    new_t = np.linspace(0, 1, 484)
    f1Fr.append(f.derivative(1)(new_t).T)
    f2Fr.append(f.derivative(2)(new_t).T)

for i in fd_registered_M_7_12:
    f = make_interp_spline(i.grid_points[0], i.data_matrix[0, :, 0])
    new_t = np.linspace(0, 1, 484)
    f1Mr.append(f.derivative(1)(new_t).T)
    f2Mr.append(f.derivative(2)(new_t).T)

f1Mr = np.array(f1Mr)
f2Mr = np.array(f2Mr)
f1Fr = np.array(f1Fr)
f2Fr = np.array(f2Fr)

df_t_mean_F_7_12 = stats.trim_mean(f1Fr, 0.05, axis=0)
df_t_mean_M_7_12 = stats.trim_mean(f1Mr, 0.05, axis=0)

plt.figure()
plt.title("First Derivative")
plt.plot(df_t_mean_F_7_12, "r", alpha=0.5, label="F Tr.Mean")
plt.plot(df_t_mean_M_7_12, "b", alpha=0.5, label="M Tr.Mean")
plt.legend()

df2_t_mean_F_7_12 = stats.trim_mean(f2Fr, 0.05, axis=0)
df2_t_mean_M_7_12 = stats.trim_mean(f2Mr, 0.05, axis=0)

plt.figure()
plt.title("Second Derivative")
plt.plot(df2_t_mean_F_7_12, "r", alpha=0.5, label="F Tr.Mean")
plt.plot(df2_t_mean_M_7_12, "b", alpha=0.5, label="M Tr.Mean")
plt.legend()

#%%

f1M = []
f2M = []
f1F = []
f2F = []

for i in fd_F_7_12:
    f = make_interp_spline(i.grid_points[0], i.data_matrix[0, :, 0])
    new_t = np.linspace(0, 1, 484)
    f1F.append(f.derivative(1)(new_t).T)
    f2F.append(f.derivative(2)(new_t).T)

for i in fd_M_7_12:
    f = make_interp_spline(i.grid_points[0], i.data_matrix[0, :, 0])
    new_t = np.linspace(0, 1, 484)
    f1M.append(f.derivative(1)(new_t).T)
    f2M.append(f.derivative(2)(new_t).T)

f1M = np.array(f1M)
f2M = np.array(f2M)
f1F = np.array(f1F)
f2F = np.array(f2F)

fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.set_title("1st Derivative before warping")
ax2.set_title("2nd Derivative before warping")

df_t_mean_F_7_12 = stats.trim_mean(f1F, 0.05, axis=0)
df_t_mean_M_7_12 = stats.trim_mean(f1M, 0.05, axis=0)

ax1.plot(new_t, df_t_mean_F_7_12, "r", alpha=0.5, label="STTC Tr.Mean")
ax1.plot(new_t, df_t_mean_M_7_12, "b", alpha=0.5, label="CTR Tr.Mean")
ax1.legend()

df2_t_mean_F_7_12 = stats.trim_mean(f2F, 0.05, axis=0)
df2_t_mean_M_7_12 = stats.trim_mean(f2M, 0.05, axis=0)

ax2.plot(new_t, df2_t_mean_F_7_12, "r", alpha=0.5, label="STTC Tr.Mean")
ax2.plot(new_t, df2_t_mean_M_7_12, "b", alpha=0.5, label="CTR Tr.Mean")
ax2.legend()

#%%

f1Mw = []
f2Mw = []
f1Fw = []
f2Fw = []

for i in warping_F_7_12:
    f = make_interp_spline(i.grid_points[0], i.data_matrix[0, :, 0])
    new_t = np.linspace(0, 1, 484)
    f1Fw.append(f.derivative(1)(new_t).T)
    f2Fw.append(f.derivative(2)(new_t).T)

for i in warping_M_7_12:
    f = make_interp_spline(i.grid_points[0], i.data_matrix[0, :, 0])
    new_t = np.linspace(0, 1, 484)
    f1Mw.append(f.derivative(1)(new_t).T)
    f2Mw.append(f.derivative(2)(new_t).T)

f1Mw = np.array(f1Mw)
f2Mw = np.array(f2Mw)
f1Fw = np.array(f1Fw)
f2Fw = np.array(f2Fw)

df_t_mean_F_7_12 = stats.trim_mean(f1Fw, 0.05, axis=0)
df_t_mean_M_7_12 = stats.trim_mean(f1Mw, 0.05, axis=0)

fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.set_title("1st Derivative Warping")
ax2.set_title("2nd Derivative Warping")

ax1.plot(new_t, df_t_mean_F_7_12, "r", alpha=0.5, label="STTC Tr.Mean")
ax1.plot(new_t, df_t_mean_M_7_12, "b", alpha=0.5, label="CTR Tr.Mean")
ax1.legend()


df2_t_mean_F_7_12 = stats.trim_mean(f2Fw, 0.05, axis=0)
df2_t_mean_M_7_12 = stats.trim_mean(f2Mw, 0.05, axis=0)

ax2.plot(new_t, df2_t_mean_F_7_12, "r", alpha=0.5, label="STTC Tr.Mean")
ax2.plot(new_t, df2_t_mean_M_7_12, "b", alpha=0.5, label="CTR Tr.Mean")
ax2.legend()

#%% DEPTH MEASURES

depth = skfda.exploratory.depth.ModifiedBandDepth()

depth_F_7_12 = depth(fd_F_7_12)
index = np.where(depth_F_7_12 == np.amax(depth_F_7_12))[0][0]
print("FEMALE")
print(
    "Maximum Depth Function: " + str(index) + "\nValue: " + str(np.amax(depth_F_7_12))
)

fd_F_7_12[index].plot()

c = [(v, i) for i, v in enumerate(depth_F_7_12)]
c.sort(key=lambda tup: tup[0])

fig = plt.figure()
ax = fig.gca(projection="3d")
pal = sns.color_palette(palette="dark:salmon_r", n_colors=len(c))


def cc(arg):
    return mcolors.to_rgba(arg, alpha=0.6)


xs = np.linspace(0, 1, len(fd_F_7_12.data_matrix[0, :, 0]))
verts = []
zs = np.arange(len(c))
for tup in c:
    ys = fd_F_7_12.data_matrix[tup[1], :, 0]
    # ys[0], ys[-1] = 0, 0
    verts.append(list(zip(xs, ys)))

poly = LineCollection(verts, colors=pal)
poly.set_alpha(0.7)
ax.add_collection3d(poly, zs=zs, zdir="y")

ax.set_xlabel("X")
ax.set_xlim3d(0, 1)
ax.set_ylabel("Y")
ax.set_ylim3d(0, len(c))
ax.set_zlabel("Z")
ax.set_zlim3d(np.amin(fd_F_7_12.data_matrix), np.amax(fd_F_7_12.data_matrix))

plt.show()

#%%

# Make data.
X = xs
Y = zs
X, Y = np.meshgrid(X, Y)

nn = np.array([x[1] for x in c])

fig, (ax1, ax2) = plt.subplots(2)

cs = ax1.contourf(X, Y, fd_F_7_12.data_matrix[nn, :, 0], cmap=cm.coolwarm)
plt.colorbar(cs, ax=ax1)
ax1.set_title("Smoothed Curves STTC ordered by Depth")

depth_M_7_12 = depth(fd_M_7_12)
index = np.where(depth_M_7_12 == np.amax(depth_M_7_12))[0][0]
print("\nMALE")
print(
    "Maximum Depth Function: " + str(index) + "\nValue: " + str(np.amax(depth_M_7_12))
)

fd_M_7_12[index].plot()

c = [(v, i) for i, v in enumerate(depth_M_7_12)]
c.sort(key=lambda tup: tup[0])

xs = np.linspace(0, 1, len(fd_M_7_12.data_matrix[0, :, 0]))
zs = np.arange(len(c))

# Make data.
X = xs
Y = zs
X, Y = np.meshgrid(X, Y)

nn = np.array([x[1] for x in c])

cs = ax2.contourf(X, Y, fd_M_7_12.data_matrix[nn, :, 0], cmap=cm.coolwarm)
plt.colorbar(cs, ax=ax2)
ax2.set_title("Smoothed Curves CTR ordered by Depth")


#%%

v_n, p_val, dist = skfda.inference.anova.oneway_anova(
    fd_F_7_12, fd_M_7_12, n_reps=400, return_dist=True, equal_var=False
)
print("NO REGISTRATION")
print("Statistic: ", v_n)
print("p-value: ", p_val)
# print('Distribution: ', dist)

v_n, p_val, dist = skfda.inference.anova.oneway_anova(
    fd_registered_F_7_12,
    fd_registered_M_7_12,
    n_reps=100,
    return_dist=True,
    equal_var=False,
)
print("REGISTRATION")
print("Statistic: ", v_n)
print("p-value: ", p_val)
# print('Distribution: ', dist)

#%%
depth_M_7_12 = depth(fd_registered_M_7_12)
indexM = np.where(depth_M_7_12 == np.amax(depth_M_7_12))[0][0]

depth_F_7_12 = depth(fd_registered_F_7_12)
indexF = np.where(depth_F_7_12 == np.amax(depth_F_7_12))[0][0]


amplitude = skfda.misc.metrics.amplitude_distance(
    fd_registered_M_7_12[indexM], fd_registered_F_7_12[indexF]
)
print("AMPLITUDE:" + str(amplitude))
#%%
# from skfda.preprocessing.registration import ElasticRegistration
# from skfda.preprocessing.registration.elastic import elastic_mean

# elastic_mean(fd_M_7_12[0:3]).plot()

# elastic_registration = ElasticRegistration(template=elastic_mean(fd_M_7_12[0:3]))
# fd_align = elastic_registration.fit_transform(fd_F_7_12)

# fd_align.plot()

#%%

#%%

plt.close("all")

#%% STTC individual depth compared to CTR depths
M_dm = fd_M_7_12.data_matrix
F_dm = fd_F_7_12.data_matrix

depths_sttc = []

# run new depth method for each 1 STTC individual and CTR group
for i, f_data in enumerate(F_dm):
    # assemble new FDataGrid
    M_plus_dm = np.concatenate((M_dm, [f_data]))
    M_plus_fd = skfda.FDataGrid(M_plus_dm, fd_M_7_12.grid_points[0])

    # compute depths
    depth_M_plus = depth(M_plus_fd)
    i_depth = depth_M_plus[-1]
    depths_sttc.append(i_depth)

    # print result of i-th individual of the STTC group
    print(str(i + 1) + "-depth sttc: " + str(i_depth))

    # I want to compare i-th value os STTC with the min, max and median os CTR group
    d_min = min(depth_M_plus)
    d_mean = mean(depth_M_plus)
    d_median = median(depth_M_plus)
    d_max = max(depth_M_plus)

    if i_depth == d_min:
        print("   MIN")
    elif i_depth == d_max:
        print("   MAX")
    else:
        print("   MEAN DIFF: " + str(d_mean - i_depth))
        print("   MED. DIFF: " + str(d_median - i_depth))
        
#%%
plt.figure()
plt.boxplot((depths_sttc,depth(fd_M_7_12)))
plt.scatter(np.ones(len(depths_sttc)),depths_sttc)
plt.scatter(2*np.ones(len(fd_M_7_12)),depth(fd_M_7_12))

plt.figure()
plt.hist(depth(fd_M_7_12),density=True,bins=15)
plt.hist(depths_sttc,density=True,bins=15)

# %%

template=stats.trim_mean(fd_registered_M_7_12.data_matrix[:,:,0], 0.05, axis=0)

CTRsub=fd_registered_M_7_12.data_matrix[:,:,0]
STTCsub=fd_registered_F_7_12.data_matrix[:,:,0]

plt.figure()
plt.boxplot((depth(fd_registered_F_7_12),depth(fd_registered_M_7_12)))

comp=skfda.FDataGrid(np.concatenate((CTRsub,STTCsub)))
D=depth(comp)

#%%
S=D[-29:]
C=D[:-29]

plt.figure()
plt.boxplot((S,C))
# %%


CTRsub=fd_M_7_12.data_matrix[:,:,0]
STTCsub=fd_F_7_12.data_matrix[:,:,0]

plt.figure()
plt.boxplot((depth(fd_F_7_12),depth(fd_M_7_12)))

comp=skfda.FDataGrid(np.concatenate((CTRsub,STTCsub)))
D=depth(comp)

#%%
S=D[-29:]
C=D[:-29]

plt.figure()
plt.boxplot((S,C))

#%%
plt.close("all")