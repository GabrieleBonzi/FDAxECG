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
    knots, t_points = compute_knots2(peakList, _n_cheb)

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

    return y - np.mean(y)


# %%
def getLandmarks(waveList, patientList, sampling_rate=SAMPLING_RATE):
    peak = waveList[patientList, :, 0]
    sub = peak[:, 0]

    peak = [(peak[:, i] - peak[:, 0]) / sampling_rate for i in range(peak.shape[1])]
    peak = np.transpose(np.array(peak))
    peak = peak[:, 2:8]

    return np.array(peak)


# %%
smoothedF = [
    smoothedECG(ecgF[p], waves_F[p], show_figures=False) for p in PATIENT_F_7_12
]

# max samples are 484
maxSamples = max(map(len, smoothedF))
t = np.linspace(0, maxSamples * (1 / SAMPLING_RATE), maxSamples)

#%%
for i in range(len(smoothedF)):
    x = smoothedF[i]
    xc = np.zeros(maxSamples - x.size)
    smoothedF[i] = np.concatenate((x, xc))
    smoothedF[i] = skfda.representation.grid.FDataGrid(smoothedF[i])


#%%
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
    # plt.legend()


# %%
fd_registered_F = fd_F.compose(warpingF)
fig = fd_registered_F.plot()
plt.title("Female Subjects")

# %% Males
smoothedM = [
    smoothedECG(ecgM[p], waves_M[p], show_figures=False) for p in PATIENT_M_7_12
]

# max samples are 484
maxSamples = max(map(len, smoothedM))
t = np.linspace(0, maxSamples * (1 / 500), maxSamples)

#%%
for i in range(len(smoothedM)):
    x = smoothedM[i]
    xc = np.zeros(maxSamples - x.size)
    smoothedM[i] = np.concatenate((x, xc))
    smoothedM[i] = skfda.representation.grid.FDataGrid(smoothedM[i])

# %%
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
plt.title("Male Subjects")

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

#%%
from skfda.exploratory.visualization import Boxplot

fdBoxplot_M = Boxplot(fd_registered_M)
# fdBoxplot_M.show_full_outliers = True
fdBoxplot_M.plot()
plt.title("Male Subjects")

fdBoxplot_F = Boxplot(fd_registered_F)
# fdBoxplot_M.show_full_outliers = True
fdBoxplot_F.plot()
plt.title("Female Subjects")

#%%

from skfda.exploratory.visualization import Boxplot

fdBoxplot_M = Boxplot(warpingM)
# fdBoxplot_M.show_full_outliers = True
fdBoxplot_M.plot()
plt.title("Male Subjects")

fdBoxplot_F = Boxplot(warpingF)
# fdBoxplot_M.show_full_outliers = True
fdBoxplot_F.plot()
plt.title("Female Subjects")

#%%
import statsmodels.api as sm

sm.graphics.fboxplot(warpingM.data_matrix[:, :, 0], wfactor=2.5)
plt.title("Male Subjects")
sm.graphics.fboxplot(warpingF.data_matrix[:, :, 0], wfactor=2.5)
plt.title("Female Subjects")

sm.graphics.fboxplot(fd_registered_M.data_matrix[:, :, 0], wfactor=2.5)
plt.title("Male Subjects")
sm.graphics.fboxplot(fd_registered_F.data_matrix[:, :, 0], wfactor=2.5)
plt.title("Female Subjects")

# #%% A COSA SERVE?

# from skfda.exploratory.visualization import MagnitudeShapePlot

# msplot = MagnitudeShapePlot(fdatagrid=fd_registered_M)

# color = 0.3
# outliercol = 0.7

# msplot.color = color
# msplot.outliercol = outliercol
# msplot.plot()
# plt.title("Male Subjects")

# msplot = MagnitudeShapePlot(fdatagrid=fd_registered_F)
# msplot.color = color
# msplot.outliercol = outliercol
# msplot.plot()
# plt.title("Female Subjects")

#%%

from skfda.preprocessing.dim_reduction.projection import FPCA

n = 10

fpca = FPCA(n_components=n)
fpca.fit(fd_registered_M)

evr_M = fpca.explained_variance_ratio_ * 100

plt.bar(range(n), evr_M, alpha=0.6, label="Male")

print("MALE:  " + str(np.sum(evr_M[:5])))

fpca = FPCA(n_components=n)
fpca.fit(fd_registered_F)

evr_F = fpca.explained_variance_ratio_ * 100

plt.bar(range(n), evr_F, alpha=0.6, label="Female")
plt.title("FPCA (20) - Morning")
plt.legend()

print("Female:  " + str(np.sum(evr_F[:5])))

plt.figure()
plt.bar(range(n), np.cumsum(evr_M), alpha=0.6, label="Male")
plt.bar(range(n), np.cumsum(evr_F), alpha=0.6, label="Female")
plt.title("Cumulative Variance (20) - Morning")
plt.legend()

#%% DERIVATIVES
# Finite differences: forward approximation
# %%
x = fd_registered_M.grid_points[0]
y = fd_registered_M.data_matrix
y = y.reshape(y.shape[0], y.shape[1])

dydx_M = centeredFiniteDistance2D(x, y)

plt.figure()
plt.plot(dydx_M.T)
plt.title("Male Subjects ∂y/∂x")

x = fd_registered_F.grid_points[0]
y = fd_registered_F.data_matrix
y = y.reshape(y.shape[0], y.shape[1])

dydx_F = centeredFiniteDistance2D(x, y)

plt.figure()
plt.plot(dydx_F.T)
plt.title("Female Subjects ∂y/∂x")

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
import statsmodels.api as sm

sm.graphics.fboxplot(dydx_M, wfactor=2.5)
plt.title("Male Subjects")
sm.graphics.fboxplot(dydx_F, wfactor=2.5)
plt.title("Female Subjects")

#%%
from skfda.exploratory.visualization import Boxplot

dydx_M = skfda.FDataGrid(dydx_M)
fdBoxplot_M = Boxplot(dydx_M)
# fdBoxplot_M.show_full_outliers = True
fdBoxplot_M.plot()
plt.title("Male Subjects")

dydx_F = skfda.FDataGrid(dydx_F)
fdBoxplot_F = Boxplot(dydx_F)
# fdBoxplot_M.show_full_outliers = True
fdBoxplot_F.plot()
plt.title("Female Subjects")
