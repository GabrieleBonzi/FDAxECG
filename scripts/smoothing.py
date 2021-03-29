#%%
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
F = pd.read_csv(data_processed + "female.csv", index_col="ecg_id")
M = pd.read_csv(data_processed + "male.csv", index_col="ecg_id")

# load ECG interval data
waves_F = np.load(data_processed + "waves_F.npy")
waves_M = np.load(data_processed + "waves_M.npy")

# load ECG signals
ecgM = load_raw_data(M, SAMPLING_RATE, data_raw)
ecgF = load_raw_data(F, SAMPLING_RATE, data_raw)

# %% Bootstrap patients
PATIENT_F = random.choices(range(waves_F.shape[0]), k=5)
PATIENT_M = random.choices(range(waves_M.shape[0]), k=5)

# %% smoothing function
def smoothedECG(ECG, intervals, show_figures=False, _beat=BEAT, _n_cheb=N_CHEB):
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

    # plot raw + smoothed ECG
    if show_figures:
        plt.figure()
        plt.plot(heartbeatRaw.data_matrix[0, :, 0], label="ECG raw")
        plt.plot(heartbeatSmoothed.data_matrix[0, :, 0], label="ECG smoothed")
        plt.legend()
        plt.show()
    return heartbeatSmoothed


# %%
smoothedFDataGrid = [
    smoothedECG(ecgF[p], waves_F[p], show_figures=False).data_matrix for p in PATIENT_F
]
length = max((len(el[0]) for el in smoothedFDataGrid))
smoothedFDataGrid = [[col[0] for col in row[0]] for row in smoothedFDataGrid]
# %% convert to numpy array
smoothedFDataGrid = ListToArray_2D(smoothedFDataGrid)
# %% convert to FDataGrid
fd = skfda.FDataGrid(smoothedFDataGrid)

# %% Alignment
_beat = BEAT
intervals = waves_F[p]
peakList = [el[_beat] for el in intervals]

peak = (peakList - peakList[0]) / SAMPLING_RATE
a = np.array([peak[1:-2]])
b = np.array([peak])
# skfda.preprocessing.registration.landmark_registration_warping(smoother.fit_transform(hartbeatRaw), a)
