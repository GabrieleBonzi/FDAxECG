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
    
    y=heartbeatSmoothed.data_matrix[0,:,0]
    
    #interpolation to obtain the same number of samples
    f=make_interp_spline(t_points, y)
    new_t=np.linspace(0,t_points[-1],500)
    y_new = f(new_t).T
    
    
    # plot raw + smoothed ECG
    if show_figures:
        plt.figure()
        plt.plot(t_points, heartbeatRaw.data_matrix[0, :, 0], label="ECG raw")
        plt.plot(t_points, heartbeatSmoothed.data_matrix[0, :, 0], label="ECG smoothed")
        plt.plot(new_t,y_new)
        plt.legend()
        plt.show()
        
    return y_new


# %%
fd=[]
for p in PATIENT_F:
    smoothed = smoothedECG(ecgF[p], waves_F[p], show_figures=False)
    smoothed = skfda.FDataGrid(smoothed)
    
    fd.append(smoothed)

for i in range(len(PATIENT_F)-1):
    fd[i+1].concatenate(fd[i])


# %% Alignment LANDMARK FEATURE


peak=waves_F[PATIENT_F,:,0]
sub=peak[:,0]

peak = [(peak[:,i] - peak[:,0]) / SAMPLING_RATE for i in np.arange(peak.shape[1])]
peak=np.transpose(np.array(peak))
peak=peak[:,1:8]

land = np.array([peak])

#%%
skfda.preprocessing.registration.landmark_registration_warping(fd[4], land)
