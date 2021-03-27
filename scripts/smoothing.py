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

#%%
F = pd.read_csv(data_processed + "female.csv", index_col="ecg_id")
M = pd.read_csv(data_processed + "male.csv", index_col="ecg_id")

# load array
waves_F = np.load(data_processed + "waves_F.npy")
waves_M = np.load(data_processed + "waves_M.npy")

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
SAMPLING_RATE = 500
LEAD = 0
BEAT = 0
ecgM = load_raw_data(M, SAMPLING_RATE, data_raw)
ecgF = load_raw_data(F, SAMPLING_RATE, data_raw)
N_CHEB = 21

# %% Females
PATIENT_F = random.choices(np.arange(waves_F.shape[0]), k=5)

for p in PATIENT_F:
    peakList = [el[BEAT] for el in waves_F[p]]
    start = int(peakList[0])
    stop = int(peakList[-1])
    knots, t_points = compute_knots(peakList, N_CHEB)

    fd_F = skfda.FDataGrid(
        ecgF[p, start:stop, LEAD],
        t_points,
        dataset_name="ECG leads",
        argument_names=["time"],
        coordinate_names=["mV"],
    )
    # basis
    # fd_F_basis = fd_F.to_basis(skfda.representation.basis.BSpline(n_basis=4))
    N_BASIS = len(knots) + 2
    basis = skfda.representation.basis.BSpline(
        n_basis=N_BASIS, domain_range=[knots[0], knots[-1]], knots=knots
    )
    # basis.plot()

    # smoother
    smoother = skfda.preprocessing.smoothing.BasisSmoother(basis, method="cholesky")

    # smooothato
    fd_F_smoothed = smoother.fit_transform(fd_F)

    # figure
    plt.figure()
    plt.title("Subject N° " + str(p))
    plt.plot(fd_F.data_matrix[0, :, 0], label="ECG raw")
    plt.plot(fd_F_smoothed.data_matrix[0, :, 0], label="ECG smoothed")
    plt.scatter(knots*500,np.zeros(len(knots)))
    plt.legend()
    plt.show()

# %% Males
PATIENT_M = random.choices(np.arange(waves_M.shape[0]), k=5)
print(PATIENT_M)
for p in PATIENT_M:
    peakList = [el[BEAT] for el in waves_M[p]]
    start = int(peakList[0])
    stop = int(peakList[-1])
    knots, t_points = compute_knots(peakList, N_CHEB)

    fd_M = skfda.FDataGrid(
        ecgM[p, start:stop, LEAD],
        t_points,
        dataset_name="ECG leads",
        argument_names=["time"],
        coordinate_names=["mV"],
    )
    # basis
    # fd_M_basis = fd_M.to_basis(skfda.representation.basis.BSpline(n_basis=4))
    N_BASIS = len(knots) + 2
    basis = skfda.representation.basis.BSpline(
        n_basis=N_BASIS, domain_range=[knots[0], knots[-1]], knots=knots
    )
    # basis.plot()

    # smoother
    smoother = skfda.preprocessing.smoothing.BasisSmoother(basis, method="cholesky")

    # smooothato
    fd_M_smoothed = smoother.fit_transform(fd_M)

    # figure
    plt.figure()
    plt.title("Subject N° " + str(p))
    plt.plot(fd_M.data_matrix[0, :, 0], label="ECG raw")
    plt.plot(fd_M_smoothed.data_matrix[0, :, 0], label="ECG smoothed")
    plt.legend()
    plt.show()

# %% Basis
import skfda.preprocessing.smoothing.validation as val

basis1 = skfda.representation.basis.BSpline(
    n_basis=N_BASIS, domain_range=[0, (stop - start) / SAMPLING_RATE], knots=knots
)
#%%

x = np.polynomial.chebyshev.chebpts1(N_CHEB)
xx = np.polynomial.chebyshev.chebpts1(N_CHEB - 2)

x[0 : int(N_CHEB / 2)] = 1 + x[0 : int(N_CHEB / 2)]
x[int(N_CHEB / 2) :] = x[int(N_CHEB / 2) :] - 1
x = np.sort(x)

z = np.concatenate((x, xx))
z = np.sort(z)

z = np.interp(z, (z.min(), z.max()), (0, (stop - start) / SAMPLING_RATE))

knots2 = [el[BEAT] for el in waves_M[p]]
knots2 = (np.array(knots2) - start) / SAMPLING_RATE

knots2 = np.concatenate((z, knots2))
knots2 = np.sort(knots2)

knots2 = knots2[1 : len(knots2) - 1]

#%%

basis = skfda.representation.basis.BSpline(
    n_basis=51, domain_range=[0, (stop - start) / SAMPLING_RATE], knots=knots2
)
basis.plot()

# smoother
smoother = skfda.preprocessing.smoothing.BasisSmoother(
    basis, method="cholesky", smoothing_parameter=0
)

# smooothato
fd_M_smoothed = smoother.fit_transform(fd_M)

# %% figure
x_points = np.interp(
    knots2, (knots2.min(), knots2.max()), (0, len(fd_M.data_matrix[0, :, 0]))
)
plt.figure()
plt.plot(fd_M.data_matrix[0, :, 0], label="ECG raw")
plt.plot(fd_M_smoothed.data_matrix[0, :, 0], label="ECG smoothed")
plt.scatter(x_points, np.zeros(len(knots2)))
plt.legend()
plt.show()


#%%
import skfda.preprocessing.smoothing.validation as val
from sklearn import metrics

fd_M = skfda.FDataGrid(
    ecgM[p, start:stop, LEAD],
    t_points,
    dataset_name="ECG leads",
    argument_names=["time"],
    coordinate_names=["mV"],
)

for i in [pow(0.5, x) for x in np.arange(10)]:

    smoother = skfda.preprocessing.smoothing.BasisSmoother(
        basis, method="cholesky", smoothing_parameter=i
    )
    fd_M_smoothed = smoother.fit_transform(fd_M)

    m = metrics.mean_squared_error(
        fd_M.data_matrix[0, :, 0], fd_M_smoothed.data_matrix[0, :, 0]
    )

    plt.plot(fd_M_smoothed.data_matrix[0, :, 0], label="ECG smoothed")

    print(m)
