
def compute_knots2(peakList=[0.0, 1.0], n_chebyshev=20, sampling_rate=500):
    if len(peakList) < 9:
        raise Exception(
            "Input list too short({}), expected to be 9".format(len(peakList))
        )
    start = peakList[0]
    stop = peakList[-1]
    middle = peakList[4]
    t_points = np.linspace(0, (stop - start) / sampling_rate, int((stop - start)))

    chebyshev = np.polynomial.chebyshev.chebpts1(n_chebyshev*2)
    cheb_start = chebyshev[int(len(chebyshev)/2):]
    cheb_stop = chebyshev [:int(len(chebyshev)/2)]
    
    a = np.interp(chebyshev, (cheb_start.min(), cheb_start.max()), (start, middle))
    b = np.interp(chebyshev, (cheb_stop.min(), cheb_stop.max()), (middle, stop))

    knots = np.concatenate((a, b, np.array(peakList)))
    knots = np.unique(knots)
    knots = np.sort(knots)
    knots = (knots - start) / sampling_rate

    return knots, t_points
# %%
def smoothedECG2(ECG, intervals, show_figures=False, _beat=BEAT, _n_cheb=N_CHEB):
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
    
    y=heartbeatSmoothed.data_matrix[0,:,0]
    
    #interpolation to obtain the same number of samples
    f=make_interp_spline(t_points, y)
    new_t=np.linspace(0,t_points[-1],500)
    y_new = f(new_t).T
    
    
    # plot raw + smoothed ECG
    if show_figures:
        plt.figure()
        plt.plot(t_points, heartbeatRaw.data_matrix[0, :, 0], label="ECG raw")
        #plt.plot(t_points, heartbeatSmoothed.data_matrix[0, :, 0], label="ECG smoothed")
        plt.plot(new_t,y_new)
        plt.legend()
        plt.show()
        
    return y_new

#%%
PATIENT_F = random.choices(range(waves_F.shape[0]), k=5)

for p in PATIENT_F:
    smoothed = smoothedECG(ecgF[p], waves_F[p], show_figures=True,_n_cheb=20)
    smoothed = smoothedECG2(ecgF[p], waves_F[p], show_figures=True,_n_cheb=18)

print(PATIENT_F)