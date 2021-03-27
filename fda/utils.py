# in notebook.ipynb
# from projectname.config import data_path
import numpy as np
import wfdb


def hello_fda():
    print("Hello FDA")


def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path + f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path + f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data


def aggregate_diagnostic(y_dic):
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return list(set(tmp))


def has_nan(list):
    arr = np.array(list)
    return np.isnan(arr) | (arr < 0)


def compute_knots(peakList=[0.0, 1.0], n_chebyshev=20, sampling_rate=500):
    start = peakList[0]
    stop = peakList[-1]
    middle = (stop + start) / 2
    t_points = np.linspace(0, (stop - start) / sampling_rate, int((stop - start)))

    chebyshev = np.polynomial.chebyshev.chebpts1(n_chebyshev)
    a = np.interp(chebyshev, (chebyshev.min(), chebyshev.max()), (start, middle))
    b = np.interp(chebyshev, (chebyshev.min(), chebyshev.max()), (middle, stop))

    knots = np.concatenate((a, b, np.array(peakList)))
    knots = np.unique(knots)
    knots = np.sort(knots)
    knots = (knots - start) / sampling_rate

    return knots, t_points