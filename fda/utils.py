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