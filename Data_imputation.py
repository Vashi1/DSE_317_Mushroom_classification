# Use on the Transformed Data
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
train_data = pd.read_csv('mushroom_trn_data.csv')

# While calling also specify value of k, otherwise default value of 5 will be taken for k
def knn_imputer(data_df, k=5):
    knn = KNNImputer(n_neighbors=k)
    data_headers = list(data_df.columns.values)
    data_imp = knn.fit_transform(data_df)
    data_imp_df = pd.DataFrame(data_imp, columns=data_headers)
    data_imp_df.loc[train_data['stalk-root'].isnull(), data_imp_df.columns.str.startswith("stalk-root_")] = np.nan
    return data_imp_df



def mean_imputer(data_df):
    si_m = SimpleImputer(strategy='mean')
    data_headers = list(data_df.columns.values)
    data_imp_mean = si_m.fit_transform(data_df)
    data_imp_mean_df = pd.DataFrame(data_imp_mean, columns=data_headers)
    return data_imp_mean_df


def median_imputer(data_tf):
    si_med = SimpleImputer(strategy='median')
    data_headers = list(data_tf.columns.values)
    data_imp_med = si_med.fit_transform(data_tf)
    data_imp_med_df = pd.DataFrame(data_imp_med, columns=data_headers)
    return data_imp_med_df


def mf_imputer(data_tf):
    si_mf = SimpleImputer(strategy='most_frequent')
    data_headers = list(data_tf.columns.values)
    data_imp_mf = si_mf.fit_transform(data_tf)
    data_imp_mf_df = pd.DataFrame(data_imp_mf, columns=data_headers)
    return data_imp_mf_df
