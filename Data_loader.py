import pandas as pd
import numpy as np

# Loading the data
train_data = pd.read_csv("mushroom_trn_data.csv")

# Converting Categorical Data into Numerical Data through one-hot-encoding


# Dummy Encoding
def dummy_enc(data_df):
    enc = pd.get_dummies(data_df, columns=['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat'], dtype = int)
    enc.loc[train_data['stalk-root'].isnull(), enc.columns.str.startswith("stalk-root_")] = np.nan
    return enc
def dummy_enc_y(data_df_y):
    enc = pd.get_dummies(data_df_y)
    return enc