"""
This module aims to get the datasets.
"""
import os
import sys
import numpy as np
import pandas as pd

def data_processing(data_dir):
    """
    This function aims to get the in silico dataset as well as the experimental dataset
    and prepare useful data for ML trainning.

    Inputs: data_dir: a string for path leading to datastore (directory).
            It is setup in main module (setup_path function)

    Returns: - All data of features
             - Train data, test data and experimental data
    """
    # setup file name of dataset
    train_name = 'InSilico_dataset'
    test_name = 'Experimental_dataset'

    # read and split data
    train_path = os.path.join(data_dir, train_name + '.csv')
    test_path = os.path.join(data_dir, test_name + '.csv')

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    X_train = train.iloc[:, 5:].to_numpy()
    y_train = train.iloc[:, 4].to_numpy()

    X_test = test.iloc[:, 5:-1].to_numpy()
    y_test = test.iloc[:, 4].to_numpy()
    exp_table = test[['LB', 'SMI', 'Exp_BF3_Affinity']].drop_duplicates()
    exp_table.reset_index(drop=True)

    feature_names = train.columns[5:]  # get all features

    return train, test, X_train, y_train, X_test, y_test, exp_table, feature_names
