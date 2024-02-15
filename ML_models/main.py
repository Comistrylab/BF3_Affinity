"""
Main module to execute BF3 Affinity programs.

This module executes programs based on user's choices of  program.
- For programs, there are 2 options: 'Cross Validation', 'Train Test Validatation'
"""
from train_model import cross_validation
from train_model import train_test_validate
from load_data import data_processing
import sys
import os
import warnings

# IGNORE WARNINGS
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter(action="ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=ConvergenceWarning)

#PROGRAM SELECTION FOR USERS
def user_input():
    """
    This function aims to get prgoram input from users.
    """
    print("Please choose a program: Cross Validation, Train Test Validation, Both")
    program = input("Program: ")
    while program not in ['Cross Validation', 'Train Test Validation', 'Both']:
        print("Error input for the 'program' variable.", end=' ')
        print("Please try again with a valid input.")
        print("\nPlease choose a program: Cross Validation, Train Test Validatation, Both")
        program = input("Program: ")

    return program

# SET UP PATHs
def setup_path():
    """
    This function aims to set up necessary paths to run the programs.

    Returns: (1) data_path: The path that leads to the directory where data is stored
             (2) results_path: If the path is already exists, return the path.
                               Otherwises, this function will create a new
                               directory to store results from programs.
    """
    curr_working_dir = os.path.dirname(__file__)
    sys.path.append(curr_working_dir)

    data_path = os.path.join(curr_working_dir, "data")

    result_path = os.path.join(curr_working_dir, "results")

    return data_path, result_path


if __name__ == '__main__':

    program = user_input()
    # SETUP PATH
    data_path, result_path = setup_path()

    # LOAD DATA
    train, test, X_train, y_train, X_test, y_test, exp_table, feature_names = data_processing(data_path)

    # RUN PROGRAM
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        if program == 'Cross Validation':
            cross_validation(X_train, y_train, feature_names, result_path)
        elif program == 'Train Test Validation':
            train_test_validate(test, X_train, y_train, X_test, y_test, exp_table, result_path)
        elif program == 'Both':
            cross_validation(X_train, y_train, feature_names, result_path)
            train_test_validate(test, X_train, y_train, X_test, y_test, exp_table, result_path)