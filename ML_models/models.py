"""
This module aims to get Machine Learning models for preparation in next step.
"""
# IMPORT ML MODULES
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# IMPORT UTIL MODULES
import os
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter(action="ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=ConvergenceWarning)


def get_models():
        """
        This function aims to get different Machine Learning models to prepare for 
        training process. Each model is setup with optimized hyperparameters.

        Return: A list contains tuples. Each tuples contains (1) Name of the Machine 
                Learning algorithm and (2) Model of the corresponding algorithm
        """
        # DEFINE MODELS
        lr= LinearRegression()

        ridge= Ridge(alpha=0.01)

        gb=GradientBoostingRegressor(random_state=22,
                                        n_estimators=190,
                                        criterion='friedman_mse',
                                        max_depth=7,
                                        max_features=None,
                                        learning_rate=0.08)
        
        rf=RandomForestRegressor(random_state=22,
                                        n_estimators=250,
                                        criterion='friedman_mse',
                                        max_features=None,
                                        max_depth=21,
                                        bootstrap=True)
        
        # put into a list
        models = [('LR', lr),
                ('Ridge', ridge),
                ('GB', gb),
                ('RF', rf)]

        return models

