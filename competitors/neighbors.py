"""
Different automatic neighbor selection strategies for clustering.

GUIDELINES FOR ADDING NEW NEIGHBOR SELECTION FUNCTIONS:
=======================================================
Function signature and parameter formats : 
    - Data : "X" (numpy.ndarray of shape (n_samples, n_features)). Will be provided by the experiment framework.
    - You can name your other hyperparameters as you wish.
Your function must return a positive integer.

"""
import numpy as np


def log_neighbors(X, factor = 1):
    return   max(1, int(np.floor(factor*np.log(X.shape[0]))))