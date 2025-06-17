import pandas as pd
import numpy as np

import sys
import os

# Get the current working directory (where the notebook is running)
notebook_dir = os.getcwd()

# Go up one level to the parent directory
parent_dir = os.path.abspath(os.path.join(notebook_dir, '..'))

# Add parent directory to sys.path
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Now you can import your module
from utils import factor_model_solution, sim_factor_model

X = np.array([[2, 1, 3, 4, 10],
              [5, -1, 4, 5, 3],
              [3, 5, 9, 5, 2]])


X_data_sim = (sim_factor_model(np.array([[10, 10, 0, 0, 0],
                                 [0, 0, 10, 10, 10]]).T,  np.array([1, 1, 1, 1, 1]), np.array([0, 0, 0, 0, 0]), nsim=3))

_, lambda_hat = factor_model_solution(X_data_sim, k=2)


print(X_data_sim)
print(lambda_hat)

