import numpy as np



def kalman_filter(training_set, initial_state_mean, initial_state_var):
    T = len(training_set)
    # Initialize Kalman Filter with Expert Knowledge

    predicted_mean = initial_state_mean
    predicted_var = initial_state_var

    # Output Arrays

    filtered_state_means = np.zeros(T)
    filtered_state_vars = np.zeros(T)