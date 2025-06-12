import numpy as np
from KalmanFilterAlgorithm import kalman_filter

# Computes log_likelihood Function with Kalman Filter Algorithm
# (Theta are the parameters to optimize and y is the training set)
def kalman_filter_objfcn(theta, y):

    # Unpack the parameters
    phi, sigma_eta, mu = theta

    # Derived parameters
    a = 2 * np.log(mu)
    B = 1
    H = np.pi ** 2 / 2
    Q = sigma_eta ** 2

    a1 = 0.0
    P1 = Q / (1 - phi ** 2) if abs(phi) < 1 else 1.0

    # Returns negative log_likelihood Function, because we minimize
    return kalman_filter(y, a, B, phi, H, Q, a1, P1, return_loglike=True)

