import numpy as np

# Computes log_likelihood Function with Kalman Filter Algorithm
def mle(training_set, B, a, c, H, phi, Q):

    T = len(training_set)
    log_likelihood = - (T / 2) * np.log(2 * np.pi)
    predicted_mean = 0
    predicted_var = 1

    for t in range(T):
        # Predict observation y_t
        y_mean = a + B * predicted_mean
        y_var = B * predicted_var * B + H

        # Prediction error
        v_t = training_set[t] - y_mean

        # Update log_likelihood Function
        log_likelihood -= 0.5 * np.log(y_var)
        log_likelihood -= 0.5 * (v_t**2) / y_var

        # Update Kalman Gain
        k_t = predicted_var * B / y_var

        # Update posterior for alpha_t
        updated_mean = predicted_mean + k_t * v_t
        updated_var = (1 - k_t * B) * predicted_var

        # Predict next state alpha_t+1
        predicted_mean = c + phi * updated_mean
        predicted_var = phi * updated_var * phi + Q

    # Returns negative log_likelihood Function, because we minimize
    return -log_likelihood


def mle_wrapper(parameters, training_set):
    B, a, c, H, phi, Q = parameters
    return mle(training_set, B, a, c, H, phi, Q)
