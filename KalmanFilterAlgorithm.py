import numpy as np

def kalman_filter(y, a, B, Phi, H, Q, a1, P1, return_loglike=False):

    T = len(y)

    # Output arrays
    filtered_means = np.zeros(T)
    filtered_vars = np.zeros(T)
    prior_means = np.zeros(T)
    prior_vars = np.zeros(T)

    prior_means[0] = a1
    prior_vars[0] = P1

    # Initialize likelihood function
    log_likelihood = 0

    for t in range(T):

        # Prediction error variance
        F_t = B * prior_vars[t] * B + H

        # Kalman gain
        K_t = (prior_vars[t] * B) / F_t

        # Prediction error
        v_t = y[t] - (a + B * prior_means[t])

        # Update log_likelihood function
        log_likelihood += -0.5 * (np.log(2 * np.pi) + np.log(F_t) + (v_t ** 2) / F_t)

        # Update and store posterior means and variances
        filtered_means[t] = prior_means[t] + K_t * v_t
        filtered_vars[t] = prior_vars[t] *  (1 - K_t * B)

        # Update time, if it is not final step
        if t < T - 1:
            prior_means[t + 1] = Phi * filtered_means[t]
            prior_vars[t + 1] = (Phi**2) * filtered_vars[t] + Q


    if return_loglike:
        return -log_likelihood
    else:
        return filtered_means, filtered_vars, prior_means, prior_vars
