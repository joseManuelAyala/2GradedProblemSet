import numpy as np



def kalman_filter(training_set, initial_state_mean, initial_state_var, B, H, a, c, phi, Q):
    T = len(training_set)

    # Initialize Kalman Filter with Expert Knowledge
    predicted_mean = initial_state_mean
    predicted_var = initial_state_var

    # Output Arrays
    filtered_state_means = np.zeros(T)
    filtered_state_vars = np.zeros(T)

    for t in range(T):

        # Kalman Gain
        k = predicted_var * B / (B * predicted_var * B + H)

        # Prediction Error
        v_t = training_set[t] - (a + B * predicted_mean)

        # Update posterior for alpha_t
        updated_mean = predicted_mean + k * v_t
        updated_var = (1 - k * B) * predicted_var

        # Save estimations in arrays
        filtered_state_means[t] = updated_mean
        filtered_state_vars[t] = updated_var

        # Predict next state alpha_t+1
        predicted_mean = c + phi * updated_mean
        predicted_var = phi * updated_var * phi + Q

    # Return with Kalman Filter filtered means and variances
    return filtered_state_means, filtered_state_vars
