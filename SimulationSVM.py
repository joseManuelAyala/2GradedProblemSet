import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from KalmanFilterAlgorithm import kalman_filter
from MLE import mle, mle_wrapper

# Fix seed for our test-case
np.random.seed(1234)
T = 2500

# Define SV parameters
phi = 0.95
sigma_eta = 0.2
sigma = 1.0

# Array for hidden volatility values (log-volatility)
h_true = np.zeros(T)

# Start from the stationary distribution of the AR(1)
h_true[0] = np.random.normal(loc=0, scale=sigma_eta / np.sqrt(1 - phi**2))

for t in range(1, T):
    h_true[t] = phi * h_true[t-1] + np.random.normal(0, sigma_eta)

# Returns (mean zero)

xi = np.random.normal(0, 1, T)
r = sigma * np.exp(0.5 * h_true) * xi

# Apply Kalman Filter

# Convert returns with log
log_squared_returns = np.log(r**2 + 1e-6)

# Define Parameters for the Kalman Filter
a = 0
B = 1
H = 0.2

c = 0
Q = sigma_eta**2

initial_state_mean = 0
initial_state_var = 1

# Call Kalman Filter
filtered_means, filtered_vars = kalman_filter(log_squared_returns, initial_state_mean,
                                              initial_state_var, B, H, a, c, phi, Q)


# Minimize log_likelihood Function

initial_guess = [B, a, c, H, phi, Q]
bounds = [(0.8, 1.2), (-0.5, 0.5), (-0.5, 0.5), (0.05, 0.5), (0.9, 0.99), (0.005, 0.1)]

# Optimize Parameters
optimized_parameters = minimize(mle_wrapper, initial_guess, args=(log_squared_returns,), method='L-BFGS-B',
                                bounds=bounds)
theta_hat = optimized_parameters.x
print(theta_hat)

# Call Kalman Filter with estimated parameters
B_hat, a_hat, c_hat, H_hat, phi_hat, Q_hat = theta_hat
filtered_means_mle, filtered_vars_mle = kalman_filter(log_squared_returns, initial_state_mean, initial_state_var,
                                                      B_hat, H_hat, a_hat, c_hat, phi_hat, Q_hat)

# Plot volatility and returns
plt.figure(figsize=(12, 6))

# Plot latent volatility and Kalman estimate
plt.subplot(2, 1, 1)
plt.plot(h_true, color='darkorange', label='True $h_t$')
plt.plot(filtered_means, color='blue', alpha=0.7, label='Kalman $\hat{h}_t$ (manual params)')
plt.plot(filtered_means_mle, color='green', alpha=0.7, label='Kalman $\hat{h}_t$ (MLE params)')
plt.title("Latent and Estimated Log-Volatility")
plt.xlabel("Time")
plt.ylabel("$h_t$")
plt.legend()

# Plot returns
plt.subplot(2, 1, 2)
plt.plot(r, color='steelblue')
plt.title("Simulated Returns $r_t$")
plt.xlabel("Time")
plt.ylabel("$r_t$")


plt.tight_layout()
plt.show()