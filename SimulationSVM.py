import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm
from KalmanFilterAlgorithm import kalman_filter
from MLE import mle_wrapper
from arch import arch_model
from coverage_ratio import coverage_ratio

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

# Optimize Parameters (theta_hat optimized parameters)
optimized_parameters = minimize(mle_wrapper, initial_guess, args=(log_squared_returns,), method='L-BFGS-B',
                                bounds=bounds)
theta_hat = optimized_parameters.x
print(theta_hat)

# Call Kalman Filter with estimated parameters
B_hat, a_hat, c_hat, H_hat, phi_hat, Q_hat = theta_hat
filtered_means_mle, filtered_vars_mle = kalman_filter(log_squared_returns, initial_state_mean, initial_state_var,
                                                      B_hat, H_hat, a_hat, c_hat, phi_hat, Q_hat)


# Fit EGARCH(1,1) Model to the normal returns (not the log-squared ones)
model = arch_model(r, vol='EGARCH', p=1, q=1, dist='normal')
res = model.fit(disp='off')

# Define conditional volatility estimates
egarch_volatility = res.conditional_volatility

# Convert to log_volatility to fit the returns
egarch_log_volatility = np.log(egarch_volatility ** 2)

# Convert volatility estimates for Kalman(MLE) and EGARCH
kalman_sigma = np.exp(0.5 * filtered_means_mle)

# VaR and ES Parameters
alpha = 0.01
z_alpha = norm.ppf(alpha)
phi_z = norm.pdf(z_alpha)

# VaR and ES for Kalman approach
VaR_kalman = z_alpha * kalman_sigma
ES_kalman = - (phi_z / alpha) * kalman_sigma

# VaR and ES for EGARCH approach
VaR_egarch = z_alpha * egarch_volatility
ES_egarch = - (phi_z / alpha) * egarch_volatility

# Compute coverage ratios
coverage_kalman = coverage_ratio(r, VaR_kalman)
coverage_egarch = coverage_ratio(r, VaR_egarch)

# Plot volatility and returns
plt.figure(figsize=(12, 6))

# Plot latent volatility, Kalman estimate and Kalman with MLE estimated parameters
plt.subplot(2, 1, 1)
plt.plot(h_true, color='darkorange', label='True $h_t$')
plt.plot(filtered_means, color='blue', alpha=0.7, label='Kalman $\hat{h}_t$ (manual params)')
plt.plot(filtered_means_mle, color='green', alpha=0.7, label='Kalman $\hat{h}_t$ (MLE params)')
plt.plot(egarch_log_volatility, color='purple', alpha=0.6, label='EGARCH $\hat{h}_t$')
plt.title("Latent and Estimated Log-Volatility")
plt.xlabel("Time")
plt.ylabel("$h_t$ (log-volatility)")
plt.legend()

# Plot returns
plt.subplot(2, 1, 2)
plt.plot(r, color='steelblue')
plt.title("Simulated Returns $r_t$")
plt.xlabel("Time")
plt.ylabel("$r_t$")


plt.tight_layout()
plt.show()

# Individual Estimate Comparisons
plt.figure(figsize=(14, 10))

# 1. Kalman (manual)
plt.subplot(3, 1, 1)
plt.plot(h_true, label='True $h_t$', color='darkorange')
plt.plot(filtered_means, label='Kalman (manual)', color='blue', alpha=0.7)
plt.title("Kalman Filter (Manual Parameters)")
plt.xlabel("Time")
plt.ylabel("Log-Volatility $h_t$")
plt.legend()

# 2. Kalman (MLE)
plt.subplot(3, 1, 2)
plt.plot(h_true, label='True $h_t$', color='darkorange')
plt.plot(filtered_means_mle, label='Kalman (MLE)', color='green', alpha=0.7)
plt.title("Kalman Filter (MLE Parameters)")
plt.xlabel("Time")
plt.ylabel("Log-Volatility $h_t$")
plt.legend()

# 3. EGARCH
plt.subplot(3, 1, 3)
plt.plot(h_true, label='True $h_t$', color='darkorange')
plt.plot(egarch_log_volatility, label='EGARCH', color='purple',  alpha=0.7)
plt.title("EGARCH(1,1) Estimate")
plt.xlabel("Time")
plt.ylabel("Log-Volatility $h_t$")
plt.legend()

plt.tight_layout()
plt.show()


# Plot VaR and ES of the Kalman approach
plt.figure(figsize=(12, 5))
plt.plot(r, label='Returns', color='steelblue')
plt.plot(VaR_kalman, label='VaR 1% (Kalman)', color='green')
plt.plot(ES_kalman, label='ES 1% (Kalman)', color='limegreen', linestyle='--')
plt.title("Kalman Filter: 1% Value-at-Risk and Expected Shortfall")
plt.xlabel("Time")
plt.ylabel("Return")
plt.legend()
plt.tight_layout()
plt.show()

# Plot VaR and ES of the EGARCH approach
plt.figure(figsize=(12, 5))
plt.plot(r, label='Returns', color='steelblue')
plt.plot(VaR_egarch, label='VaR 1% (EGARCH)', color='purple')
plt.plot(ES_egarch, label='ES 1% (EGARCH)', color='violet', linestyle='--')
plt.title("EGARCH(1,1): 1% Value-at-Risk and Expected Shortfall")
plt.xlabel("Time")
plt.ylabel("Return")
plt.legend()
plt.tight_layout()
plt.show()

# Plot the coverage ratio
