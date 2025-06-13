import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm
from KalmanFilterAlgorithm import kalman_filter
from MLE import kalman_filter_objfcn
from arch import arch_model
from coverage_ratio import coverage_ratio

# Fix seed for our test-case
np.random.seed(1234)
T = 2500

# Define SV parameters
volatility_persistence= 0.95
vol_vol = 0.2
vol = - (vol_vol ** 2) / (2 * (1 - volatility_persistence ** 2))

# Array for hidden volatility values (log-volatility)
h_true = np.zeros(T)

# Start from the stationary distribution of the AR(1)
h_true[0] = np.random.normal(loc=vol, scale=vol_vol / np.sqrt(1 - volatility_persistence ** 2))

for t in range(1, T):
    h_true[t] = vol + volatility_persistence * h_true[t - 1] + np.random.normal(0, vol_vol)

# Returns (mean zero)

xi = np.random.normal(0, 1, T)
r = np.exp(0.5 * h_true) * xi

# Apply Kalman Filter

# Convert returns with log
log_squared_returns = np.log(r**2 + 1e-6)

# Define Parameters for the Kalman Filter
a = 0
B = 1.0
H = 0.2

c = 0
Q = vol_vol ** 2

initial_state_mean = 0
initial_state_var = 1

# Call Kalman Filter
filtered_means, filtered_vars, _, _ = kalman_filter(log_squared_returns, a, B, volatility_persistence, H, Q,
                                                    initial_state_mean, initial_state_var)


# Minimize log_likelihood Function
phi = 0.9
sigma_eta = 0.3
sigma = 1.2
initial_guess = [0.9, 0.3]
bounds = [(0.001, 0.999), (1e-6, None)]

# Optimize Parameters (theta_hat optimized parameters)
optimized_parameters = minimize(kalman_filter_objfcn, initial_guess, args=(log_squared_returns,), method='L-BFGS-B',
                                bounds=bounds)
theta_hat = optimized_parameters.x


# Call Kalman Filter with estimated parameters
phi_hat, sigma_eta_hat = theta_hat
a_hat = -1.27
c_hat = 0.0
H_hat = np.pi ** 2 / 2
Q_hat = sigma_eta_hat**2
filtered_means_mle, filtered_vars_mle, _, _ = kalman_filter(log_squared_returns, a_hat, B, phi_hat, H_hat, Q_hat,
                                                      initial_state_mean, initial_state_var)


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

# Print results of coverage ratio
print("EGARCH(1,1) parameter estimates:\n", res.params)
print(f"\nKF VaR coverage = {100*coverage_kalman:.2f}% (expected {alpha*100}%)")
print(f"EGARCH VaR coverage = {100*coverage_egarch:.2f}% (expected {alpha*100}%)")
