import numpy as np
import matplotlib.pyplot as plt

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

# Plot volatility and returns
plt.figure(figsize=(12, 6))

# Plot latent volatility
plt.subplot(2, 1, 1)
plt.plot(h_true, color='darkorange')
plt.title("Latent Log-Volatility $h_t$")
plt.xlabel("Time")
plt.ylabel("$h_t$")

# Plot returns
plt.subplot(2, 1, 2)
plt.plot(r, color='steelblue')
plt.title("Simulated Returns $r_t$")
plt.xlabel("Time")
plt.ylabel("$r_t$")

plt.tight_layout()
plt.show()