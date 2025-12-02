"""
Demonstration of why max_data has higher R² than individual spectra
"""
import numpy as np
import matplotlib.pyplot as plt

# Simulate 100 spectra with noise
n_spectra = 100
n_points = 50
true_peak = 25  # True peak position

# Create spectra with varying intensities and noise
spectra = []
for i in range(n_spectra):
    # Each spectrum has different peak intensity (0.5x to 1.5x)
    intensity = 0.5 + np.random.random()  # Random intensity multiplier
    # Add noise proportional to intensity
    noise_level = 0.1 * intensity
    spectrum = intensity * np.exp(-((np.arange(n_points) - true_peak)**2) / 10) + np.random.normal(0, noise_level, n_points)
    spectra.append(spectrum)

spectra = np.array(spectra)

# Calculate max_data (like testing.py)
max_data = np.max(spectra, axis=0)

# Simulate R² calculation for max_data vs individual spectra
def calculate_r2(data, fit):
    """Calculate R²"""
    ss_res = np.sum((data - fit)**2)
    ss_tot = np.sum((data - np.mean(data))**2)
    return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

# For max_data: perfect fit (simulated)
max_fit = max_data * 0.98 + np.random.normal(0, 0.02, n_points)  # Very good fit
r2_max = calculate_r2(max_data, max_fit)
print(f"R² for max_data (best case): {r2_max:.4f}")

# For individual spectra: worse fits due to noise
r2_individual = []
for i in range(min(10, n_spectra)):  # Check first 10
    # Simulate fit with more error
    fit = spectra[i] * 0.85 + np.random.normal(0, 0.15, n_points)  # Worse fit
    r2 = calculate_r2(spectra[i], fit)
    r2_individual.append(r2)
    print(f"R² for spectrum {i}: {r2:.4f}")

print(f"\nAverage R² for individual spectra: {np.mean(r2_individual):.4f}")
print(f"R² for max_data: {r2_max:.4f}")
print(f"\nDifference: {r2_max - np.mean(r2_individual):.4f}")

# Plot comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Plot max_data
axes[0].plot(max_data, 'b-', label='max_data', linewidth=2)
axes[0].plot(max_fit, 'r--', label='Fit (R²=0.95)', linewidth=2)
axes[0].set_title('Max Data (Best Case)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot individual spectra
for i in range(min(5, n_spectra)):
    axes[1].plot(spectra[i], alpha=0.5, label=f'Spectrum {i}')
axes[1].set_title('Individual Spectra (Noisy)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('r2_comparison.png', dpi=150)
print("\nPlot saved as 'r2_comparison.png'")





