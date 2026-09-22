"""
Diagnostic version: checks whether there's a startup transient distorting
the earlier correlation estimate, by comparing signal statistics across
different windows of the recording (start / middle / end).
"""

import pandas as pd
import numpy as np
from scipy.signal import correlate
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

CONDITION = "0.01"
HALL_SENSOR = "A"
BASE_DIR = "C:/Users/likit/BLDC_Dataset_Backup/RAW_DATA_before_RGB"

file_path = f"{BASE_DIR}/{CONDITION}/{CONDITION}_delay_{HALL_SENSOR}.xlsx"
if not os.path.exists(file_path):
    alt_path = f"{BASE_DIR}/{CONDITION}/{CONDITION}_dealy_{HALL_SENSOR}.xlsx"
    if os.path.exists(alt_path):
        file_path = alt_path

print(f"Loading: {file_path}")
df = pd.read_excel(file_path)
n_total = len(df)
print(f"Total samples: {n_total}")
print()

sig_a_full = df['ia'].to_numpy()
sig_b_full = df['ib'].to_numpy()
sig_c_full = df['ic'].to_numpy()

# Check stats across 4 windows spread through the recording
windows = {
    "start (0-10k)": (0, 10000),
    "early (50k-60k)": (50000, 60000),
    "middle (300k-310k)": (300000, 310000),
    "late (580k-590k)": (580000, 590000),
}

print("=== Checking for a startup transient (stats per window) ===")
for label, (s, e) in windows.items():
    print(f"\n{label}:")
    for name, sig in [('ia', sig_a_full), ('ib', sig_b_full), ('ic', sig_c_full)]:
        chunk = sig[s:e]
        print(f"  {name}: mean={np.mean(chunk):.4f}, std={np.std(chunk):.4f}, "
              f"min={np.min(chunk):.4f}, max={np.max(chunk):.4f}")

# Plot a long stretch to visually spot any transient region
plt.figure(figsize=(14, 5))
plot_n = min(100000, n_total)
plt.plot(sig_a_full[:plot_n], label='ia', alpha=0.7, linewidth=0.5)
plt.plot(sig_b_full[:plot_n], label='ib', alpha=0.7, linewidth=0.5)
plt.plot(sig_c_full[:plot_n], label='ic', alpha=0.7, linewidth=0.5)
plt.title(f'First {plot_n} samples - looking for startup transient')
plt.xlabel('Sample index')
plt.ylabel('Current')
plt.legend()
plt.tight_layout()
plt.savefig('transient_check.png', dpi=120)
print(f"\nSaved plot: transient_check.png (shows first {plot_n} samples)")

# Now redo the lag/correlation estimate using a LATE, steady-state window
# instead of the very start, to avoid any transient
print("\n=== Re-estimating phase relationship using a steady-state window (300k-360k) ===")
s, e = 300000, 360000
sig_a = sig_a_full[s:e] - np.mean(sig_a_full[s:e])
sig_b = sig_b_full[s:e] - np.mean(sig_b_full[s:e])
sig_c = sig_c_full[s:e] - np.mean(sig_c_full[s:e])

def estimate_lag_fast(s1, s2, max_lag=25000):
    corr = correlate(s1, s2, mode='full', method='fft')
    lags = np.arange(-len(s2) + 1, len(s1))
    mask = (lags >= -max_lag) & (lags <= max_lag)
    corr_w, lags_w = corr[mask], lags[mask]
    best_idx = np.argmax(corr_w)
    best_lag = lags_w[best_idx]
    norm = np.linalg.norm(s1) * np.linalg.norm(s2)
    best_corr = corr_w[best_idx] / norm if norm > 0 else 0
    return best_lag, best_corr

lag_ab, corr_ab = estimate_lag_fast(sig_a, sig_b)
lag_ac, corr_ac = estimate_lag_fast(sig_a, sig_c)
lag_bc, corr_bc = estimate_lag_fast(sig_b, sig_c)

print(f"ia vs ib -> lag: {lag_ab} samples, correlation: {corr_ab:.4f}")
print(f"ia vs ic -> lag: {lag_ac} samples, correlation: {corr_ac:.4f}")
print(f"ib vs ic -> lag: {lag_bc} samples, correlation: {corr_bc:.4f}")
