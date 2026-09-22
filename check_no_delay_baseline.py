"""
Same diagnostic as before, but for the no_delay (healthy) baseline file,
to check whether phase A's much smaller amplitude (seen in the 0.01s
Hall-A fault file) is caused by the fault, or is just normal for this
dataset regardless of fault status.
"""

import pandas as pd
import numpy as np
from scipy.signal import correlate
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
import os

BASE_DIR = "C:/Users/likit/BLDC_Dataset_Backup/RAW_DATA_before_RGB"
NO_DELAY_DIR = f"{BASE_DIR}/no_delay"

# Auto-detect the Excel file in the no_delay folder (name unknown)
matches = glob.glob(f"{NO_DELAY_DIR}/*.xlsx") + glob.glob(f"{NO_DELAY_DIR}/*.xls")
if not matches:
    print(f"No Excel file found in {NO_DELAY_DIR} -- listing folder contents instead:")
    for f in os.listdir(NO_DELAY_DIR):
        print(" ", f)
    raise SystemExit("Fix BASE_DIR/NO_DELAY_DIR or the file pattern and rerun.")

file_path = matches[0]
print(f"Found and loading: {file_path}")
df = pd.read_excel(file_path)
n_total = len(df)
print(f"Columns: {list(df.columns)}")
print(f"Total samples: {n_total}")
print()

sig_a_full = df['ia'].to_numpy()
sig_b_full = df['ib'].to_numpy()
sig_c_full = df['ic'].to_numpy()

windows = {
    "start (0-10k)": (0, 10000),
    "early (50k-60k)": (50000, 60000),
    "middle (300k-310k)": (300000, 310000),
    "late (580k-590k)": (580000, min(590000, n_total)),
}

print("=== Stats per window (no_delay baseline) ===")
for label, (s, e) in windows.items():
    if e > n_total:
        continue
    print(f"\n{label}:")
    for name, sig in [('ia', sig_a_full), ('ib', sig_b_full), ('ic', sig_c_full)]:
        chunk = sig[s:e]
        print(f"  {name}: mean={np.mean(chunk):.4f}, std={np.std(chunk):.4f}, "
              f"min={np.min(chunk):.4f}, max={np.max(chunk):.4f}")

plt.figure(figsize=(14, 5))
plot_n = min(100000, n_total)
plt.plot(sig_a_full[:plot_n], label='ia', alpha=0.7, linewidth=0.5)
plt.plot(sig_b_full[:plot_n], label='ib', alpha=0.7, linewidth=0.5)
plt.plot(sig_c_full[:plot_n], label='ic', alpha=0.7, linewidth=0.5)
plt.title(f'no_delay baseline - first {plot_n} samples')
plt.xlabel('Sample index')
plt.ylabel('Current')
plt.legend()
plt.tight_layout()
plt.savefig('no_delay_transient_check.png', dpi=120)
print(f"\nSaved plot: no_delay_transient_check.png")

# Steady-state correlation check (same window as the fault-file check: 300k-360k)
s, e = 300000, min(360000, n_total)
if e - s > 1000:
    print(f"\n=== Steady-state phase relationship (samples {s}-{e}) ===")
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
else:
    print("\nNot enough samples for the 300k-360k window on this file -- skipping correlation check.")
