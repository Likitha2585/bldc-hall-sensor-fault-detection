"""
EXPERIMENT: Can ib and ic be reconstructed from ia alone using a fixed
timing-shift + amplitude-scale relationship, calibrated on HEALTHY data,
and does reconstruction accuracy degrade as the Hall-sensor fault gets
more severe?

This is a standalone, additive experiment. It does NOT modify, overwrite,
or interfere with BLDC_Hall_Detection.py, BLDC_Hall_Detection_full.py,
or results_summary.csv -- it writes its own separate output files:
  - phase_reconstruction_results.csv
  - phase_reconstruction_<condition>.png  (one plot per condition)
"""

import pandas as pd
import numpy as np
from scipy.signal import correlate
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
import os
import csv

BASE_DIR = "C:/Users/likit/BLDC_Dataset_Backup/RAW_DATA_before_RGB"
HALL_SENSOR = "A"  # consistently use the Hall-A-fault files across conditions

CONDITIONS = ["no_delay", "0.0001", "0.005", "0.01"]

CALIB_WINDOW = (300000, 360000)   # used ONLY to learn the shift/scale, from no_delay
TEST_WINDOW = (400000, 460000)    # held-out window used for the actual accuracy test
PAD = 30000                        # extra samples loaded around the test window so shifting doesn't clip

def find_file(condition):
    if condition == "no_delay":
        matches = glob.glob(f"{BASE_DIR}/no_delay/*.xlsx")
        if not matches:
            raise FileNotFoundError(f"No file found in {BASE_DIR}/no_delay")
        return matches[0]
    path = f"{BASE_DIR}/{condition}/{condition}_delay_{HALL_SENSOR}.xlsx"
    if os.path.exists(path):
        return path
    alt = f"{BASE_DIR}/{condition}/{condition}_dealy_{HALL_SENSOR}.xlsx"
    if os.path.exists(alt):
        return alt
    raise FileNotFoundError(f"Could not find file for condition {condition}, sensor {HALL_SENSOR}")

def load_phases(path):
    df = pd.read_excel(path)
    return df['ia'].to_numpy(), df['ib'].to_numpy(), df['ic'].to_numpy()

def shift_signal(sig, shift):
    """Shift sig by `shift` samples. Positive shift delays the signal."""
    result = np.zeros_like(sig)
    if shift > 0:
        if shift < len(sig):
            result[shift:] = sig[:-shift]
    elif shift < 0:
        s = -shift
        if s < len(sig):
            result[:-s] = sig[s:]
    else:
        result = sig.copy()
    return result

def estimate_lag(s1, s2, max_lag=25000):
    s1c, s2c = s1 - np.mean(s1), s2 - np.mean(s2)
    corr = correlate(s1c, s2c, mode='full', method='fft')
    lags = np.arange(-len(s2c) + 1, len(s1c))
    mask = (lags >= -max_lag) & (lags <= max_lag)
    corr_w, lags_w = corr[mask], lags[mask]
    best_idx = np.argmax(corr_w)
    return lags_w[best_idx], corr_w[best_idx] / (np.linalg.norm(s1c) * np.linalg.norm(s2c))

# ============================================================
# STEP 1: Calibrate on the healthy (no_delay) file only
# ============================================================
print("=== STEP 1: Calibrating on no_delay (healthy) data ===")
no_delay_path = find_file("no_delay")
ia_full, ib_full, ic_full = load_phases(no_delay_path)

cs, ce = CALIB_WINDOW
ia_calib = ia_full[cs:ce]
ib_calib = ib_full[cs:ce]
ic_calib = ic_full[cs:ce]

lag_ab, corr_ab = estimate_lag(ia_calib, ib_calib)
lag_ac, corr_ac = estimate_lag(ia_calib, ic_calib)

# Verify sign convention empirically: try both +lag and -lag, keep whichever
# actually gives the higher correlation when applied to the calibration data.
def best_shift_sign(ia_ref, target_ref, lag):
    opt1 = shift_signal(ia_ref, lag)
    opt2 = shift_signal(ia_ref, -lag)
    c1 = np.corrcoef(opt1, target_ref)[0, 1] if np.std(opt1) > 0 else -1
    c2 = np.corrcoef(opt2, target_ref)[0, 1] if np.std(opt2) > 0 else -1
    return (lag, c1) if c1 >= c2 else (-lag, c2)

lag_ab, verify_corr_ab = best_shift_sign(ia_calib, ib_calib, lag_ab)
lag_ac, verify_corr_ac = best_shift_sign(ia_calib, ic_calib, lag_ac)

scale_ib = np.std(ib_calib) / np.std(ia_calib)
scale_ic = np.std(ic_calib) / np.std(ia_calib)

print(f"Learned: lag_ab={lag_ab} samples (verified corr={verify_corr_ab:.4f}), scale_ib={scale_ib:.4f}")
print(f"Learned: lag_ac={lag_ac} samples (verified corr={verify_corr_ac:.4f}), scale_ic={scale_ic:.4f}")
print()

# ============================================================
# STEP 2 + 3: Apply this calibration to each condition's held-out test
# window, and measure reconstruction accuracy
# ============================================================
print("=== STEP 2+3: Testing reconstruction accuracy per condition ===")
results = []
ts, te = TEST_WINDOW

for condition in CONDITIONS:
    path = find_file(condition)
    ia_c, ib_c, ic_c = load_phases(path)

    lo = max(0, ts - PAD)
    hi = min(len(ia_c), te + PAD)
    ia_window = ia_c[lo:hi]

    ib_hat_full = scale_ib * shift_signal(ia_window, lag_ab)
    ic_hat_full = scale_ic * shift_signal(ia_window, lag_ac)

    rel_ts, rel_te = ts - lo, te - lo
    ib_hat = ib_hat_full[rel_ts:rel_te]
    ic_hat = ic_hat_full[rel_ts:rel_te]
    ib_actual = ib_c[ts:te]
    ic_actual = ic_c[ts:te]

    def score(actual, predicted):
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        norm_rmse = rmse / (np.std(actual) + 1e-9)
        corr = np.corrcoef(actual, predicted)[0, 1] if np.std(predicted) > 0 else 0
        return norm_rmse, corr

    nrmse_b, corr_b = score(ib_actual, ib_hat)
    nrmse_c, corr_c = score(ic_actual, ic_hat)

    print(f"[{condition}] ib: norm_RMSE={nrmse_b:.3f}, correlation={corr_b:.3f}  |  "
          f"ic: norm_RMSE={nrmse_c:.3f}, correlation={corr_c:.3f}")

    results.append({
        "condition": condition,
        "ib_norm_rmse": round(nrmse_b, 4),
        "ib_correlation": round(corr_b, 4),
        "ic_norm_rmse": round(nrmse_c, 4),
        "ic_correlation": round(corr_c, 4),
    })

    # Save a short comparison plot per condition
    plot_n = 3000
    plt.figure(figsize=(12, 4))
    plt.plot(ib_actual[:plot_n], label='Actual ib', alpha=0.8)
    plt.plot(ib_hat[:plot_n], label='Reconstructed ib (from ia)', alpha=0.8, linestyle='--')
    plt.title(f'Reconstruction check - condition: {condition} (first {plot_n} test samples)')
    plt.xlabel('Sample index')
    plt.ylabel('Current')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'phase_reconstruction_{condition}.png', dpi=120)
    plt.close()

with open('phase_reconstruction_results.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    writer.writeheader()
    writer.writerows(results)

print()
print("Saved: phase_reconstruction_results.csv")
print("Saved: phase_reconstruction_<condition>.png for each condition")
print()
print("=== Summary ===")
print("If norm_RMSE grows and correlation drops as condition severity increases")
print("(no_delay -> 0.0001 -> 0.005 -> 0.01), that quantitatively confirms")
print("reconstruction accuracy degrades specifically because of the fault.")
