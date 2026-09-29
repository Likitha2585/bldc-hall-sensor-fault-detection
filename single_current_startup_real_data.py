"""
Runs the single-current generator (single_current_startup.py) against the
REAL no_delay recording instead of a simulation.

Only ia from the file is fed to the generator. ib and ic are produced by
the generator alone and then compared against the file's own real ib/ic,
which are used ONLY to check the answer, never fed into the generator.

The real recording starts near standstill, same idea as the simulated
demo, but this is the actual start-up transient in the actual motor data.

Outputs (new files only, nothing existing is touched):
    single_current_startup_real_results.csv
    single_current_startup_real.png
"""

import glob
import csv
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from single_current_startup import SingleCurrentToThreePhase

BASE_DIR = "C:/Users/likit/BLDC_Dataset_Backup/RAW_DATA_before_RGB"
ANALYSE_SAMPLES = 60000   # how much of the file to feed through the generator
BLOCK = 2000               # error reported every this many samples
STEADY_CHECK = (30000, 60000)   # window used only to pick the phase sequence sign

path = glob.glob(f"{BASE_DIR}/no_delay/*.xlsx")[0]
print("Loading:", path)
df = pd.read_excel(path)
t = df["time"].to_numpy()
ia = df["ia"].to_numpy(dtype=float)
ib = df["ib"].to_numpy(dtype=float)
ic = df["ic"].to_numpy(dtype=float)
dt = float(np.median(np.diff(t[:1000])))
n = min(ANALYSE_SAMPLES, len(ia))
print(f"Sample step: {dt:.2e} s. Using the first {n} samples ({n*dt*1000:.1f} ms).")

def run(sign, f_init=100.0):
    gen = SingleCurrentToThreePhase(dt=dt, f_init=f_init, f_min=20.0, f_max=800.0,
                                     amp_floor=0.05 * np.std(ia[STEADY_CHECK[0]:STEADY_CHECK[1]]),
                                     sequence=sign)
    out = np.empty((n, 4))
    for i in range(n):
        out[i] = gen.update(ia[i])
    return out

s0, s1 = STEADY_CHECK
out_pos = run(+1)
out_neg = run(-1)
c_pos = np.corrcoef(out_pos[s0:s1, 1], ib[s0:s1])[0, 1]
c_neg = np.corrcoef(out_neg[s0:s1, 1], ib[s0:s1])[0, 1]
out = out_pos if c_pos >= c_neg else out_neg
seq = "+1 (normal)" if c_pos >= c_neg else "-1 (reversed)"
print(f"Phase sequence that matches the real data: {seq}")

_, ib_h, ic_h, f_h = out.T
rms_b = np.std(ib[s0:s1])
rms_c = np.std(ic[s0:s1])

nb = n // BLOCK
rows = []
for i in range(nb):
    a, b = i * BLOCK, (i + 1) * BLOCK
    eb = np.sqrt(np.mean((ib[a:b] - ib_h[a:b]) ** 2)) / rms_b
    ec = np.sqrt(np.mean((ic[a:b] - ic_h[a:b]) ** 2)) / rms_c
    rows.append({"time_ms": round(a * dt * 1000, 2), "freq_hz": round(float(f_h[a]), 1),
                 "ib_err": round(float(eb), 3), "ic_err": round(float(ec), 3)})

with open("single_current_startup_real_results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

print("\nError = RMSE / real steady-state amplitude (0 = perfect, 0.15 = 15%)")
print(f"{'time(ms)':>9} {'freq(Hz)':>9} {'ib err':>7} {'ic err':>7}")
for r in rows[::max(1, nb // 20)]:
    print(f"{r['time_ms']:>9} {r['freq_hz']:>9} {r['ib_err']:>7} {r['ic_err']:>7}")

worst = np.array([max(r["ib_err"], r["ic_err"]) for r in rows])
settle = None
for i in range(nb):
    if np.all(worst[i:] < 0.15):
        settle = rows[i]["time_ms"]
        break
print("-" * 40)
if settle is not None:
    print(f"Generated ib and ic stay within 15% of the real currents from t = {settle:.1f} ms onward.")
else:
    print("Did not settle within 15% in the analysed window.")
print(f"Steady-state error (last 10 blocks): ib {np.mean([r['ib_err'] for r in rows[-10:]]):.3f}, "
      f"ic {np.mean([r['ic_err'] for r in rows[-10:]]):.3f}")

tm = t[:n] * 1000
fig, ax = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
ax[0].plot(tm, ia[:n], linewidth=0.7)
ax[0].set_ylabel("ia measured"); ax[0].set_title("Real no_delay recording: only ia is measured -> ib, ic generated")
ax[1].plot(tm, ib[:n], label="real ib", linewidth=1.4)
ax[1].plot(tm, ib_h, "--", label="generated ib", linewidth=0.9)
ax[1].set_ylabel("ib"); ax[1].legend(loc="upper right")
ax[2].plot(tm, ic[:n], label="real ic", linewidth=1.4)
ax[2].plot(tm, ic_h, "--", label="generated ic", linewidth=0.9)
ax[2].set_ylabel("ic"); ax[2].legend(loc="upper right")
ax[3].plot(tm, f_h, linewidth=1.0)
ax[3].set_ylabel("tracked freq (Hz)"); ax[3].set_xlabel("time since start (ms)")
fig.tight_layout()
fig.savefig("single_current_startup_real.png", dpi=120)
print("\nSaved: single_current_startup_real_results.csv, single_current_startup_real.png")
