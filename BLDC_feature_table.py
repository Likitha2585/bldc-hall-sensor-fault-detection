"""
Per-recording feature table (uses the cached currents from BLDC_raw_current_classifier.py).
Shows, for every recording, the mean +- std of the key features over its steady-state windows,
then the mean per severity level. If the fault signature is real (not just recording identity)
the values should move steadily normal -> 0.0001 -> 0.005 -> 0.01 and agree across files A/B/C.
Writes: raw_feature_table.csv (new file only)
"""
import glob, os, csv
import numpy as np
from BLDC_raw_current_classifier import windows_for, FEATS, LEVELS, CACHE

SHOW = ["recon_b", "recon_c", "corr_lag_b", "corr_lag_c", "sum_rms/a", "h3/h1", "rms_b/a", "rms_c/a", "period_ms"]
SKIP_MS = 150.0
idx = [FEATS.index(f) for f in SHOW]
rows, by_level = [], {lv: [] for lv in LEVELS}
for p in sorted(glob.glob(f"{CACHE}/*.npz")):
    lv, name = os.path.basename(p)[:-4].split("__", 1)
    z = np.load(p)
    X = windows_for(z["ia"], z["ib"], z["ic"], float(z["fs"]), SKIP_MS)
    if len(X) < 3:
        continue
    m, s = X[:, idx].mean(0), X[:, idx].std(0)
    rows.append((lv, name, len(X), m, s))
    by_level.setdefault(lv, []).append(m)

order = {lv: i for i, lv in enumerate(LEVELS)}
rows.sort(key=lambda r: (order.get(r[0], 9), r[1]))
print(f"{'recording':<24}{'n':>3}" + "".join(f"{f:>13}" for f in SHOW))
for lv, name, n, m, s in rows:
    print(f"{lv+'/'+name[-12:]:<24}{n:>3}" + "".join(f"{v:>13.3f}" for v in m))
print("\n=== MEAN PER LEVEL (should move steadily with severity) ===")
print(f"{'level':<24}{'files':>3}" + "".join(f"{f:>13}" for f in SHOW))
lvl_rows = []
for lv in LEVELS:
    if by_level.get(lv):
        M = np.mean(by_level[lv], axis=0)
        print(f"{lv:<24}{len(by_level[lv]):>3}" + "".join(f"{v:>13.3f}" for v in M))
        lvl_rows.append([lv, len(by_level[lv])] + [round(float(v), 4) for v in M])
with open("raw_feature_table.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["recording", "n_windows"] + [f"{x}_mean" for x in SHOW] + [f"{x}_std" for x in SHOW])
    for lv, name, n, m, s in rows:
        w.writerow([f"{lv}/{name}", n] + [round(float(v), 4) for v in m] + [round(float(v), 4) for v in s])
    w.writerow([]); w.writerow(["level_mean", "n_files"] + SHOW)
    w.writerows(lvl_rows)
print("\nSaved: raw_feature_table.csv")
