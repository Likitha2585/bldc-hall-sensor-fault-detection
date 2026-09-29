"""
Runs the two-stage single-current block on EVERY recording folder under BASE_DIR
(no_delay + the Hall-sensor-delay folders) and prints one comparison table.
Only ia goes into the block; real ib/ic are used only to score it and pick the sign.
Writes: single_current_startup_2stage_all_summary.csv  (new file only)
"""
import glob, os, csv
import numpy as np
import pandas as pd
from single_current_startup_2stage import TwoStageThreePhase
from single_current_startup_2stage_real import run_block, ANALYSE_SAMPLES

BASE_DIR = "C:/Users/likit/BLDC_Dataset_Backup/RAW_DATA_before_RGB"


def summarise(folder, path):
    df = pd.read_excel(path)
    t = df["time"].to_numpy(); ia = df["ia"].to_numpy(float)
    ib = df["ib"].to_numpy(float); ic = df["ic"].to_numpy(float)
    dt = float(np.median(np.diff(t[:1000]))); fs = 1 / dt
    n = min(ANALYSE_SAMPLES, len(ia))
    ia, ib, ic = ia[:n], ib[:n], ic[:n]
    q = slice(3 * n // 4, n)
    outs = {s: run_block(ia, fs, s) for s in (+1, -1)}

    def score(o):
        m = ~np.isnan(o[:, 0]); m[: n // 2] = False
        return np.corrcoef(o[m, 0], ib[m])[0, 1] if m.sum() > 100 else -1.0
    sign = +1 if score(outs[+1]) >= score(outs[-1]) else -1
    o = outs[sign]
    on = np.flatnonzero(o[:, 2] == 1)
    first_ms = on[0] * dt * 1000 if on.size else float("nan")
    m = o[q, 2] == 1
    if m.sum() > 10:
        eb = np.sqrt(np.mean((o[q, 0][m] - ib[q][m]) ** 2)) / np.std(ib[q])
        ec = np.sqrt(np.mean((o[q, 1][m] - ic[q][m]) ** 2)) / np.std(ic[q])
    else:
        eb = ec = float("nan")
    return {"recording": folder, "samples": n, "ia_first10ms_min": round(float(ia[:int(0.01*fs)].min()), 2),
            "stage2_on_ms": round(float(first_ms), 1), "ib_err_steady": round(float(eb), 3),
            "ic_err_steady": round(float(ec), 3), "sequence": sign}


rows = []
for d in sorted(p for p in glob.glob(f"{BASE_DIR}/*") if os.path.isdir(p)):
    files = glob.glob(f"{d}/*.xlsx")
    if not files:
        continue
    name = os.path.basename(d)
    print(f"\n--- {name}: {files[0]}")
    r = summarise(name, files[0])
    rows.append(r)
    print(r)

print("\n=== SUMMARY (error = RMSE / steady amplitude, last quarter, Stage-2 samples only) ===")
print(f"{'recording':<14}{'stage2 on (ms)':>15}{'ib err':>9}{'ic err':>9}{'seq':>5}")
for r in rows:
    print(f"{r['recording']:<14}{r['stage2_on_ms']:>15}{r['ib_err_steady']:>9}{r['ic_err_steady']:>9}{r['sequence']:>5}")
with open("single_current_startup_2stage_all_summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("\nSaved: single_current_startup_2stage_all_summary.csv")
