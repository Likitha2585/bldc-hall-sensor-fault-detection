"""
Runs the TWO-STAGE single-current block (single_current_startup_2stage.py) on the
REAL no_delay recording.

Only ia is fed to the block. Real ib/ic are used ONLY to score the answer and to
pick the phase-sequence sign, never as inputs to the generator.

Prints:
  1) an ia diagnostic table (mean/min/max/std per 10 ms) -> shows whether the first
     part of the recording is an inrush/locked-rotor surge or a real oscillation
  2) when Stage 2 turns on, and generated-vs-real error per block

New files written (nothing existing is touched):
    single_current_startup_2stage_real_results.csv
    single_current_startup_2stage_real.png
"""
import glob
import csv
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from single_current_startup_2stage import TwoStageThreePhase

BASE_DIR = "C:/Users/likit/BLDC_Dataset_Backup/RAW_DATA_before_RGB"
ANALYSE_SAMPLES = 1_000_000   # first 0.5 s at 2 MHz (Stage 2 needs a few full cycles)
BLOCK_MS = 10.0               # report every 10 ms


def run_block(ia, fs, sign):
    blk = TwoStageThreePhase(fs, f_min=10.0, f_max=300.0, sequence=sign)
    out = np.empty((len(ia), 3))
    for i, x in enumerate(ia):
        out[i] = blk.update(x)
    return out


def analyse(t, ia, ib, ic, prefix="single_current_startup_2stage_real"):
    dt = float(np.median(np.diff(t[:1000])))
    fs = 1.0 / dt
    n = min(ANALYSE_SAMPLES, len(ia))
    ia, ib, ic, t = ia[:n], ib[:n], ic[:n], t[:n]
    blk_n = int(BLOCK_MS * 1e-3 * fs)
    nb = n // blk_n
    print(f"Sample step: {dt:.2e} s (fs = {fs/1e6:.2f} MHz). Using first {n} samples ({n*dt*1000:.0f} ms).")

    print("\nia diagnostic (is the start a surge or a real oscillation?)")
    print(f"{'time(ms)':>9} {'mean':>8} {'std':>8} {'min':>8} {'max':>8}")
    for k in range(0, min(nb, 12)):
        s = ia[k * blk_n:(k + 1) * blk_n]
        print(f"{k*BLOCK_MS:>9.0f} {s.mean():>8.2f} {s.std():>8.2f} {s.min():>8.2f} {s.max():>8.2f}")

    # reference amplitude from the last quarter (steady state)
    q = slice(3 * n // 4, n)
    ref_b, ref_c = np.std(ib[q]), np.std(ic[q])

    outs = {s: run_block(ia, fs, s) for s in (+1, -1)}
    def score(o):
        m = ~np.isnan(o[:, 0])
        m[: n // 2] = False          # judge sign on the later half only
        if m.sum() < 100:
            return -1.0
        return np.corrcoef(o[m, 0], ib[m])[0, 1]
    sp, sn = score(outs[+1]), score(outs[-1])
    sign = +1 if sp >= sn else -1
    out = outs[sign]
    print(f"\nPhase sequence that matches the real data: {'+1 (normal)' if sign > 0 else '-1 (reversed)'}")

    ib_h, ic_h, mode = out[:, 0], out[:, 1], out[:, 2]
    on = np.flatnonzero(mode == 1)
    if on.size == 0:
        print("Stage 2 NEVER turned on: ia never showed two consistent periods in 10-300 Hz.")
    else:
        print(f"Stage 2 first turned on at {on[0]*dt*1000:.1f} ms")

    rows = []
    for k in range(nb):
        a, b = k * blk_n, (k + 1) * blk_n
        m = mode[a:b] == 1
        frac = float(m.mean())
        if m.sum() > 10:
            eb = np.sqrt(np.mean((ib_h[a:b][m] - ib[a:b][m]) ** 2)) / ref_b
            ec = np.sqrt(np.mean((ic_h[a:b][m] - ic[a:b][m]) ** 2)) / ref_c
        else:
            eb = ec = float("nan")
        rows.append({"time_ms": round(a * dt * 1000, 1), "stage2_frac": round(frac, 2),
                     "ib_err": round(float(eb), 3), "ic_err": round(float(ec), 3)})

    with open(prefix + "_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print("\nError = RMSE / real steady-state amplitude, only over samples where Stage 2 is on")
    print(f"{'time(ms)':>9} {'stage2 on':>10} {'ib err':>8} {'ic err':>8}")
    for r in rows[::max(1, nb // 25)]:
        print(f"{r['time_ms']:>9} {r['stage2_frac']:>10} {r['ib_err']:>8} {r['ic_err']:>8}")

    good = [r for r in rows if not np.isnan(r["ib_err"])]
    print("-" * 40)
    if good:
        tail = good[-10:]
        print(f"Steady-state error (last {len(tail)} blocks with output): "
              f"ib {np.mean([r['ib_err'] for r in tail]):.3f}, ic {np.mean([r['ic_err'] for r in tail]):.3f}")

    tm = t * 1000
    fig, ax = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
    ax[0].plot(tm, ia, linewidth=0.5); ax[0].set_ylabel("ia measured")
    ax[0].set_title("Real no_delay: only ia measured -> two-stage block generates ib, ic")
    ax[1].plot(tm, ib, label="real ib", linewidth=1.2); ax[1].plot(tm, ib_h, "--", label="generated ib", linewidth=0.8)
    ax[1].set_ylabel("ib"); ax[1].legend(loc="upper right")
    ax[2].plot(tm, ic, label="real ic", linewidth=1.2); ax[2].plot(tm, ic_h, "--", label="generated ic", linewidth=0.8)
    ax[2].set_ylabel("ic"); ax[2].legend(loc="upper right")
    ax[3].step(tm, mode, linewidth=1.0); ax[3].set_ylabel("stage (0=start, 1=tracking)")
    ax[3].set_xlabel("time since start (ms)")
    fig.tight_layout(); fig.savefig(prefix + ".png", dpi=120)
    print(f"\nSaved: {prefix}_results.csv, {prefix}.png")


if __name__ == "__main__":
    path = glob.glob(f"{BASE_DIR}/no_delay/*.xlsx")[0]
    print("Loading:", path)
    df = pd.read_excel(path)
    analyse(df["time"].to_numpy(),
            df["ia"].to_numpy(dtype=float),
            df["ib"].to_numpy(dtype=float),
            df["ic"].to_numpy(dtype=float))
