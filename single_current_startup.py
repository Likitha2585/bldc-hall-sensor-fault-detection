"""
SINGLE-CURRENT START-UP GENERATOR
=================================
Only ONE phase current (ia) is used. The other two (ib, ic) are produced from it
by a 120 / 240 degree phase-angle difference, sample by sample, in real time.

Why phase ANGLE and not a fixed time delay?
  At motor start the frequency is ramping up, so "delay by T/3" changes every
  instant. Instead the block tracks the instantaneous phase of ia and builds
  ib and ic from it at -120 / +120 degrees.

How it works (per sample, causal - only uses past values, so it can run on a
microcontroller):
  1. SOGI quadrature generator: turns ia into an in-phase signal v and a
     signal qv lagging it by exactly 90 degrees.
  2. Frequency-locked loop (FLL): keeps the generator tuned while the
     frequency changes during start-up.
  3. Inverse Clarke transform:
        ib = -1/2 * v + (sqrt(3)/2) * qv
        ic = -1/2 * v - (sqrt(3)/2) * qv        (sequence = +1;  -1 swaps them)

USE IT IN YOUR OWN CODE
    gen = SingleCurrentToThreePhase(dt=50e-6)      # dt = control-loop period
    for ia_sample in stream_of_ia:
        ia, ib, ic, freq_hz = gen.update(ia_sample)

RUN THE DEMO (simulated start-up, frequency ramping from near zero):
    python single_current_startup.py
    python single_current_startup.py --f0 5 --f1 140 --ramp 0.25 --amp 10 --noise 0.02
Outputs: single_current_startup_demo.png and a printed summary.
"""

import argparse
import numpy as np

SQRT3_2 = np.sqrt(3.0) / 2.0


class SingleCurrentToThreePhase:
    def __init__(self, dt, f_init=10.0, f_min=2.0, f_max=400.0,
                 k=1.414, gamma=200.0, amp_floor=0.5, sequence=+1):
        """
        dt         control-loop sample period in seconds (e.g. 50e-6 for 20 kHz)
        f_init     starting guess for the electrical frequency (Hz)
        f_min/max  limits for the tracked frequency (Hz)
        k          SOGI damping (1.414 is the standard choice)
        gamma      FLL speed: higher locks faster but is noisier
        amp_floor  current (A) below which the FLL normalisation is held, so it
                   does not misbehave right at t = 0 when the current is ~0
        sequence   +1 if ib lags ia by 120 deg (normal), -1 for reverse rotation
        """
        self.dt = dt
        self.w = 2 * np.pi * f_init
        self.w_min, self.w_max = 2 * np.pi * f_min, 2 * np.pi * f_max
        self.k, self.gamma, self.floor = k, gamma, amp_floor
        self.seq = 1 if sequence >= 0 else -1
        self.v = 0.0
        self.qv = 0.0

    def update(self, ia_sample):
        """Feed ONE measured ia sample. Returns (ia, ib, ic, freq_hz)."""
        eps = ia_sample - self.v
        self.v += (self.k * self.w * eps - self.w * self.qv) * self.dt
        self.qv += (self.w * self.v) * self.dt

        norm = max(self.v ** 2 + self.qv ** 2, self.floor ** 2)
        self.w += -self.gamma * self.w * eps * self.qv / norm * self.dt
        self.w = min(max(self.w, self.w_min), self.w_max)

        ib = -0.5 * self.v + self.seq * SQRT3_2 * self.qv
        ic = -0.5 * self.v - self.seq * SQRT3_2 * self.qv
        return self.v, ib, ic, self.w / (2 * np.pi)


def three_phase_from_known_angle(amplitude, theta):
    """If the start-up angle is already known (open-loop reference), no
    estimation is needed - just place the phases 120/240 degrees apart."""
    return (amplitude * np.sin(theta),
            amplitude * np.sin(theta - 2 * np.pi / 3),
            amplitude * np.sin(theta - 4 * np.pi / 3))


# ------------------------------------------------------------------ demo
def simulate_startup(f0, f1, ramp, amp, noise, dt, total, seed=1):
    n = int(total / dt)
    t = np.arange(n) * dt
    f = np.where(t < ramp, f0 + (f1 - f0) * t / ramp, f1)
    theta = 2 * np.pi * np.cumsum(f) * dt
    ia, ib, ic = three_phase_from_known_angle(amp, theta)
    rng = np.random.default_rng(seed)
    ia_meas = ia + rng.normal(0, noise * amp, n)      # the ONLY measured signal
    return t, f, ia, ib, ic, ia_meas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--f0", type=float, default=5.0, help="start frequency (Hz)")
    ap.add_argument("--f1", type=float, default=140.0, help="final frequency (Hz)")
    ap.add_argument("--ramp", type=float, default=0.25, help="ramp time (s)")
    ap.add_argument("--amp", type=float, default=10.0, help="current amplitude (A)")
    ap.add_argument("--noise", type=float, default=0.02, help="noise, fraction of amplitude")
    ap.add_argument("--dt", type=float, default=50e-6, help="control period (s)")
    ap.add_argument("--total", type=float, default=0.4, help="simulated time (s)")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t, f, ia, ib, ic, ia_meas = simulate_startup(a.f0, a.f1, a.ramp, a.amp, a.noise, a.dt, a.total)

    gen = SingleCurrentToThreePhase(dt=a.dt, f_init=max(a.f0, 5.0), amp_floor=0.05 * a.amp)
    out = np.array([gen.update(x) for x in ia_meas])
    ia_h, ib_h, ic_h, f_h = out.T

    rms = a.amp / np.sqrt(2)
    blk = int(0.01 / a.dt)
    nb = len(t) // blk

    def block_err(true, hat):
        return np.array([np.sqrt(np.mean((true[i*blk:(i+1)*blk] - hat[i*blk:(i+1)*blk]) ** 2)) / rms
                         for i in range(nb)])

    eb, ec = block_err(ib, ib_h), block_err(ic, ic_h)
    worst = np.maximum(eb, ec)
    ok = np.where(worst < 0.15)[0]
    settle = None
    for i in ok:
        if np.all(worst[i:] < 0.15):
            settle = i * 0.01
            break

    print("Only ia is measured; ib and ic are generated from it.")
    print(f"Start-up: {a.f0} -> {a.f1} Hz over {a.ramp} s, amplitude {a.amp} A, noise {a.noise*100:.0f}%")
    print("Error = RMSE / rms current  (0 = perfect, 0.15 = 15% of the current)")
    print(f"{'time(ms)':>9} {'freq true':>10} {'freq est':>9} {'ib err':>7} {'ic err':>7}")
    for i in range(0, nb, max(1, nb // 12)):
        j = i * blk
        print(f"{i*10:>9} {f[j]:>10.1f} {f_h[j]:>9.1f} {eb[i]:>7.2f} {ec[i]:>7.2f}")
    print("-" * 47)
    if settle is not None:
        print(f"ib and ic stay within 15% of the true currents from t = {settle*1000:.0f} ms onward.")
    else:
        print("ib and ic did not settle within 15% in this run (try a longer time or higher gamma).")
    print(f"Steady-state error (last 100 ms): ib {eb[-10:].mean():.3f}, ic {ec[-10:].mean():.3f}")

    fig, ax = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
    ax[0].plot(t * 1000, ia_meas, linewidth=0.8)
    ax[0].set_ylabel("ia measured (A)")
    ax[0].set_title("Only ia is measured  ->  ib and ic are generated")
    ax[1].plot(t * 1000, ib, label="true ib", linewidth=1.6)
    ax[1].plot(t * 1000, ib_h, "--", label="generated ib", linewidth=1.0)
    ax[1].set_ylabel("ib (A)"); ax[1].legend(loc="upper right")
    ax[2].plot(t * 1000, ic, label="true ic", linewidth=1.6)
    ax[2].plot(t * 1000, ic_h, "--", label="generated ic", linewidth=1.0)
    ax[2].set_ylabel("ic (A)"); ax[2].legend(loc="upper right")
    ax[3].plot(t * 1000, f, label="true frequency", linewidth=1.6)
    ax[3].plot(t * 1000, f_h, "--", label="tracked frequency", linewidth=1.0)
    ax[3].set_ylabel("frequency (Hz)"); ax[3].set_xlabel("time since start (ms)")
    ax[3].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig("single_current_startup_demo.png", dpi=120)
    print("\nSaved: single_current_startup_demo.png")


if __name__ == "__main__":
    main()
