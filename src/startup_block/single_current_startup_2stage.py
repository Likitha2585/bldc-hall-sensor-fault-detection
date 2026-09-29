"""
Two-stage single-current start-up block for a six-step BLDC drive.

Only ia is measured. ib and ic are produced from it (ib = ia delayed by T/3,
ic = ia delayed by 2T/3, T = electrical period).

Why a delay line and not a sine-tracking loop (SOGI/PLL):
  six-step phase currents are NOT sinusoids; they are quasi-square blocks that are
  exact 120-degree shifted copies of each other. A delay line keeps the true shape,
  a sine model only reproduces the fundamental.

STAGE 1  (start-up, rotor locked / inrush / too few cycles seen):
    No electrical frequency exists yet. The block reports mode=0 and outputs NaN.
    The drive runs its normal open-loop forced start here (not ia-based).
STAGE 2  (tracking):
    Enabled only when ia has a valid oscillation: two consecutive periods agree
    within `tol`, frequency is inside [f_min, f_max]. Then ib/ic come from the
    delay line. Uses only past samples of ia (causal).
"""
import numpy as np


class TwoStageThreePhase:
    def __init__(self, fs, f_min=10.0, f_max=300.0, tol=0.15,
                 hyst_frac=0.3, peak_tau=0.02, sequence=+1):
        self.fs, self.f_min, self.f_max, self.tol = fs, f_min, f_max, tol
        self.hyst_frac = hyst_frac
        self.peak_decay = 1.0 - 1.0 / (fs * peak_tau)
        self.seq = sequence                 # +1: ib lags ia by 120 deg
        n = int(np.ceil(fs / f_min)) + 4    # buffer holds one longest period
        self.buf = np.zeros(n)
        self.n = n
        self.i = 0                          # sample counter
        self.peak = 0.0
        self.low = False                    # seen ia < -thr since last event
        self.prev_ev = None                 # time (in samples) of last rising crossing
        self.periods = []                   # recent periods in samples
        self.T = None                       # period used for delays (samples)
        self.prev = 0.0

    def _delayed(self, d):
        """ia delayed by d samples (fractional, linear interpolation)."""
        d = min(max(d, 0.0), self.n - 2)
        k = int(d); fr = d - k
        a = self.buf[(self.i - k) % self.n]
        b = self.buf[(self.i - k - 1) % self.n]
        return (1 - fr) * a + fr * b

    def update(self, ia):
        self.buf[self.i % self.n] = ia
        self.peak = max(abs(ia), self.peak * self.peak_decay)
        thr = self.hyst_frac * self.peak

        # rising zero-crossing detection with hysteresis (one event per period)
        if ia < -thr:
            self.low = True
        elif self.low and ia > thr and self.prev <= thr:
            frac = (thr - self.prev) / (ia - self.prev + 1e-12)
            t_ev = self.i - 1 + frac
            if self.prev_ev is not None:
                self.periods.append(t_ev - self.prev_ev)
                self.periods = self.periods[-3:]
                self._validate()
            self.prev_ev = t_ev
            self.low = False
        self.prev = ia

        out = (np.nan, np.nan, 0)
        if self.T is not None:
            d = self.T / 3.0
            s = 1 if self.seq > 0 else -1
            ib = self._delayed(d if s > 0 else 2 * d)
            ic = self._delayed(2 * d if s > 0 else d)
            out = (ib, ic, 1)
        self.i += 1
        return out

    def _validate(self):
        p = self.periods
        if len(p) < 2:
            return
        a, b = p[-2], p[-1]
        f = self.fs / b
        ok = abs(a - b) / max(a, b) < self.tol and self.f_min <= f <= self.f_max
        if ok:
            self.T = b
        else:
            self.T = None      # lost lock -> back to stage 1

    @property
    def freq(self):
        return None if self.T is None else self.fs / self.T


# ---------------------------------------------------------------- demo ----
def six_step(theta):
    return np.clip(2.0 * np.sin(theta), -1, 1)   # quasi-square, 120-deg blocks


def make_startup(fs=100e3, f0=5.0, f1=140.0, ramp=0.25, total=0.5,
                 inrush=True, amp=10.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(total * fs)) / fs
    f = np.where(t < ramp, f0 + (f1 - f0) * t / ramp, f1)
    th = 2 * np.pi * np.cumsum(f) / fs
    ia, ib, ic = (amp * six_step(th - k * 2 * np.pi / 3) for k in range(3))
    if inrush:  # one-sided decaying surge on every phase, first ~10 ms
        surge = 4 * amp * np.exp(-t / 0.004)
        ia, ib, ic = ia + surge, ib + surge, ic + surge
    n = 0.02 * amp
    return t, f, ia + n * rng.standard_normal(t.size), ib, ic


if __name__ == "__main__":
    fs = 100e3
    t, f, ia, ib, ic = make_startup(fs)
    blk = TwoStageThreePhase(fs)
    out = np.array([blk.update(x) for x in ia])
    mode = out[:, 2]
    first = np.argmax(mode == 1) / fs * 1e3
    print(f"Stage 2 first enabled at {first:.1f} ms")
    print(" time(ms)  mode  ib err  ic err")
    for a in range(0, int(0.5 * fs), int(0.05 * fs)):
        sl = slice(a, a + int(0.05 * fs))
        m = mode[sl] == 1
        if m.sum() < 10:
            print(f"{a/fs*1e3:8.0f}   1-stage   --      --"); continue
        rb = np.sqrt(np.mean((out[sl, 0][m] - ib[sl][m]) ** 2)) / 7.0
        rc = np.sqrt(np.mean((out[sl, 1][m] - ic[sl][m]) ** 2)) / 7.0
        print(f"{a/fs*1e3:8.0f}   2-stage {rb:6.3f}  {rc:6.3f}")
