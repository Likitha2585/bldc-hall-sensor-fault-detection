"""
BLDC Hall-sensor digital twin (Streamlit).

Run from the repo root:
    python -m streamlit run dashboard/digital_twin.py

The twin is the HEALTHY-MOTOR MODEL used in the paper: in a healthy six-step BLDC drive
    ib(t) = ia(t - T/3),   ic(t) = ia(t - 2T/3)
The twin predicts ib and ic from ia, compares them with the measured currents, and reports
a health indicator (correlation rho and residual e). A displaced Hall sensor breaks the
relation, so rho falls and e rises.

Two data sources:
  1. Simulated motor: a six-step BLDC current generator with an adjustable Hall displacement.
  2. Real recording: upload a CSV/XLSX with ia, ib, ic columns (e.g. the dataset recordings).
"""
import numpy as np

# ------------------------------------------------------------------ core (no UI) ----------
# Six-step current pattern per 60-degree sector, columns = (ia, ib, ic). Rows sum to zero.
SECTORS = np.array([[1, -1, 0], [1, 0, -1], [0, 1, -1], [-1, 1, 0], [-1, 0, 1], [0, -1, 1]], float)
# Which Hall sensor produces the edge that starts each sector (Ha high 0-180, Hb 120-300, Hc 240-60 deg)
EDGE_OWNER = ["A", "C", "B", "A", "C", "B"]


def lowpass(x, fs, tau):
    """First-order low-pass (models the RL current rise time)."""
    if tau <= 0:
        return x
    a = np.exp(-1.0 / (fs * tau))
    try:
        from scipy.signal import lfilter
        return lfilter([1 - a], [1, -a], x)
    except Exception:
        y = np.empty_like(x)
        acc = x[0]
        for n, v in enumerate(x):
            acc = a * acc + (1 - a) * v
            y[n] = acc
        return y


def simulate_motor(fe=138.0, disp_deg=0.0, hall="A", i_pk=10.0, fs=100e3, dur=0.08,
                   tau=0.0002, noise=0.01, seed=0):
    """Six-step phase currents with one displaced Hall sensor (displacement in electrical degrees)."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * fs)) / fs
    T = 1.0 / fe
    edges = []
    for cyc in range(-1, int(dur * fe) + 3):
        for k in range(6):
            te = cyc * T + k * T / 6
            if EDGE_OWNER[k] == hall:
                te += disp_deg / 360.0 * T
            edges.append(te)
    edges = np.array(edges)
    idx = np.searchsorted(edges, t, side="right") - 1
    sector = idx % 6
    i = i_pk * SECTORS[sector]                     # shape (N, 3)
    i = np.column_stack([lowpass(i[:, c], fs, tau) for c in range(3)])
    i = i + rng.normal(0, noise * i_pk, i.shape)
    return t, i[:, 0], i[:, 1], i[:, 2]


def estimate_period(ia, fs, fmin=50.0, fmax=300.0):
    """Electrical period from the autocorrelation peak of ia."""
    x = ia - np.mean(ia)
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n]
    ac = ac / (ac[0] + 1e-12)
    lo, hi = int(fs / fmax), min(int(fs / fmin), n - 1)
    lag = lo + int(np.argmax(ac[lo:hi]))
    return lag / fs


def twin_predict(ia, fs, T):
    """Healthy-motor twin: ib_hat = ia delayed by T/3, ic_hat = ia delayed by 2T/3."""
    n = np.arange(len(ia), dtype=float)
    ib_hat = np.interp(n - T / 3 * fs, n, ia)
    ic_hat = np.interp(n - 2 * T / 3 * fs, n, ia)
    return ib_hat, ic_hat


def health_metrics(ia, ib, ic, fs, T=None):
    if T is None:
        T = estimate_period(ia, fs)
    ib_hat, ic_hat = twin_predict(ia, fs, T)
    s = int(np.ceil(2 * T * fs))                  # skip the delay-line warm-up
    def rho(a, b): return float(np.corrcoef(a[s:], b[s:])[0, 1])
    def err(a, b): return float(np.sqrt(np.mean((a[s:] - b[s:]) ** 2)) / (np.sqrt(np.mean(a[s:] ** 2)) + 1e-12))
    return dict(T=T, rho_b=rho(ib, ib_hat), rho_c=rho(ic, ic_hat),
                e_b=err(ib, ib_hat), e_c=err(ic, ic_hat), ib_hat=ib_hat, ic_hat=ic_hat)


def sweep(hall, fe, i_pk, fs, tau, noise, max_deg=55, steps=23):
    degs = np.linspace(0, max_deg, steps)
    rb, rc = [], []
    for d in degs:
        _, a, b, c = simulate_motor(fe, d, hall, i_pk, fs, 0.08, tau, noise)
        m = health_metrics(a, b, c, fs)
        rb.append(m["rho_b"]); rc.append(m["rho_c"])
    return degs, np.array(rb), np.array(rc)


# ------------------------------------------------------------------ UI --------------------
def main():
    import streamlit as st
    import pandas as pd
    import matplotlib.pyplot as plt
    import time

    st.set_page_config(page_title="BLDC Hall-Sensor Digital Twin", layout="wide")
    st.title("BLDC Hall-Sensor Digital Twin")
    st.caption(
        "The twin is the healthy-motor model: ib(t) = ia(t - T/3) and ic(t) = ia(t - 2T/3). "
        "It predicts ib and ic from ia and compares them with the measured currents."
    )

    threshold = st.sidebar.slider("Alarm threshold on correlation", 0.80, 0.999, 0.990, 0.001,
                                  help="Alarm if the correlation between measured and twin-predicted phase falls below this.")
    source = st.sidebar.radio("Data source", ["Simulated motor", "Real recording (upload)"])

    def show(ia, ib, ic, fs, label_x_ms=None, title_suffix=""):
        m = health_metrics(ia, ib, ic, fs)
        rho_min = min(m["rho_b"], m["rho_c"])
        faulty = rho_min < threshold
        flat = np.std(ia) < 0.2 * max(np.std(ib), np.std(ic), 1e-12)
        at_limit = abs(m["T"] - 1 / 300.0) < 0.03 * (1 / 300.0) or abs(m["T"] - 1 / 50.0) < 0.03 * (1 / 50.0)
        if at_limit:
            st.warning("No repeating period was found in ia (estimated T is at the search limit). "
                       "Check the ia / ib / ic column choices in the sidebar; the status is not meaningful until then.")
        elif flat:
            st.warning("ia carries very little current compared with ib / ic. On a HEALTHY recording this usually means "
                       "the wrong column was chosen. On a heavily faulted recording (for example the 0.01 s files) it can be "
                       "a genuine effect of the fault, so read the result with care.")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Twin status", "FAULT SUSPECTED" if faulty else "HEALTHY")
        c2.metric("rho_b / rho_c", f"{m['rho_b']:.3f} / {m['rho_c']:.3f}")
        c3.metric("Residual e_b / e_c", f"{m['e_b']:.3f} / {m['e_c']:.3f}")
        c4.metric("Estimated period T", f"{m['T'] * 1000:.2f} ms")
        T = m["T"]
        span = int(min(len(ia), 3 * T * fs))
        start = max(int(2 * T * fs), 0)
        sl = slice(start, start + span)
        tt = (np.arange(len(ia)) / fs * 1000)[sl]
        fig, ax = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
        ax[0].plot(tt, ib[sl], label="measured ib")
        ax[0].plot(tt, m["ib_hat"][sl], "--", label="twin ib (ia delayed T/3)")
        ax[0].set_ylabel("ib (A)"); ax[0].legend(loc="upper right", fontsize=8)
        ax[1].plot(tt, ic[sl], label="measured ic")
        ax[1].plot(tt, m["ic_hat"][sl], "--", label="twin ic (ia delayed 2T/3)")
        ax[1].set_ylabel("ic (A)"); ax[1].set_xlabel("time (ms)"); ax[1].legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        return m

    if source == "Simulated motor":
        st.info("Simulated six-step BLDC drive. This is NOT measured data. The displacement is in electrical "
                "degrees and is not the same scale as the dataset's 0.0001 / 0.005 / 0.01 s timer delays.")
        hall = st.sidebar.selectbox("Displaced Hall sensor", ["A", "B", "C"])
        fe = st.sidebar.slider("Electrical frequency (Hz)", 60, 250, 138)
        i_pk = st.sidebar.slider("Peak current (A)", 2.0, 20.0, 10.0)
        noise = st.sidebar.slider("Sensor noise (fraction of peak)", 0.0, 0.10, 0.01, 0.005)
        disp = st.sidebar.slider("Hall displacement (electrical degrees)", 0.0, 55.0, 0.0, 1.0)
        fs, tau = 100e3, 0.0002
        placeholder = st.empty()
        if st.sidebar.button("Animate: ramp the fault up"):
            for d in np.linspace(0, 55, 28):
                _, a, b, c = simulate_motor(fe, d, hall, i_pk, fs, 0.08, tau, noise)
                with placeholder.container():
                    st.subheader(f"Displacement {d:.0f} degrees")
                    show(a, b, c, fs)
                time.sleep(0.15)
        else:
            _, a, b, c = simulate_motor(fe, disp, hall, i_pk, fs, 0.08, tau, noise)
            with placeholder.container():
                show(a, b, c, fs)
        st.subheader("Health indicator versus displacement")
        degs, rb, rc = sweep(hall, fe, i_pk, fs, tau, noise)
        fig, ax = plt.subplots(figsize=(8, 3.2))
        ax.plot(degs, rb, "o-", label="rho_b"); ax.plot(degs, rc, "s-", label="rho_c")
        ax.axhline(threshold, color="red", linestyle="--", label="alarm threshold")
        ax.axvline(disp, color="grey", linestyle=":", label="current setting")
        ax.set_xlabel("Hall displacement (electrical degrees)"); ax.set_ylabel("correlation with twin")
        ax.legend(fontsize=8)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    else:
        st.info("Upload a recording with three current columns (ia, ib, ic). The twin uses only ia to predict "
                "ib and ic, and compares them with the recorded ones.")
        up = st.sidebar.file_uploader("CSV or XLSX recording", type=["csv", "xlsx"])
        if up is None:
            st.stop()
        df = pd.read_csv(up) if up.name.lower().endswith(".csv") else pd.read_excel(up)
        num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if len(num) < 3:
            st.error("Need at least three numeric columns."); st.stop()
        with st.expander("First rows of the uploaded file (check which column is which)"):
            st.dataframe(df.head(10))
            st.write("Columns:", list(df.columns))

        def guess(key, fallback):
            for i, c in enumerate(num):
                name = str(c).lower().replace(" ", "").replace("_", "")
                if name in (key, "i" + key[-1], "current" + key[-1]) or name.endswith(key):
                    return i
            return min(fallback, len(num) - 1)

        ca = st.sidebar.selectbox("ia column", num, index=guess("ia", 0))
        cb = st.sidebar.selectbox("ib column", num, index=guess("ib", 1))
        cc = st.sidebar.selectbox("ic column", num, index=guess("ic", 2))
        fs_raw = st.sidebar.number_input("Original sampling rate (Hz)", value=2_000_000, step=100_000)
        dec = st.sidebar.number_input("Decimation factor", value=20, min_value=1)
        skip_ms = st.sidebar.number_input("Skip start-up transient (ms)", value=150, min_value=0)
        win_ms = st.sidebar.number_input("Analysis window (ms)", value=100, min_value=20)
        fs = fs_raw / dec
        def prep(col):
            x = df[col].to_numpy(float)
            n = len(x) // dec * dec
            return x[:n].reshape(-1, dec).mean(axis=1)
        a, b, c = prep(ca), prep(cb), prep(cc)
        s0 = int(skip_ms / 1000 * fs); s1 = s0 + int(win_ms / 1000 * fs)
        a, b, c = a[s0:s1], b[s0:s1], c[s0:s1]
        if len(a) < int(0.03 * fs):
            st.error("Window too short for the chosen settings."); st.stop()
        show(a, b, c, fs)
        st.caption("Reference values from the paper: healthy rho about 0.998; at the largest fault rho about 0.29. "
                   "With one healthy recording the alarm threshold is not statistically validated.")

    with st.expander("What this twin is, and what it is not"):
        st.markdown(
            "- It is a **model-based twin of the healthy phase-current relationship**, not a full electromagnetic or thermal model.\n"
            "- In simulated mode the 'motor' is a generator, so results show the principle, not real-motor performance.\n"
            "- In real mode it only works on a rotating motor (after the start-up surge).\n"
            "- The alarm threshold is a setting; it has not been validated on more than one healthy recording."
        )


if __name__ == "__main__":
    main()
