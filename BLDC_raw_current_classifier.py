"""
Hall-sensor fault detection from the RAW phase currents (no images, no ImageNet).

Per 40 ms window of steady-state current we compute ~30 hand-made features
(RMS ratios, phase correlations, harmonics, sum of the three currents, and the
"start-up reconstruction residual": how well ib, ic equal ia delayed by T/3, 2T/3),
then classify normal vs fault with small classical models.

Two honest evaluations (no test-set tuning anywhere, fixed hyper-parameters):
  A  time-split  : first 70% of every recording trains, last 30% tests (with a gap).
  B  unseen file : train on the other fault file(s) of that level, test on the held-out
                   fault file (normal still time-split). Harder, closer to "new motor run".
Metric: balanced accuracy (chance = 50%), plus macro-F1 in the CSV.

CAVEAT (state it in the report): there is only ONE normal recording, so a model could
partly learn "which recording is this" instead of "is the Hall sensor displaced".

Outputs (new files): raw_current_results.csv ; raw_cache/*.npz (decimated currents)
"""
import os, glob, csv, argparse
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score

BASE_DIR = "C:/Users/likit/BLDC_Dataset_Backup/RAW_DATA_before_RGB"
LEVELS = ["no_delay", "0.0001", "0.005", "0.01"]
DEC = 20                 # 2 MHz -> 100 kHz (block mean)
WIN_S, HOP_S = 0.040, 0.010
CACHE = "raw_cache"

FEATS = ["rms_b/a", "rms_c/a", "mean_a/rms", "mean_b/rms", "mean_c/rms",
         "crest_a", "crest_b", "crest_c", "kurt_a", "skew_a",
         "corr_ab", "corr_bc", "corr_ca", "sum_rms/a", "period_ms", "ac_peak",
         "h3/h1", "h5/h1", "h7/h1",
         "recon_b", "recon_c", "corr_lag_b", "corr_lag_c"]
RECON = ["recon_b", "recon_c", "corr_lag_b", "corr_lag_c"]


# ----------------------------------------------------------------- loading
def load_recording(path, folder):
    os.makedirs(CACHE, exist_ok=True)
    tag = f"{CACHE}/{folder}__{os.path.splitext(os.path.basename(path))[0]}.npz"
    if os.path.exists(tag):
        z = np.load(tag)
        return z["ia"], z["ib"], z["ic"], float(z["fs"])
    print("  reading Excel (slow, cached afterwards):", path)
    df = pd.read_excel(path, usecols=["time", "ia", "ib", "ic"])
    t = df["time"].to_numpy()
    dt = float(np.median(np.diff(t[:1000])))
    def dec(x):
        x = df[x].to_numpy(dtype=np.float32)
        n = len(x) // DEC * DEC
        return x[:n].reshape(-1, DEC).mean(1)
    ia, ib, ic = dec("ia"), dec("ib"), dec("ic")
    fs = 1.0 / (dt * DEC)
    np.savez(tag, ia=ia, ib=ib, ic=ic, fs=fs)
    return ia, ib, ic, fs


# ---------------------------------------------------------------- features
def _corr(x, y):
    x = x - x.mean(); y = y - y.mean()
    d = np.sqrt((x * x).sum() * (y * y).sum()) + 1e-12
    return float((x * y).sum() / d)


def window_features(a, b, c, fs):
    n = len(a)
    ra, rb, rc = (np.sqrt(np.mean(v ** 2)) for v in (a, b, c))
    if ra < 1e-6 or rb < 1e-6 or rc < 1e-6:
        return None
    x = a - a.mean()
    F = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(F * np.conj(F))[:n]
    ac = ac / (ac[0] + 1e-12)
    lo, hi = int(fs / 300), int(fs / 50)
    k = lo + int(np.argmax(ac[lo:hi]))
    if 0 < k < n - 1:                       # parabolic refinement
        y0, y1, y2 = ac[k - 1], ac[k], ac[k + 1]
        den = y0 - 2 * y1 + y2
        T = k + (0.5 * (y0 - y2) / den if abs(den) > 1e-12 else 0.0)
    else:
        T = float(k)
    peak = float(ac[k])

    idx = np.arange(n)
    def delayed(d):
        j = idx - d
        ok = j >= 0
        return ok, np.interp(j[ok], idx, a)
    okb, ad_b = delayed(T / 3.0)
    okc, ad_c = delayed(2.0 * T / 3.0)
    recon_b = np.sqrt(np.mean((b[okb] - ad_b) ** 2)) / rb
    recon_c = np.sqrt(np.mean((c[okc] - ad_c) ** 2)) / rc
    corr_lag_b = _corr(b[okb], ad_b)
    corr_lag_c = _corr(c[okc], ad_c)

    spec = np.abs(np.fft.rfft(x * np.hanning(n)))
    df_ = fs / n
    f0 = fs / T
    def amp(kk):
        i = int(round(kk * f0 / df_))
        if i + 2 >= len(spec):
            return 0.0
        return float(spec[max(i - 1, 0):i + 2].max())
    h1 = amp(1) + 1e-12

    def stats(v):
        m = v.mean(); s = v.std() + 1e-12; z = (v - m) / s
        return m / (np.sqrt(np.mean(v ** 2)) + 1e-12), np.abs(v).max() / (np.sqrt(np.mean(v ** 2)) + 1e-12), \
               float(np.mean(z ** 4)), float(np.mean(z ** 3))
    ma, ca, ka, sa = stats(a); mb, cb, _, _ = stats(b); mc, cc, _, _ = stats(c)
    return [rb / ra, rc / ra, ma, mb, mc, ca, cb, cc, ka, sa,
            _corr(a, b), _corr(b, c), _corr(c, a),
            float(np.sqrt(np.mean((a + b + c) ** 2)) / ra),
            T / fs * 1000.0, peak, amp(3) / h1, amp(5) / h1, amp(7) / h1,
            recon_b, recon_c, corr_lag_b, corr_lag_c]


def windows_for(ia, ib, ic, fs, skip_ms):
    s0 = int(skip_ms * 1e-3 * fs)
    W, H = int(WIN_S * fs), int(HOP_S * fs)
    rows = []
    for a in range(s0, len(ia) - W, H):
        f = window_features(ia[a:a + W].astype(float), ib[a:a + W].astype(float), ic[a:a + W].astype(float), fs)
        if f is not None and np.all(np.isfinite(f)):
            rows.append(f)
    return np.array(rows)


# ------------------------------------------------------------- evaluation
def models():
    return {
        "LogReg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced")),
        "SVM-RBF": make_pipeline(StandardScaler(), SVC(class_weight="balanced")),
        "RandForest": RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                             random_state=0, n_jobs=-1),
    }


def fit_score(model, Xtr, ytr, Xte, yte, cols=None):
    if cols is not None:
        Xtr, Xte = Xtr[:, cols], Xte[:, cols]
    model.fit(Xtr, ytr)
    p = model.predict(Xte)
    return balanced_accuracy_score(yte, p), f1_score(yte, p, average="macro")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE_DIR)
    ap.add_argument("--skip-ms", type=float, default=150.0, help="drop the start-up transient")
    a = ap.parse_args()

    recs = []                                   # (level, filename, X)
    for lv in LEVELS:
        files = sorted(glob.glob(f"{a.base}/{lv}/*.xlsx"))
        if not files:
            files = sorted(glob.glob(f"{CACHE}/{lv}__*.npz"))
        for p in files:
            name = os.path.splitext(os.path.basename(p))[0]
            cp = f"{CACHE}/{lv}__{name}.npz"
            ia, ib, ic, fs = load_recording(p if p.endswith(".xlsx") else cp, lv) if p.endswith(".xlsx") \
                else (lambda z: (z["ia"], z["ib"], z["ic"], float(z["fs"])))(np.load(p))
            X = windows_for(ia, ib, ic, fs, a.skip_ms)
            print(f"{lv:<9}{name:<26}{len(X):>5} windows  (fs {fs/1e3:.0f} kHz, {len(ia)/fs*1000:.0f} ms)")
            if len(X) >= 10:
                recs.append((lv, name, X))
    if not any(r[0] == "no_delay" for r in recs):
        raise SystemExit("No usable no_delay windows.")

    gap = int(round(WIN_S / HOP_S))
    def tsplit(X):
        m = len(X); ntr = int(0.7 * m)
        return X[:max(ntr - gap, 1)], X[ntr:]
    norm = [tsplit(r[2]) for r in recs if r[0] == "no_delay"]
    Ntr = np.vstack([n[0] for n in norm]); Nte = np.vstack([n[1] for n in norm])

    out = []
    print("\n=== Balanced accuracy % (chance = 50). A = time-split, B = unseen fault file ===")
    print(f"{'level':<9}{'model':<12}{'A: time-split':>15}{'B: unseen file':>16}")
    for lv in LEVELS[1:]:
        fr = [r for r in recs if r[0] == lv]
        if not fr:
            continue
        tr_te = [tsplit(r[2]) for r in fr]
        Xtr = np.vstack([Ntr] + [t[0] for t in tr_te]); ytr = np.r_[np.zeros(len(Ntr)), np.ones(sum(len(t[0]) for t in tr_te))]
        Xte = np.vstack([Nte] + [t[1] for t in tr_te]); yte = np.r_[np.zeros(len(Nte)), np.ones(sum(len(t[1]) for t in tr_te))]
        for mn, mdl in models().items():
            accA, f1A = fit_score(mdl, Xtr, ytr, Xte, yte)
            accB = f1B = float("nan")
            if len(fr) >= 2:
                sc = []
                for h in range(len(fr)):
                    Xo = np.vstack([fr[j][2] for j in range(len(fr)) if j != h])
                    Xt = np.vstack([Ntr, Xo]); yt = np.r_[np.zeros(len(Ntr)), np.ones(len(Xo))]
                    Xh = np.vstack([Nte, fr[h][2]]); yh = np.r_[np.zeros(len(Nte)), np.ones(len(fr[h][2]))]
                    sc.append(fit_score(models()[mn], Xt, yt, Xh, yh))
                accB, f1B = np.mean([s[0] for s in sc]), np.mean([s[1] for s in sc])
            out.append({"level": lv, "model": mn, "bal_acc_timesplit": round(100 * accA, 1),
                        "f1_timesplit": round(f1A, 3),
                        "bal_acc_unseen_file": None if np.isnan(accB) else round(100 * accB, 1)})
            bs = "n/a (1 file)" if np.isnan(accB) else f"{100*accB:.1f}"
            print(f"{lv:<9}{mn:<12}{100*accA:>15.1f}{bs:>16}")

        # ablation + importances on the time split with RandomForest
        rf = models()["RandForest"]
        allc = list(range(len(FEATS)))
        no_rec = [i for i, f in enumerate(FEATS) if f not in RECON]
        only_rec = [i for i, f in enumerate(FEATS) if f in RECON]
        s_all = fit_score(rf, Xtr, ytr, Xte, yte, allc)[0]
        s_no = fit_score(models()["RandForest"], Xtr, ytr, Xte, yte, no_rec)[0]
        s_only = fit_score(models()["RandForest"], Xtr, ytr, Xte, yte, only_rec)[0]
        rf.fit(Xtr, ytr)
        top = sorted(zip(rf.feature_importances_, FEATS), reverse=True)[:5]
        print(f"   [{lv}] RandForest time-split: all features {100*s_all:.1f}% | without reconstruction "
              f"features {100*s_no:.1f}% | reconstruction features only {100*s_only:.1f}%")
        print("   top features:", ", ".join(f"{n} ({v:.2f})" for v, n in top))
        out.append({"level": lv, "model": "RF ablation (all/no-recon/recon-only)",
                    "bal_acc_timesplit": f"{100*s_all:.1f}/{100*s_no:.1f}/{100*s_only:.1f}",
                    "f1_timesplit": "", "bal_acc_unseen_file": ""})

    with open("raw_current_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    print("\nSaved: raw_current_results.csv")


if __name__ == "__main__":
    main()
