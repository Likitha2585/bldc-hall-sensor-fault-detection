"""
Extra model comparison for Hall-sensor fault detection (normal vs faulty).

WHAT IS NEW vs BLDC_Hall_Detection_full.py
  1. 5 newer backbones added to the original 8:
       MobileNetV3Small, MobileNetV3Large, EfficientNetV2B0, ConvNeXtTiny, NASNetMobile
  2. EVERY model gets its OWN correct input preprocessing (the old run fed all models
     MobileNet-style scaling; EfficientNet/ConvNeXt/V3 expect raw 0-255 pixels).
     So the original 8 are re-scored fairly too.
  3. Each model/condition is repeated over several random splits (seeds) and reported as
     mean +- std, because one 48-image test set is too small to rank models.
  4. Reports parameters and single-image latency (the paper's other two criteria).

SAME AS BEFORE: frozen ImageNet backbone -> Dense(1024, relu) -> Dense(2, softmax),
Adam, EarlyStopping(patience 5, restore best), batch 16, 80/20 stratified split,
majority-class undersampling to match the no_delay count.
DIFFERENT (for speed): backbone features are computed once and cached, so the head trains
in seconds and many seeds are affordable. No on-the-fly augmentation (the dataset's
'augmented' images are already augmented by its authors).

Outputs (new files only):
  new_methods_runs.csv, new_methods_summary.csv, feature_cache/
"""
import os, glob, time, csv, json, argparse
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

A = tf.keras.applications
CANDIDATE_DIRS = [
    "C:/Users/likit/BLDC_Dataset_Backup/data_in_RGB_after_preprocessing",
    "D:/Likitha Project/data_in_RGB_after_preprocessing",
    "D:/data_in_RGB_after_preprocessing",
]
CONDS = ["0.0001", "0.005", "0.01"]
POOL = 300            # fault images kept per condition (each seed draws 120 of them)
CACHE = "feature_cache"

# name: (constructor, preprocess_input, is_new)
MODELS = {
    "MobileNetV1":      ("MobileNet",        A.mobilenet.preprocess_input,        False),
    "MobileNetV2":      ("MobileNetV2",      A.mobilenet_v2.preprocess_input,     False),
    "VGG16":            ("VGG16",            A.vgg16.preprocess_input,            False),
    "ResNet50":         ("ResNet50",         A.resnet50.preprocess_input,         False),
    "DenseNet121":      ("DenseNet121",      A.densenet.preprocess_input,         False),
    "InceptionV3":      ("InceptionV3",      A.inception_v3.preprocess_input,     False),
    "Xception":         ("Xception",         A.xception.preprocess_input,         False),
    "EfficientNetB0":   ("EfficientNetB0",   A.efficientnet.preprocess_input,     False),
    "MobileNetV3Small": ("MobileNetV3Small", A.mobilenet_v3.preprocess_input,     True),
    "MobileNetV3Large": ("MobileNetV3Large", A.mobilenet_v3.preprocess_input,     True),
    "EfficientNetV2B0": ("EfficientNetV2B0", A.efficientnet_v2.preprocess_input,  True),
    "ConvNeXtTiny":     ("ConvNeXtTiny",     A.convnext.preprocess_input,         True),
    "NASNetMobile":     ("NASNetMobile",     A.nasnet.preprocess_input,           True),
}


def find_rgb_dir():
    for d in CANDIDATE_DIRS:
        if os.path.isdir(d):
            return d
    raise SystemExit("Could not find data_in_RGB_after_preprocessing. Edit CANDIDATE_DIRS at the top "
                     "of this file to the folder that contains no_delay, 0.0001, 0.005, 0.01.")


def list_images(folder):
    files = []
    for ext in ("png", "jpg", "jpeg", "bmp"):
        files += glob.glob(f"{folder}/*.{ext}")
    return sorted(files)


def load_batch(paths):
    return np.stack([np.asarray(Image.open(p).convert("RGB").resize((224, 224)), dtype=np.float32)
                     for p in paths])


def build_backbone(name):
    ctor = getattr(A, MODELS[name][0])
    return ctor(weights="imagenet", include_top=False, input_shape=(224, 224, 3), pooling="avg")


def extract(name, paths, tag):
    """Backbone features for `paths`, cached on disk. Also stores params + latency."""
    os.makedirs(CACHE, exist_ok=True)
    fp = f"{CACHE}/{name}_{tag}.npy"
    meta_fp = f"{CACHE}/{name}_meta2.json"
    if os.path.exists(fp) and os.path.exists(meta_fp):
        return np.load(fp), json.load(open(meta_fp))
    bb = build_backbone(name)
    pre = MODELS[name][1]
    if os.path.exists(meta_fp):
        meta = json.load(open(meta_fp))
    else:
        x1 = tf.constant(pre(load_batch(paths[:1]).copy()))
        run1 = tf.function(lambda x: bb(x, training=False))   # compiled, not eager
        for _ in range(5):
            run1(x1)
        t0 = time.time()
        for _ in range(20):
            run1(x1).numpy()
        meta = {"params_M": round(bb.count_params() / 1e6, 2),
                "latency_ms": round((time.time() - t0) / 20 * 1000, 1)}
        json.dump(meta, open(meta_fp, "w"))
    feats = []
    for i in range(0, len(paths), 32):
        x = pre(load_batch(paths[i:i + 32]).copy())
        feats.append(bb.predict(x, batch_size=16, verbose=0))
    feats = np.concatenate(feats)
    np.save(fp, feats)
    tf.keras.backend.clear_session()
    return feats, meta


def train_head(Xtr, ytr, Xte, yte, seed):
    tf.keras.utils.set_random_seed(seed)
    Xt, Xv, yt, yv = train_test_split(Xtr, ytr, test_size=0.2, stratify=ytr, random_state=seed)
    head = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(Xtr.shape[1],)),
        tf.keras.layers.Dense(1024, activation="relu"),
        tf.keras.layers.Dense(2, activation="softmax")])
    head.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    head.fit(Xt, yt, validation_data=(Xv, yv), epochs=30, batch_size=16, verbose=0,
             callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5,
                                                         restore_best_weights=True)])
    pred = head.predict(Xte, verbose=0).argmax(1)
    return accuracy_score(yte, pred), f1_score(yte, pred, average="macro")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="1 new model, 1 condition, 2 seeds (path/sanity check)")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--conds", nargs="*", default=None)
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    models = a.models or list(MODELS)
    conds = a.conds or CONDS
    seeds = a.seeds
    if a.quick:
        models, conds, seeds = ["MobileNetV3Small"], ["0.005"], 2

    root = find_rgb_dir()
    print("Using images from:", root)
    normal = list_images(f"{root}/no_delay/augmented")
    if not normal:
        raise SystemExit(f"No images found in {root}/no_delay/augmented")
    n = len(normal)
    rng0 = np.random.default_rng(12345)
    pools = {}
    for c in conds:
        f = list_images(f"{root}/{c}/augmented")
        if not f:
            raise SystemExit(f"No images found in {root}/{c}/augmented")
        pools[c] = [f[i] for i in rng0.permutation(len(f))[:POOL]]
        print(f"  {c}: {len(f)} fault images (pool {len(pools[c])}), no_delay: {n}")

    runs = []
    for m in models:
        for c in conds:
            paths = normal + pools[c]
            t0 = time.time()
            feats, meta = extract(m, paths, f"{c}_p{POOL}_n{n}")
            fn, ff = feats[:n], feats[n:]
            accs, f1s = [], []
            for s in range(seeds):
                r = np.random.default_rng(s)
                sel = r.permutation(len(ff))[:n]
                X = np.concatenate([fn, ff[sel]]); y = np.array([0] * n + [1] * n)
                Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=s)
                acc, f1 = train_head(Xtr, ytr, Xte, yte, s)
                accs.append(acc); f1s.append(f1)
                runs.append({"model": m, "new": MODELS[m][2], "cond": c, "seed": s,
                             "acc": round(acc, 4), "f1_macro": round(f1, 4)})
            print(f"{m:<17}{c:<8} acc {100*np.mean(accs):5.1f} +- {100*np.std(accs):4.1f}%  "
                  f"({time.time()-t0:.0f}s, {meta['params_M']}M params, {meta['latency_ms']} ms)")

    with open("new_methods_runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(runs[0])); w.writeheader(); w.writerows(runs)

    summ = []
    for m in models:
        meta = json.load(open(f"{CACHE}/{m}_meta2.json"))
        per = {c: [r["acc"] for r in runs if r["model"] == m and r["cond"] == c] for c in conds}
        row = {"model": m + ("*" if MODELS[m][2] else "")}
        for c in conds:
            row[f"acc_{c}"] = f"{100*np.mean(per[c]):.1f}+-{100*np.std(per[c]):.1f}"
        row["avg_acc"] = round(100 * float(np.mean([np.mean(v) for v in per.values()])), 1)
        row["params_M"], row["latency_ms"] = meta["params_M"], meta["latency_ms"]
        summ.append(row)
    summ.sort(key=lambda r: -r["avg_acc"])
    with open("new_methods_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0])); w.writeheader(); w.writerows(summ)

    print(f"\n=== SUMMARY: test accuracy %, mean+-std over {seeds} seeds  (* = new model; 50% = chance) ===")
    hdr = f"{'model':<19}" + "".join(f"{c:>12}" for c in conds) + f"{'avg':>7}{'paramsM':>9}{'ms/img':>8}"
    print(hdr)
    for r in summ:
        print(f"{r['model']:<19}" + "".join(f"{r['acc_'+c]:>12}" for c in conds)
              + f"{r['avg_acc']:>7}{r['params_M']:>9}{r['latency_ms']:>8}")
    print("\nSaved: new_methods_runs.csv, new_methods_summary.csv")


if __name__ == "__main__":
    main()
