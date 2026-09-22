# BLDC Hall-Sensor Fault Detection

Deep learning pipeline for detecting Hall-sensor displacement faults in BLDC motors from phase-current data, using transfer learning across 8 CNN architectures.

Built as an adaptation of the reference implementation from [yklee1014/BLDC-Fault-Detection](https://github.com/yklee1014/BLDC-Fault-Detection), applied to the [IEEE DataPort BLDC-HSD dataset](https://ieeexplore.ieee.org/document/10707261).

## What this does

The dataset contains motor phase-current images (already converted from raw signals to 224x224 RGB images) under 4 conditions: normal operation (`no_delay`) and three Hall-sensor delay severities (`0.0001s`, `0.005s`, `0.01s`). This project trains and compares 8 pretrained CNN architectures (MobileNetV1/V2, VGG16, ResNet50, DenseNet121, InceptionV3, Xception, EfficientNetB0) as binary classifiers (normal vs. faulty) for each severity level.

## Key result

Two runs are included, kept separately on purpose so the effect of a real bug fix is fully visible.

### Run 1 (original) — all models fed MobileNet-specific preprocessing

Inherited from the reference implementation: every model, regardless of architecture, was fed images normalized for MobileNet specifically. Full results in [`results_summary.csv`](./results_summary.csv).

| Model | 0.0001s | 0.005s | 0.01s | Average |
|---|---|---|---|---|
| **MobileNetV1** | 72.9% | **79.2%** | 70.8% | **74.3%** |
| InceptionV3 | 64.6% | 68.8% | 66.7% | 66.7% |
| Xception | 60.4% | 66.7% | 68.8% | 65.3% |
| DenseNet121 | 66.7% | 62.5% | 56.2% | 61.8% |
| VGG16 | 60.4% | 50.0% | 68.8% | 59.7% |
| MobileNetV2 | 60.4% | 54.2% | 64.6% | 59.7% |
| ResNet50 | 50.0% | 50.0% | 70.8% | 56.9% |
| EfficientNetB0 | 50.0% | 50.0% | 50.0% | 50.0% |

MobileNetV1 appeared to win clearly, and EfficientNetB0/ResNet50/VGG16 appeared to fail outright (50% = pure majority-class guessing).

### Run 2 (corrected) — each model given its own correct preprocessing

Diagnosed the issue above as a likely preprocessing mismatch and fixed it: each architecture is now given its own correct `preprocess_input` function (e.g. `resnet50.preprocess_input` for ResNet50, `efficientnet.preprocess_input` for EfficientNetB0, etc.). Script: [`BLDC_Hall_Detection_full_v2.py`](./BLDC_Hall_Detection_full_v2.py). Full results in [`results_summary_v2.csv`](./results_summary_v2.csv).

| Model | 0.0001s | 0.005s | 0.01s | Average | Change vs. Run 1 |
|---|---|---|---|---|---|
| **EfficientNetB0** | 72.9% | 75.0% | 72.9% | **73.6%** | **+23.6** |
| MobileNetV1 | 66.7% | 70.8% | 66.7% | 68.1% | -6.2 |
| ResNet50 | 62.5% | 68.8% | 68.8% | 66.7% | +9.8 |
| InceptionV3 | 66.7% | 68.8% | 64.6% | 66.7% | 0.0 |
| VGG16 | 68.8% | 68.8% | 60.4% | 66.0% | +6.3 |
| DenseNet121 | 54.2% | 70.8% | 68.8% | 64.6% | +2.8 |
| Xception | 52.1% | 70.8% | 62.5% | 61.8% | -3.5 |
| MobileNetV2 | 54.2% | 64.6% | 47.9% | 55.6% | -4.1 |

**The theory was confirmed, dramatically for EfficientNetB0**: it went from complete class collapse (50%, always predicting "fault") to the single best-performing model overall (73.6%) once given its correct preprocessing — a +23.6 point swing. ResNet50 and VGG16 also improved meaningfully. MobileNetV1's small drop is normal run-to-run variance from the random undersampling (only 120 of ~715+ fault images are sampled each run), not a real regression, since its preprocessing never changed between runs.

**Takeaway:** the original "MobileNetV1 wins clearly" conclusion was itself partly an artifact of the preprocessing bug. With it fixed, EfficientNetB0 and MobileNetV1 are the two strongest models, both meaningfully ahead of the rest — a good illustration of why a surprising result is worth investigating rather than taken at face value.

## Supplementary experiment: can 1 current sensor replace 3?

A second, independent experiment ([`phase_reconstruction_experiment.py`](./phase_reconstruction_experiment.py)) tests whether the two missing phase currents can be mathematically reconstructed from a single sensor (using the theoretical 120°/240° phase relationship), calibrated on healthy data and tested across all four conditions.

**Result:** reconstruction is near-perfect on healthy data but degrades monotonically and severely as fault severity increases — meaning single-sensor reconstruction cannot safely replace 3 physical sensors, since it fails specifically when a fault occurs. Full results in [`phase_reconstruction_results.csv`](./phase_reconstruction_results.csv).

| Condition | ib correlation | ic correlation |
|---|---|---|
| no_delay (healthy) | 0.998 | 0.999 |
| 0.0001s | 0.986 | 0.970 |
| 0.005s | 0.880 | 0.667 |
| 0.01s | 0.364 | 0.140 |

**Interesting side-finding:** since reconstruction error grows predictably with fault severity, the error itself could serve as a lightweight, AI-free fault indicator — a promising direction for future work alongside the CNN-based approach above.

## Setup

```powershell
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install tensorflow scikit-learn numpy pillow openpyxl
```

## Usage

```powershell
# Quick 1-model x 1-condition test
python BLDC_Hall_Detection.py

# Full 8-model x 3-condition comparison, original preprocessing (-> results_summary.csv)
python BLDC_Hall_Detection_full.py

# Full 8-model x 3-condition comparison, corrected per-model preprocessing (-> results_summary_v2.csv)
python BLDC_Hall_Detection_full_v2.py
```

Edit the `QUICK_TEST` flag near the top of either `_full.py` script to switch between a fast sanity check and the full comparison run.

Live results dashboard (reads `results_summary.csv` directly, updates automatically when you rerun the original script):

```powershell
python app.py
# then open http://127.0.0.1:5000
```

## Dataset

Not included in this repo due to size. Download from [IEEE DataPort](https://ieeexplore.ieee.org/document/10707261) and update the `BASE_DATA_DIR` path in the scripts to point to your local copy.

## Methodology highlights

- **Class imbalance correction**: the raw dataset has ~120 normal images vs. ~715+ fault images per condition. Random undersampling of the majority class was used to ensure balanced training batches.
- **Transfer learning**: each model uses a frozen ImageNet-pretrained backbone with a new trainable classification head (`GlobalAveragePooling2D` → `Dense(1024)` → `Dense(2, softmax)`).
- **Data augmentation**: random flip, rotation, and zoom applied during training only.
- **Best-checkpoint selection**: `EarlyStopping` with `restore_best_weights=True` ensures the reported model is the best-performing epoch, not just the final one.

## Limitations

- Small dataset per condition (~240 images after balancing) — causes some run-to-run variance from random undersampling.
- All training was CPU-only.
- The corrected run (Run 2) still uses a fixed 224x224 input size and standard ImageNet-pretrained backbones; no further architecture-specific tuning (e.g. input resolution, unfreezing later layers) was explored.

See the full project report for a complete write-up of methodology, results, and limitations.
