# BLDC Hall-Sensor Fault Detection

Deep learning pipeline for detecting Hall-sensor displacement faults in BLDC motors from phase-current data, using transfer learning across 8 CNN architectures.

Built as an adaptation of the reference implementation from [yklee1014/BLDC-Fault-Detection](https://github.com/yklee1014/BLDC-Fault-Detection), applied to the [IEEE DataPort BLDC-HSD dataset](https://ieeexplore.ieee.org/document/10707261).

## What this does

The dataset contains motor phase-current images (already converted from raw signals to 224x224 RGB images) under 4 conditions: normal operation (`no_delay`) and three Hall-sensor delay severities (`0.0001s`, `0.005s`, `0.01s`). This project trains and compares 8 pretrained CNN architectures (MobileNetV1/V2, VGG16, ResNet50, DenseNet121, InceptionV3, Xception, EfficientNetB0) as binary classifiers (normal vs. faulty) for each severity level.

## Key result

**MobileNetV1 was the best-performing and most efficient model**, averaging 74.3% test accuracy across all three fault severities, and training in under 90 seconds per run — outperforming every larger architecture tested. Full results are in [`results_summary.csv`](./results_summary.csv).

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

**Note:** ResNet50, VGG16, and EfficientNetB0's low scores are likely attributable to a preprocessing mismatch (all models were fed MobileNet-specific input normalization, inherited from the reference implementation) rather than a genuine architectural limitation. See the full report for details.

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

# Full 8-model x 3-condition comparison (results saved to results_summary.csv)
python BLDC_Hall_Detection_full.py
```

Edit the `QUICK_TEST` flag near the top of `BLDC_Hall_Detection_full.py` to switch between a fast sanity check and the full comparison run.

## Dataset

Not included in this repo due to size. Download from [IEEE DataPort](https://ieeexplore.ieee.org/document/10707261) and update the `BASE_DATA_DIR` path in the scripts to point to your local copy.

## Methodology highlights

- **Class imbalance correction**: the raw dataset has ~120 normal images vs. ~715+ fault images per condition. Random undersampling of the majority class was used to ensure balanced training batches.
- **Transfer learning**: each model uses a frozen ImageNet-pretrained backbone with a new trainable classification head (`GlobalAveragePooling2D` → `Dense(1024)` → `Dense(2, softmax)`).
- **Data augmentation**: random flip, rotation, and zoom applied during training only.
- **Best-checkpoint selection**: `EarlyStopping` with `restore_best_weights=True` ensures the reported model is the best-performing epoch, not just the final one.

## Limitations

- Small dataset per condition (~240 images after balancing).
- Preprocessing was not corrected per-architecture (see note above).
- All training was CPU-only.

See the full project report for a complete write-up of methodology, results, and limitations.
