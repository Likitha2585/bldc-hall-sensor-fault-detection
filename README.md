# BLDC Hall-Sensor Fault Detection

Identification of Hall-sensor displacement in a BLDC motor drive from phase-current data, comparing
image-based CNNs, classical classifiers on raw-current features, and a single-current start-up block.

Data: [IEEE DataPort BLDC-HSD dataset](https://ieeexplore.ieee.org/document/10707261) (not included in this
repo because of its size). The CNN pipeline is adapted from
[yklee1014/BLDC-Fault-Detection](https://github.com/yklee1014/BLDC-Fault-Detection).

## Key findings

1. **The first CNN results were too optimistic.** The reference pipeline passed the *test* set as the validation
   set, so the best epoch was chosen on the test images. It also fed every network MobileNet-style input scaling.
   The first run reported MobileNetV1 at 74.3%.
2. **Corrected protocol** (per-model preprocessing, separate validation split, 5 random splits): all 13 CNNs fall to
   **48.6-59.7%** average accuracy, close to the 50% chance level. Best: EfficientNetB0 at 59.7%.
   MobileNetV1 drops from 74.3% to 48.6%.
3. **Current-based features** (23 features from the raw phase currents): SVM and random forest reach 100% balanced
   accuracy on the evaluated tests. Caveat: only one healthy recording, 300 ms recordings, and possibly simulated
   data, so this needs validation on more and measured data.
4. **Single-current start-up block** (measures `ia`, generates `ib` and `ic`): about 5-7% steady-state error on the
   healthy recording after roughly 59 ms. It cannot work during the initial inrush surge, so it uses an open-loop
   start and switches to tracking once the motor is rotating. Its error grows with fault size.

### Corrected CNN results (mean test accuracy, %)

| Model | 0.0001 s | 0.005 s | 0.01 s | Avg |
| --- | --- | --- | --- | --- |
| EfficientNetB0 | 58.8 | 60.0 | 60.4 | 59.7 |
| ConvNeXt-Tiny | 54.2 | 57.1 | 62.1 | 57.8 |
| EfficientNetV2-B0 | 55.8 | 58.3 | 58.3 | 57.5 |
| ResNet50 | 52.5 | 52.9 | 54.6 | 53.3 |
| MobileNetV3-Small | 49.6 | 55.0 | 52.5 | 52.4 |
| MobileNetV3-Large | 52.5 | 52.9 | 51.3 | 52.2 |
| VGG16 | 52.1 | 51.7 | 52.1 | 51.9 |
| DenseNet121 | 49.2 | 53.3 | 52.5 | 51.7 |
| NASNetMobile | 48.8 | 50.4 | 52.5 | 50.6 |
| InceptionV3 | 47.5 | 52.5 | 49.6 | 49.9 |
| Xception | 49.2 | 51.3 | 47.5 | 49.3 |
| MobileNetV2 | 47.9 | 49.2 | 50.0 | 49.0 |
| MobileNetV1 | 47.1 | 48.3 | 50.4 | 48.6 |

The original (leaky) numbers are kept for comparison in `results/results_summary.csv`.

## Repository layout

```
src/training/        CNN training scripts and data_to_RGB converters
src/startup_block/   single-current start-up block and its real-data test
dashboard/           Streamlit dashboard with the corrected results
results/             result tables (CSV) and plots
exploration/         early experiments and checks (kept for reference)
reports/             plain-language report and explanations
reference/           papers and dataset descriptor used as references
archive/             superseded files (old reports, old dashboard)
```

## Setup

```
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Set `BASE_DATA_DIR` at the top of the training scripts to your local copy of the dataset.

## Usage (run from the repo root)

```
# CNN comparison
python src/training/BLDC_Hall_Detection_full_v2.py

# Single-current start-up block on the real recording
python src/startup_block/single_current_startup_real_data.py

# Results dashboard
python -m streamlit run dashboard/app.py
```

## Limitations

- Recordings are 300 ms long and there is a single healthy recording.
- Test sets are small (48 images, or about 3-4 independent windows).
- CNN backbones are frozen, not fine-tuned.
- The dataset may be simulated, so conclusions should be confirmed on measured currents.
