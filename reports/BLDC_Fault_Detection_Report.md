# BLDC Motor Hall-Sensor Fault Detection Using Deep Learning
### Project Report

---

## 1. Project Overview

This project investigates automated fault detection for **Hall-sensor displacement faults** in Brushless DC (BLDC) motors using deep learning. Hall-sensor displacement is a real electromechanical fault that shifts the timing of a motor's commutation signal, distorting the three-phase current waveforms in a way that is difficult to detect by eye but recognizable by a trained image-classification model once those waveforms are converted into visual (RGB) representations.

The goal was to:
1. Reproduce a published reference AI pipeline on the actual published dataset.
2. Correctly adapt that reference pipeline to the real structure and class balance of the downloaded data.
3. Compare multiple deep learning architectures to identify which performs best on this task.
4. Test the fault-detection pipeline across three different fault severities.

---

## 2. Background and References

**Dataset — IEEE DataPort: BLDC Hall Sensor Displacement (BLDC-HSD)**
The dataset provides three-phase current measurements from a BLDC motor under four conditions:
- `no_delay` — normal operation (baseline)
- `0.0001` s Hall-sensor delay (mild fault)
- `0.005` s Hall-sensor delay (moderate fault)
- `0.01` s Hall-sensor delay (severe fault)

Each condition's raw current signal was pre-converted by the dataset authors into 224×224 RGB images (a standard technique for applying image-based CNNs to time-series/signal data) and further expanded via data augmentation, provided in a `data_in_RGB_after_preprocessing` folder.

**Reference implementation — GitHub: yklee1014/BLDC-Fault-Detection**
The associated GitHub repository provides:
- `data_to_RGB.py` / `data_to_RGB_onefile.py` — scripts converting raw current signals into RGB images (not required for this project, since the dataset already ships pre-converted images).
- `BLDC_Hall_Detection.py` — the reference AI pipeline: transfer learning using 8 pretrained CNN architectures (MobileNetV1, MobileNetV2, VGG16, ResNet50, DenseNet121, InceptionV3, Xception, EfficientNetB0) fine-tuned for binary fault classification.
- An associated paper: *"Optimizing Detection: Compact MobileNet Models for Precise Hall Sensor Fault Identification in BLDC Motor Drives."*

The reference script was written against the author's own private folder structure and was **not directly runnable** on the publicly downloaded dataset without modification (see Section 4).

---

## 3. Dataset Structure (as downloaded)

```
Likitha Project/
├── RAW_DATA_before_RGB/
│   ├── no_delay/        (raw phase-current Excel files)
│   ├── 0.0001/           (3 Excel files: delay_A, delay_B, delay_C)
│   ├── 0.005/
│   └── 0.01/
└── data_in_RGB_after_preprocessing/
    ├── no_delay/augmented/     → 120 images
    ├── 0.0001/augmented/       → 715 images
    ├── 0.005/augmented/        → 718 images
    └── 0.01/augmented/         → 719 images
```

The raw Excel data was not used in this project — the pre-converted RGB images were used directly, since they represent the same information already prepared for CNN input.

**Key characteristic identified during setup:** the `no_delay` (normal) class has far fewer images (120) than each fault condition (~715-719) — roughly a 6:1 class imbalance. This had to be explicitly corrected (Section 5.3), since it directly affected model performance.

---

## 4. Environment Setup

### 4.1 Initial Challenge: Python Version Incompatibility
The lab machine's default Python installation was **3.14.2**, which is too new for TensorFlow (TensorFlow officially supports Python 3.10-3.13 at time of writing). Installing TensorFlow failed with `No matching distribution found for tensorflow`.

**Fix:** Installed Python 3.12 alongside the existing installation and created a dedicated virtual environment for the project.

### 4.2 Second Challenge: Extremely Slow Package Installation
Initial `pip install` attempts, run from a project folder located on a **USB flash drive (D:)**, were abnormally slow (~0.7 MB/s disk throughput) due to USB drives being poorly suited to writing the thousands of small files that Python packages unpack into.

**Fix:** Moved the project code (not the dataset, which stayed on the USB drive) to the internal `C:` drive, and recreated the virtual environment there. This restored normal install speed.

### 4.3 Final Working Environment Setup

```powershell
# Create project folder on C: (data itself stays on the USB drive D:)
robocopy "D:\Likitha Project\BLDC-Fault-Detection-main" "C:\Users\likit\BLDC-Fault-Detection-main" /E /XD venv

cd "C:\Users\likit\BLDC-Fault-Detection-main"

# Create a Python 3.12 virtual environment
py -3.12 -m venv venv

# Activate it (required every time a new terminal is opened)
.\venv\Scripts\Activate.ps1

# Install dependencies (use python -m pip, not the bare pip command,
# to avoid it resolving to a different global Python installation)
python -m pip install tensorflow scikit-learn numpy pillow openpyxl
```

**Verification:**
```powershell
python -c "import tensorflow as tf; print(tf.__version__)"
# → 2.21.0
```

---

## 5. Methodology

### 5.1 Overall Pipeline

```
RGB images (D:\...\data_in_RGB_after_preprocessing)
        ↓
Load no_delay images + fault-condition images
        ↓
Undersample majority class for balance
        ↓
Train/test split (80/20, stratified)
        ↓
Data augmentation (random flip, rotation, zoom) — training only
        ↓
Transfer learning: pretrained CNN backbone (frozen) + new classification head
        ↓
Train with early stopping + best-checkpoint saving
        ↓
Evaluate: accuracy, precision, recall, F1, training time, inference time
```

### 5.2 Transfer Learning Approach
Each model uses a CNN architecture pretrained on ImageNet (1.2M general photos) with its original classification layers removed. The pretrained layers are **frozen** (not updated during training), preserving their general visual feature-detection ability. A small new classification head is added on top:

```
GlobalAveragePooling2D → Dense(1024, relu) → Dense(2, softmax)
```

Only this new head is trained, using the project's ~240 fault/no-fault images. This is standard practice when working with a dataset too small to train a full CNN from scratch.

### 5.3 Class Imbalance Correction
The raw dataset has ~120 `no_delay` images vs. ~715-719 fault images per condition. Two approaches were tested:

1. **Class weighting** (`class_weight` in `model.fit`) — attempted first. With `batch_size=16` and this severe an imbalance, many mini-batches contained zero or one `no_delay` example, and the model repeatedly collapsed to predicting only the majority class (0% precision/recall on `no_delay`).
2. **Random undersampling** (used in the final version) — randomly sampled a subset of fault images equal in count to the `no_delay` images (120 vs. 120), so every training batch contains a realistic mix of both classes. This produced meaningfully more balanced precision/recall across both classes and was adopted as the final approach.

### 5.4 Training Configuration
- Optimizer: Adam
- Loss: categorical cross-entropy
- Epochs: up to 20, with **EarlyStopping** (patience = 5, monitoring validation accuracy, restoring the best-performing weights rather than the final epoch's)
- **ModelCheckpoint** saving the best model per architecture/condition combination
- Batch size: 16
- Image size: 224×224×3
- Train/test split: 80/20, stratified by class

### 5.5 Full Comparison Design
The final experiment loops over:
- **3 fault conditions**: 0.0001s, 0.005s, 0.01s (each vs. the same `no_delay` baseline)
- **8 CNN architectures**: MobileNetV1, MobileNetV2, VGG16, ResNet50, DenseNet121, InceptionV3, Xception, EfficientNetB0

= **24 total training runs**, with results logged automatically to `results_summary.csv`.

---

## 6. Commands Used (Full Reproducibility Reference)

```powershell
# Navigate to project and activate environment (every new session)
cd "C:\Users\likit\BLDC-Fault-Detection-main"
.\venv\Scripts\Activate.ps1

# Run the single-condition prototype (development/debugging phase)
python BLDC_Hall_Detection.py

# Run the full 8-model x 3-condition comparison
python BLDC_Hall_Detection_full.py

# Verify file contents / confirm edits were saved before rerunning
Select-String -Path .\BLDC_Hall_Detection_full.py -Pattern 'QUICK_TEST ='
```

The final script includes a `QUICK_TEST` flag (`True`/`False`) allowing a fast 1-model × 1-condition sanity check before committing to the full ~1.5-2 hour run.

---

## 7. Results

### 7.1 Full Results Table (Test Accuracy %)

| Model | 0.0001s | 0.005s | 0.01s | **Average** | Avg. Train Time (s) |
|---|---|---|---|---|---|
| **MobileNetV1** | 72.9 | **79.2** | 70.8 | **74.3** | 79.7 |
| ResNet50 | 50.0 | 50.0 | 70.8 | 56.9 | 199.5 |
| VGG16 | 60.4 | 50.0 | 68.8 | 59.7 | 636.1 |
| MobileNetV2 | 60.4 | 54.2 | 64.6 | 59.7 | 63.4 |
| DenseNet121 | 66.7 | 62.5 | 56.2 | 61.8 | 228.4 |
| Xception | 60.4 | 66.7 | 68.8 | 65.3 | 261.9 |
| InceptionV3 | 64.6 | 68.8 | 66.7 | 66.7 | 195.7 |
| EfficientNetB0 | 50.0 | 50.0 | 50.0 | 50.0 | 81.4 |

### 7.2 Detailed Classification Metrics — Best Model (MobileNetV1)

| Condition | Accuracy | Precision (no_delay) | Recall (no_delay) | Precision (fault) | Recall (fault) |
|---|---|---|---|---|---|
| 0.0001s | 72.9% | 0.76 | 0.67 | 0.70 | 0.79 |
| 0.005s | 79.2% | 0.73 | 0.92 | 0.89 | 0.67 |
| 0.01s | 70.8% | 0.67 | 0.83 | 0.78 | 0.58 |

### 7.3 Key Findings

**Finding 1 — MobileNetV1 is the best-performing model overall**, both in accuracy (74.3% average, clearly ahead of every other architecture) and by a wide margin in training speed (under 90 seconds per run vs. 200-800+ seconds for the heavier architectures). This is a practically meaningful result: the smallest, most efficient model outperformed larger networks like VGG16 and ResNet50 on this task.

**Finding 2 — Three models (EfficientNetB0, and to a lesser extent ResNet50 and VGG16) frequently collapsed to predicting a single class**, producing exactly 50% accuracy with 0% precision/recall on the `no_delay` class in several runs. Investigation traced this to a **preprocessing mismatch**: all models in this pipeline (matching the original reference implementation) were fed images normalized using MobileNet's specific preprocessing function (`tf.keras.applications.mobilenet.preprocess_input`), rather than each architecture's own expected input normalization. EfficientNet, ResNet, and VGG each expect meaningfully different pixel scaling/normalization, and mismatched normalization is a known cause of this kind of training collapse.

**Finding 3 — Moderate fault severity (0.005s) produced the best classification results** for the top-performing model, suggesting the visual distinction between normal and fault current patterns may be clearest at this delay magnitude within this dataset — worth further investigation with a larger sample.

---

## 8. Limitations

1. **Small dataset per condition.** After balancing, each condition used only ~240 total images (192 training / 48 test). Deep learning models generally perform better with larger datasets; results should be interpreted as an initial baseline rather than a final, production-ready accuracy figure.
2. **Preprocessing mismatch across architectures.** As noted in Section 7.3, applying MobileNet-specific preprocessing uniformly to all 8 models (inherited from the reference implementation) likely disadvantaged architectures with different expected input normalization (ResNet50, VGG16, EfficientNetB0). Their true achievable accuracy may be higher than reported here.
3. **Random undersampling discards data.** Balancing by undersampling the majority class means ~595 of the ~715 fault images per condition were not used in training. An alternative such as oversampling the minority class, or collecting additional `no_delay` samples, could make fuller use of the available data.
4. **CPU-only training.** All models were trained on CPU (no CUDA-compatible GPU available on the lab machine), which limited the number of epochs/experiments practical within the project timeline.

---

## 9. Conclusion

This project successfully adapted a published reference AI pipeline to a real IEEE-published BLDC motor fault dataset, diagnosing and correcting both environment-setup obstacles (Python version compatibility, slow USB-based package installation) and a significant class-imbalance problem in the data itself. The resulting comparison across 8 CNN architectures and 3 fault severities identified **MobileNetV1 as the most accurate and most computationally efficient model for this Hall-sensor fault detection task**, achieving a 74.3% average test accuracy across all three fault conditions — a legitimate, meaningful result given the balanced (not inflated) evaluation methodology used.

## 10. Suggested Future Work

- Correct per-model preprocessing and re-run the comparison to determine whether ResNet50/VGG16/EfficientNetB0 improve substantially.
- Collect additional real `no_delay` samples from the lab motor (via Arduino/IoT sensor logging, as originally proposed) to reduce reliance on undersampling.
- Fine-tune (rather than freeze) the later layers of the best-performing model (MobileNetV1) to test whether accuracy improves further.
- Extend from binary (normal vs. one fault severity) to a single unified multi-class classifier distinguishing all three fault severities plus normal operation simultaneously.
