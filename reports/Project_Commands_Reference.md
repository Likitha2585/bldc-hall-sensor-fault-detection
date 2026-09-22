# BLDC Fault Detection — Command Reference

All commands needed to set up and run every part of this project, in order. Run everything from PowerShell.

---

## 1. One-Time Environment Setup

Only needed once per machine.

```powershell
cd "C:\Users\likit\BLDC-Fault-Detection-main"

# Create a Python 3.12 virtual environment (only if not already created)
py -3.12 -m venv venv

# Install all required packages
.\venv\Scripts\Activate.ps1
python -m pip install tensorflow scikit-learn numpy pillow openpyxl scipy pandas matplotlib flask
```

---

## 2. Every Time You Open a New Terminal

Required before running anything else in a fresh PowerShell window.

```powershell
cd "C:\Users\likit\BLDC-Fault-Detection-main"
.\venv\Scripts\Activate.ps1
```

Your prompt should show `(venv)` at the start once this worked.

---

## 3. Running the CNN Fault-Detection Pipeline

### 3.1 Quick single-condition test
Fast sanity check — trains one model (MobileNetV1) on one fault condition (0.0001s).

```powershell
python BLDC_Hall_Detection.py
```

### 3.2 Full comparison (all 8 models × 3 conditions)
Takes roughly 1.5–2 hours. Writes/updates `results_summary.csv`.

```powershell
python BLDC_Hall_Detection_full.py
```

To switch between a quick test and the full run, edit the `QUICK_TEST` line near the top of `BLDC_Hall_Detection_full.py` (`True` = quick test, `False` = full run):

```powershell
(Get-Content .\BLDC_Hall_Detection_full.py) -replace 'QUICK_TEST = True', 'QUICK_TEST = False' | Set-Content .\BLDC_Hall_Detection_full.py
```

---

## 4. Running the Sensor Reconstruction Experiment

Tests whether phase currents `ib`/`ic` can be reconstructed from `ia` alone, across all 4 conditions. Takes a few seconds. Writes/updates `phase_reconstruction_results.csv` and 4 plot images.

```powershell
python phase_reconstruction_experiment.py
```

---

## 5. Running the Live Results Dashboard

Reads `results_summary.csv` and `phase_reconstruction_results.csv` fresh from disk every time the page loads — **if you rerun either experiment above and the CSV changes, just refresh your browser and the dashboard updates automatically.** No restart needed.

```powershell
python app.py
```

Then open this in your browser (Chrome, Edge, or Firefox all work):

```
http://127.0.0.1:5000
```

Leave the terminal window open while presenting — closing it stops the server. To stop it manually, click into that terminal and press:

```
Ctrl+C
```

---

## 6. Backing Up the Dataset (if not already done)

Copies the dataset off the USB drive onto the internal `C:` drive, in case the USB drive is lost or unplugged.

```powershell
robocopy "D:\Likitha Project\RAW_DATA_before_RGB" "C:\Users\likit\BLDC_Dataset_Backup\RAW_DATA_before_RGB" /E
robocopy "D:\Likitha Project\data_in_RGB_after_preprocessing" "C:\Users\likit\BLDC_Dataset_Backup\data_in_RGB_after_preprocessing" /E
```

---

## 7. Pushing Updates to GitHub

Run this after making any changes you want saved to your repository (`https://github.com/Likitha2585/bldc-hall-sensor-fault-detection`).

```powershell
git add -A
git status
git commit -m "Describe what changed here"
git push
```

Check `git status` before committing — it shows exactly what's about to be added/changed, so you can confirm nothing unwanted (like large files) is being included.

---

## Quick Reference: Typical Session

The commands you'll run most often, in order, at the start of a working session:

```powershell
cd "C:\Users\likit\BLDC-Fault-Detection-main"
.\venv\Scripts\Activate.ps1
python app.py
```

Then open `http://127.0.0.1:5000` in your browser to present the live results.
