# organize_repo.ps1
# Run from the repo root:  .\organize_repo.ps1          (preview only, changes nothing)
#                          .\organize_repo.ps1 -Apply   (actually moves the files)
# Nothing is deleted. Superseded files go to archive/ so you can always get them back.

param([switch]$Apply)

if (-not (Test-Path ".git")) { Write-Host "Run this from the repo root (the folder that contains .git)." -ForegroundColor Red; exit 1 }

function Move-Files($pattern, $dest) {
    $files = Get-ChildItem -Path . -File -Filter $pattern -ErrorAction SilentlyContinue
    foreach ($f in $files) {
        $target = Join-Path $dest $f.Name
        if ($Apply) {
            New-Item -ItemType Directory -Force -Path $dest | Out-Null
            git mv -k -- $f.Name $target 2>$null
            if (Test-Path $f.FullName) { Move-Item -LiteralPath $f.FullName -Destination $target -Force }  # untracked file
            Write-Host ("moved   {0}  ->  {1}" -f $f.Name, $dest) -ForegroundColor Green
        } else {
            Write-Host ("would move   {0}  ->  {1}" -f $f.Name, $dest)
        }
    }
}

# ---- order matters: old dashboard goes to archive BEFORE the new one is renamed ----
Move-Files "app.py"                                   "archive/old_dashboard"
Move-Files "motor_fault_dashboard*.html"              "archive"
Move-Files "bldc_results_dashboard*.html"             "archive"
Move-Files "BLDC_Fault_Detection_Report*.md"          "archive"
Move-Files "BLDC_Project_Full_Report*.docx"           "archive"

# ---- code ----
Move-Files "BLDC_Hall_Detection*.py"                  "src/training"
Move-Files "data_to_RGB*.py"                          "src/training"
Move-Files "single_current_startup*.py"               "src/startup_block"
Move-Files "startup_reconstruction*.py"               "src/startup_block"
Move-Files "check_*.py"                               "exploration"
Move-Files "phase_reconstruction*.py"                 "exploration"
Move-Files "phase_reconstruction_results.csv"         "exploration"

# ---- results, reports, references ----
Move-Files "results_summary*.csv"                     "results"
Move-Files "single_current_startup_real_results.csv"  "results"
Move-Files "single_current_startup_real.png"          "results"
Move-Files "BLDC_Project_Plain_Language_Report.pdf"   "reports"
Move-Files "BLDC_Project_Explained_Simply*.md"        "reports"
Move-Files "Optimizing_Detection_Compact_MobileNet*.pdf" "reference"
Move-Files "Descriptor_BLDC_Hall_Sensor*.pdf"         "reference"

# ---- new dashboard becomes dashboard/app.py ----
if (Test-Path "app_corrected.py") {
    if ($Apply) {
        New-Item -ItemType Directory -Force -Path "dashboard" | Out-Null
        git mv -k -- app_corrected.py dashboard/app.py 2>$null
        if (Test-Path "app_corrected.py") { Move-Item app_corrected.py dashboard/app.py -Force }
        Write-Host "moved   app_corrected.py  ->  dashboard/app.py" -ForegroundColor Green
    } else { Write-Host "would move   app_corrected.py  ->  dashboard/app.py" }
}

# ---- keep junk and personal notes out of git ----
$ignore = @("venv/", "__pycache__/", "*.pyc", "*.npy", "*.h5", "*.keras", ".DS_Store",
            "*commands_reference*", "*setup_notes*", "reports/*.docx~")
if ($Apply) {
    if (-not (Test-Path ".gitignore")) { New-Item .gitignore -ItemType File | Out-Null }
    $current = Get-Content .gitignore -ErrorAction SilentlyContinue
    foreach ($line in $ignore) { if ($current -notcontains $line) { Add-Content .gitignore $line } }
    git rm -r --cached venv __pycache__ -q 2>$null
    git add -A
    Write-Host "`nDone. Review with:  git status" -ForegroundColor Cyan
    Write-Host "Then:  git commit -m `"Organize repo into folders`"   and   git push" -ForegroundColor Cyan
} else {
    Write-Host "`nPreview only. If the list looks right, run:  .\organize_repo.ps1 -Apply" -ForegroundColor Yellow
}
