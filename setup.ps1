# setup.ps1 — One-shot setup of the virtual environment and dependencies.
#
# Run from PowerShell, in the project folder:
#     .\setup.ps1
#
# If PowerShell complains about execution policy:
#     Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
# (one-time, then re-run the setup script)

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "==> Structural Probes — environment setup" -ForegroundColor Cyan
Write-Host ""

# --- 1. Locate Python --------------------------------------------------------
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    Write-Host "ERROR: 'python' not found on PATH." -ForegroundColor Red
    Write-Host "Install Python 3.10-3.13 from https://www.python.org/downloads/ and re-run." -ForegroundColor Red
    exit 1
}
$pyVersion = & python --version 2>&1
Write-Host "Found: $pyVersion"

# --- 2. Create venv ----------------------------------------------------------
if (Test-Path .\.venv) {
    Write-Host "Existing .venv found — reusing it." -ForegroundColor Yellow
} else {
    Write-Host "Creating virtual environment in .\.venv ..."
    python -m venv .venv
}

# --- 3. Activate venv --------------------------------------------------------
Write-Host "Activating .\.venv ..."
. .\.venv\Scripts\Activate.ps1
Write-Host ("Now using: " + (& python -c "import sys; print(sys.executable)"))

# --- 4. Upgrade pip ----------------------------------------------------------
Write-Host ""
Write-Host "==> Upgrading pip ..." -ForegroundColor Cyan
python -m pip install --upgrade pip --quiet

# --- 5. Install torch (CPU build) -------------------------------------------
Write-Host ""
Write-Host "==> Installing torch (CPU build) ..." -ForegroundColor Cyan
pip install --index-url https://download.pytorch.org/whl/cpu torch --quiet

# --- 6. Install the rest of requirements ------------------------------------
Write-Host ""
Write-Host "==> Installing project requirements ..." -ForegroundColor Cyan
pip install -r requirements.txt --quiet

# --- 7. Verify imports -------------------------------------------------------
Write-Host ""
Write-Host "==> Verifying installation ..." -ForegroundColor Cyan
$verify = @"
import torch, transformers, h5py, scipy, yaml, numpy
print('torch        :', torch.__version__)
print('transformers :', transformers.__version__)
print('numpy        :', numpy.__version__)
print('h5py         :', h5py.__version__)
print('scipy        :', scipy.__version__)
print('OK — all imports succeeded.')
"@
python -c $verify

# --- 8. Done -----------------------------------------------------------------
Write-Host ""
Write-Host "==> Setup complete." -ForegroundColor Green
Write-Host ""
Write-Host "From now on, in any new PowerShell window, ACTIVATE the venv first:" -ForegroundColor Yellow
Write-Host "    .\.venv\Scripts\Activate.ps1" -ForegroundColor Yellow
Write-Host ""
Write-Host "Then run the experiments. The first commands are:" -ForegroundColor Yellow
Write-Host "    foreach (`$s in 'train','dev','test') {" -ForegroundColor Gray
Write-Host "        python -m scripts.conllu_to_text data/es_ancora/es_ancora-ud-`$s.conllu data/es_ancora/es_ancora-ud-`$s.txt" -ForegroundColor Gray
Write-Host "    }" -ForegroundColor Gray
Write-Host ""
