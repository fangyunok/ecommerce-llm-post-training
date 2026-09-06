$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path "$PSScriptRoot\..").Path
$env:PYTHONPATH = $projectRoot
$env:PYTHONUTF8 = "1"
Set-Location $projectRoot
python -u -m src.evaluation.evaluate_api

