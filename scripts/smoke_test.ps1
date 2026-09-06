$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path "$PSScriptRoot\..").Path
$env:PYTHONUTF8 = "1"
python (Join-Path $projectRoot "scripts\smoke_test.py")

