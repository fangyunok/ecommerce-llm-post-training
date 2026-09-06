$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path "$PSScriptRoot\..").Path
$env:PYTHONPATH = $projectRoot
$env:PYTHONUTF8 = "1"
Set-Location $projectRoot

python -m src.training.train_qlora `
    --model "Qwen/Qwen2.5-0.5B-Instruct" `
    --epochs 3 `
    --learning-rate 0.0002 `
    --batch-size 2 `
    --gradient-accumulation 8 `
    --max-length 1024 `
    --lora-r 16 `
    --lora-alpha 32

