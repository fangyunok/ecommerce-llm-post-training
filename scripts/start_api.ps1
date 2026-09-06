$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path "$PSScriptRoot\..").Path
Set-Location $projectRoot
$env:PYTHONPATH = $projectRoot
$env:PYTHONUTF8 = "1"
$env:HF_HOME = if ($env:HF_HOME) {
    $env:HF_HOME
} else {
    (Join-Path $projectRoot ".cache\huggingface")
}
$apiHost = if ($env:HOST) { $env:HOST } else { "127.0.0.1" }
$apiPort = if ($env:PORT) { $env:PORT } else { "8000" }

python -m uvicorn src.ecommerce_llm.app:app `
    --host $apiHost `
    --port $apiPort

