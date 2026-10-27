# Run the project's Python with caches kept inside the project.
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw 'Create the project .venv and install dependencies first (see README.md).'
}
$env:HF_HOME = Join-Path $projectRoot '.cache\huggingface'
$env:MPLCONFIGDIR = Join-Path $projectRoot '.cache\matplotlib'
$env:WIN_PD_OVERRIDE_LOCAL_APPDATA = Join-Path $projectRoot '.cache\localappdata'
if (-not $env:PYTENSOR_FLAGS) {
    $env:PYTENSOR_FLAGS = 'base_compiledir=' + (Join-Path $projectRoot '.cache\pytensor')
}
$env:PYTHONIOENCODING = 'utf-8'
& $pythonPath @args
exit $LASTEXITCODE
