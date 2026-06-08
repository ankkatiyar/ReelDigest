# ReelDigest launcher - loads .env and starts the server using the venv Python
# Usage: .\start.ps1

$envFile = Join-Path $PSScriptRoot ".env"
if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        if ($_ -match '^\s*([^#=][^=]*?)\s*=\s*(.*)\s*$') {
            [System.Environment]::SetEnvironmentVariable($matches[1].Trim(), $matches[2].Trim())
        }
    }
    Write-Host "Loaded .env"
} else {
    Write-Host "No .env file found - using existing environment variables"
}

$python = Join-Path $PSScriptRoot "venv\Scripts\python.exe"
$server = Join-Path $PSScriptRoot "server.py"
& $python $server
