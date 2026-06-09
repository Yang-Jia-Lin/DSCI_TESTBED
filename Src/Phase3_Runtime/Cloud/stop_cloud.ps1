param()

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..\..")

function Resolve-Python {
    if ($env:PYTHON) {
        return $env:PYTHON
    }
    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCmd) {
        return $pythonCmd.Source
    }
    $bundledPython = Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
    if (Test-Path $bundledPython) {
        return $bundledPython
    }
    throw "Python was not found. Set the PYTHON environment variable or add python to PATH."
}

function Stop-ProcessId {
    param([int]$ProcessId)

    $process = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
    if (-not $process) {
        return
    }

    Write-Host "[STOP] pid=$ProcessId name=$($process.ProcessName)"
    Stop-Process -Id $ProcessId -Force
}

function Stop-ListeningPorts {
    param([int[]]$Ports)

    $pids = @()
    foreach ($port in $Ports) {
        $connections = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
        $pids += $connections | Select-Object -ExpandProperty OwningProcess
    }

    $pids | Sort-Object -Unique | ForEach-Object { Stop-ProcessId $_ }
}

function Stop-ModuleProcess {
    param([string]$ModuleName)

    Get-CimInstance Win32_Process |
        Where-Object { $_.CommandLine -like "*$ModuleName*" } |
        ForEach-Object { Stop-ProcessId ([int]$_.ProcessId) }
}

$Python = Resolve-Python
$configJson = & $Python -c 'import json; from Src.Deploy.deploy_config import DEFAULT as c; print(json.dumps(dict(cloud_iperf_port=c.cloud_iperf_port, cloud_feature_port=c.cloud_feature_port, cloud_status_port=c.cloud_status_port)))'
if ($LASTEXITCODE -ne 0) {
    throw "Failed to load deploy config with Python: $Python"
}
$Config = $configJson | ConvertFrom-Json

Stop-ListeningPorts @(
    [int]$Config.cloud_iperf_port,
    [int]$Config.cloud_feature_port,
    [int]$Config.cloud_status_port
)
Stop-ModuleProcess "Src.Deploy.Cloud.run_cloud"

Write-Host "Cloud processes stopped."
