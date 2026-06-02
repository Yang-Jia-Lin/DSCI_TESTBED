param(
    [switch]$NoAlgo,
    [int[]]$FixedSplit,
    [double]$FixedThreshold = -1
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$LogsDir = Join-Path $ProjectRoot "Logs"
$Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"

New-Item -ItemType Directory -Force -Path $LogsDir | Out-Null
. (Join-Path $PSScriptRoot "..\Shared\start_helpers.ps1")

Write-Host "[Edge] Project root: $ProjectRoot"
Write-Host "[Edge] Logs directory: $LogsDir"

$Python = Resolve-Python
$configJson = & $Python -c 'import json; from Src.Deploy.deploy_config import DEFAULT as c; from Src.Deploy.Shared.bandwidth_iperf import IPERF_EXE; print(json.dumps(dict(edge_iperf_port=c.edge_iperf_port, edge_feature_port=c.edge_feature_port, edge_status_port=c.edge_status_port, algo_server_port=c.algo_server_port, iperf_exe=IPERF_EXE)))'
if ($LASTEXITCODE -ne 0) {
    throw "Failed to load deploy config with Python: $Python"
}
$Config = $configJson | ConvertFrom-Json

Start-LoggedProcess `
    -Name "edge_iperf" `
    -FilePath $Config.iperf_exe `
    -Arguments @("-s", "-p", "$($Config.edge_iperf_port)") `
    -Ports @([int]$Config.edge_iperf_port)

if (-not $NoAlgo) {
    $AlgoArgs = @("-m", "Src.Algorithm.Interface.api_server")
    if ($FixedSplit) {
        if ($FixedSplit.Count -ne 2) {
            throw "-FixedSplit expects exactly two integers, for example: -FixedSplit 0,1"
        }
        $AlgoArgs += @("--fixed-split", "$($FixedSplit[0])", "$($FixedSplit[1])")
    }
    if ($FixedThreshold -ge 0) {
        if ($FixedThreshold -gt 1) {
            throw "-FixedThreshold must be in [0, 1]."
        }
        $AlgoArgs += @("--fixed-threshold", "$FixedThreshold")
    }

    Start-LoggedProcess `
        -Name "algo_api" `
        -FilePath $Python `
        -Arguments $AlgoArgs `
        -Ports @([int]$Config.algo_server_port)
} else {
    Write-Host "[SKIP] algo_api: -NoAlgo was set."
}

Start-LoggedProcess `
    -Name "edge_service" `
    -FilePath $Python `
    -Arguments @("-m", "Src.Deploy.Edge.run_edge") `
    -Ports @([int]$Config.edge_feature_port, [int]$Config.edge_status_port)

Show-StartupDone `
    -Role "Edge" `
    -StopCommand "Src\Deploy\Edge\stop_edge.ps1"
