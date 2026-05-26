param()

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$LogsDir = Join-Path $ProjectRoot "Logs"
$Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"

New-Item -ItemType Directory -Force -Path $LogsDir | Out-Null
. (Join-Path $PSScriptRoot "..\Shared\start_helpers.ps1")

Write-Host "[Cloud] Project root: $ProjectRoot"
Write-Host "[Cloud] Logs directory: $LogsDir"

$Python = Resolve-Python
$configJson = & $Python -c 'import json; from Src.Deploy.deploy_config import DEFAULT as c; from Src.Deploy.Shared.bandwidth_iperf import IPERF_EXE; print(json.dumps(dict(cloud_iperf_port=c.cloud_iperf_port, cloud_feature_port=c.cloud_feature_port, cloud_status_port=c.cloud_status_port, iperf_exe=IPERF_EXE)))'
if ($LASTEXITCODE -ne 0) {
    throw "Failed to load deploy config with Python: $Python"
}
$Config = $configJson | ConvertFrom-Json

Start-LoggedProcess `
    -Name "cloud_iperf" `
    -FilePath $Config.iperf_exe `
    -Arguments @("-s", "-p", "$($Config.cloud_iperf_port)") `
    -Ports @([int]$Config.cloud_iperf_port)

Start-LoggedProcess `
    -Name "cloud_service" `
    -FilePath $Python `
    -Arguments @("-m", "Src.Deploy.Cloud.run_cloud") `
    -Ports @([int]$Config.cloud_feature_port, [int]$Config.cloud_status_port)

Show-StartupDone `
    -Role "Cloud" `
    -StopCommand "Src\Deploy\Cloud\stop_cloud.ps1"
