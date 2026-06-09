param()

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$LogsDir = Join-Path $ProjectRoot "Logs"
$Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"

New-Item -ItemType Directory -Force -Path $LogsDir | Out-Null
. (Join-Path $PSScriptRoot "..\Shared\start_helpers.ps1")

Write-Host "[Device] Project root: $ProjectRoot"
Write-Host "[Device] Logs directory: $LogsDir"

$Python = Resolve-Python

Start-LoggedProcess `
    -Name "device_run" `
    -FilePath $Python `
    -Arguments @("-m", "Src.Deploy.Device.run_device")

Show-StartupDone `
    -Role "Device" `
    -StopCommand "Src\Deploy\Device\stop_device.ps1"
