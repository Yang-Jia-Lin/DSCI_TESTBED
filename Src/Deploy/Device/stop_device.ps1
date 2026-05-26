param()

$ErrorActionPreference = "Stop"

function Stop-ProcessId {
    param([int]$ProcessId)

    $process = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
    if (-not $process) {
        return
    }

    Write-Host "[STOP] pid=$ProcessId name=$($process.ProcessName)"
    Stop-Process -Id $ProcessId -Force
}

Get-CimInstance Win32_Process |
    Where-Object { $_.CommandLine -like "*Src.Deploy.Device.run_device*" } |
    ForEach-Object { Stop-ProcessId ([int]$_.ProcessId) }

Write-Host "Device processes stopped."
