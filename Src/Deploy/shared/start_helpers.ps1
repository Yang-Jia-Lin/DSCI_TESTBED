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

function Get-PortOwners {
    param([int]$Port)

    $connections = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    return @($connections | Select-Object -ExpandProperty OwningProcess -Unique)
}

function Test-PortListening {
    param([int]$Port)

    return ((Get-PortOwners $Port).Count -gt 0)
}

function Wait-PortListening {
    param(
        [int]$Port,
        [int]$TimeoutSeconds = 8
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        if (Test-PortListening $Port) {
            return $true
        }
        Start-Sleep -Milliseconds 300
    }
    return (Test-PortListening $Port)
}

function Show-LogTail {
    param(
        [string]$Path,
        [int]$Tail = 20
    )

    if ((Test-Path $Path) -and ((Get-Item $Path).Length -gt 0)) {
        Write-Host "----- $Path tail -----"
        Get-Content $Path -Tail $Tail
        Write-Host "----------------------"
    }
}

function Start-LoggedProcess {
    param(
        [string]$Name,
        [string]$FilePath,
        [string[]]$Arguments,
        [int[]]$Ports = @()
    )

    if ([string]::IsNullOrWhiteSpace($FilePath)) {
        throw "$Name FilePath is empty."
    }

    foreach ($port in $Ports) {
        $owners = Get-PortOwners $port
        if ($owners.Count -gt 0) {
            Write-Host "[SKIP] ${Name}: port $port is already listening. pid=$($owners -join ',')"
            return
        }
    }

    $stdout = Join-Path $LogsDir "$Timestamp-$Name.out.log"
    $stderr = Join-Path $LogsDir "$Timestamp-$Name.err.log"

    Write-Host "[STARTING] $Name"
    Write-Host "           command: $FilePath $($Arguments -join ' ')"
    Write-Host "           stdout : $stdout"
    Write-Host "           stderr : $stderr"

    $process = Start-Process `
        -FilePath $FilePath `
        -ArgumentList $Arguments `
        -WorkingDirectory $ProjectRoot `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -WindowStyle Hidden `
        -PassThru

    Start-Sleep -Milliseconds 700
    $alive = $null -ne (Get-Process -Id $process.Id -ErrorAction SilentlyContinue)
    if (-not $alive) {
        Write-Host "[FAILED] $Name exited immediately. pid=$($process.Id)"
        Show-LogTail $stderr
        Show-LogTail $stdout
        return
    }

    Write-Host "[RUNNING] $Name pid=$($process.Id)"

    foreach ($port in $Ports) {
        if (Wait-PortListening $port) {
            $owners = Get-PortOwners $port
            Write-Host "[OK] ${Name}: port $port listening. pid=$($owners -join ',')"
        } else {
            Write-Host "[WARN] ${Name}: port $port is not listening yet. Check logs:"
            Write-Host "       stdout: $stdout"
            Write-Host "       stderr: $stderr"
            Show-LogTail $stderr
        }
    }
}

function Show-StartupDone {
    param(
        [string]$Role,
        [string]$StopCommand
    )

    Write-Host ""
    Write-Host "[$Role] startup check complete."
    Write-Host "Logs directory: $LogsDir"
    Write-Host "Stop command  : $StopCommand"
}
