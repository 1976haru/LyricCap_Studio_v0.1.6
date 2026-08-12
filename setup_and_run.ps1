param(
    [switch]$InstallOnly
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
$log = Join-Path $root 'setup_log.txt'

function Write-Step([string]$msg) {
    Write-Host ''
    Write-Host ('=' * 56) -ForegroundColor DarkCyan
    Write-Host $msg -ForegroundColor Cyan
    Write-Host ('=' * 56) -ForegroundColor DarkCyan
}

function Test-PythonVersion([string]$exe, [string[]]$prefixArgs) {
    try {
        $code = 'import sys; raise SystemExit(0 if sys.version_info[:2] in [(3,11),(3,12)] else 1)'
        & $exe @prefixArgs -c $code *> $null
        return ($LASTEXITCODE -eq 0)
    } catch {
        return $false
    }
}

function Find-CompatiblePython {
    $candidates = @(
        (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe'),
        (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python311\python.exe'),
        (Join-Path $env:ProgramFiles 'Python312\python.exe'),
        (Join-Path $env:ProgramFiles 'Python311\python.exe')
    )
    if (${env:ProgramFiles(x86)}) {
        $candidates += (Join-Path ${env:ProgramFiles(x86)} 'Python312\python.exe')
        $candidates += (Join-Path ${env:ProgramFiles(x86)} 'Python311\python.exe')
    }

    foreach ($exe in $candidates) {
        if ($exe -and (Test-Path $exe) -and (Test-PythonVersion $exe @())) {
            return @{ Exe = $exe; Args = @() }
        }
    }

    $py = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($py) {
        if (Test-PythonVersion $py.Source @('-3.12')) {
            return @{ Exe = $py.Source; Args = @('-3.12') }
        }
        if (Test-PythonVersion $py.Source @('-3.11')) {
            return @{ Exe = $py.Source; Args = @('-3.11') }
        }
    }

    $python = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($python -and (Test-PythonVersion $python.Source @())) {
        return @{ Exe = $python.Source; Args = @() }
    }
    return $null
}

function Install-PythonIfNeeded {
    $py = Find-CompatiblePython
    if ($py) { return $py }

    Write-Step 'Python 3.12 is not installed. Automatic install will start.'
    $winget = Get-Command winget.exe -ErrorAction SilentlyContinue
    if ($winget) {
        Write-Host '[1/1] Installing Python 3.12 with Windows Package Manager...' -ForegroundColor Yellow
        & $winget.Source install --id Python.Python.3.12 --exact --scope user --accept-package-agreements --accept-source-agreements --disable-interactivity
        if ($LASTEXITCODE -ne 0) {
            Write-Host '[WARNING] winget could not complete the Python install.' -ForegroundColor Yellow
        }
        Start-Sleep -Seconds 2
        $py = Find-CompatiblePython
        if ($py) {
            Write-Host '[OK] Python 3.12 installed.' -ForegroundColor Green
            return $py
        }
    }

    Write-Host ''
    Write-Host '[ACTION REQUIRED] Automatic Python installation was not available.' -ForegroundColor Yellow
    Write-Host 'The official Python Windows download page will open.'
    Write-Host 'Install Python 3.12 and CHECK "Add python.exe to PATH".'
    Start-Process 'https://www.python.org/downloads/windows/'
    Write-Host ''
    Read-Host 'After Python installation finishes, press ENTER here'
    $py = Find-CompatiblePython
    if (-not $py) {
        throw 'Python 3.11/3.12 is still not detected. Close this window, install Python 3.12, then run START_HERE.bat again.'
    }
    return $py
}

function Ensure-Venv($py) {
    $venvPython = Join-Path $root '.venv\Scripts\python.exe'
    if (-not (Test-Path $venvPython)) {
        Write-Step 'Creating LyricCap private Python environment'
        & $py.Exe @($py.Args) -m venv (Join-Path $root '.venv')
        if ($LASTEXITCODE -ne 0) { throw 'Could not create the Python virtual environment.' }
    } else {
        Write-Host '[OK] Existing private Python environment found.' -ForegroundColor Green
    }
    return $venvPython
}

function Install-Dependencies([string]$venvPython) {
    Write-Step 'Installing LyricCap packages'
    & $venvPython -m pip install --upgrade pip setuptools wheel
    if ($LASTEXITCODE -ne 0) { throw 'pip update failed.' }
    & $venvPython -m pip install -r (Join-Path $root 'requirements.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Python package installation failed.' }
}

function Ensure-FFmpeg {
    if (Get-Command ffmpeg.exe -ErrorAction SilentlyContinue) {
        Write-Host '[OK] FFmpeg found.' -ForegroundColor Green
        return
    }

    Write-Step 'FFmpeg is not installed. Automatic install will start.'
    $winget = Get-Command winget.exe -ErrorAction SilentlyContinue
    if ($winget) {
        & $winget.Source install --id Gyan.FFmpeg --exact --accept-package-agreements --accept-source-agreements --disable-interactivity
        if ($LASTEXITCODE -eq 0) {
            Write-Host '[OK] FFmpeg installation command completed.' -ForegroundColor Green
            Write-Host 'If audio alignment later says FFmpeg is missing, restart Windows once.' -ForegroundColor Yellow
            return
        }
    }
    Write-Host '[WARNING] FFmpeg could not be installed automatically.' -ForegroundColor Yellow
    Write-Host 'You can still open LyricCap, but audio alignment needs FFmpeg.'
}

try {
    Start-Transcript -Path $log -Append | Out-Null
    Clear-Host
    Write-Host '========================================================' -ForegroundColor Cyan
    Write-Host ' LyricCap Studio v0.1.5 - Audio Sync Windows Setup' -ForegroundColor Cyan
    Write-Host '========================================================' -ForegroundColor Cyan
    Write-Host 'This installer can install Python 3.12 and FFmpeg automatically.'
    Write-Host 'A setup log is saved as setup_log.txt.'

    $py = Install-PythonIfNeeded
    Write-Host ('[OK] Python: ' + $py.Exe + ' ' + ($py.Args -join ' ')) -ForegroundColor Green
    $venvPython = Ensure-Venv $py
    Install-Dependencies $venvPython
    Ensure-FFmpeg

    Write-Step 'SETUP COMPLETE'
    Write-Host 'LyricCap Studio is ready.' -ForegroundColor Green

    if (-not $InstallOnly) {
        Write-Host 'Starting the program now...'
        $env:PYTHONPATH = (Join-Path $root 'src')
        & $venvPython (Join-Path $root 'app.py')
    } else {
        Write-Host 'Next time, double-click RUN_LYRICCAP.bat.'
    }
    Stop-Transcript | Out-Null
    Write-Host ''
    Read-Host 'Press ENTER to close this setup window'
    exit 0
} catch {
    try { Stop-Transcript | Out-Null } catch {}
    Write-Host ''
    Write-Host '========================================================' -ForegroundColor Red
    Write-Host '[ERROR] SETUP DID NOT COMPLETE' -ForegroundColor Red
    Write-Host '========================================================' -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Yellow
    Write-Host ''
    Write-Host ('Log file: ' + $log)
    Write-Host 'This window will stay open so you can read or capture the error.'
    Write-Host 'Type EXIT and press ENTER only when you want to close it.'
    cmd /k
    exit 1
}
