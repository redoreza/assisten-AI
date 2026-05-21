#Requires -Version 5.1
<#
.SYNOPSIS
    One-shot setup script for Pointer (AI Avatar Campus Assistant) on Windows.
    Run from the project root:  .\scripts\setup_windows.ps1

.DESCRIPTION
    1. Checks Python 3.11-3.12 and Node.js 20+
    2. Checks/installs FFmpeg (via winget)
    3. Warns if Visual C++ Build Tools are missing (needed by InsightFace)
    4. Installs/verifies uv Python package manager
    5. Runs uv sync for Python dependencies
    6. Runs npm install for frontend
    7. Creates required data directories
    8. Copies .env.example -> .env if not present
    9. Prints next steps
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ── Helpers ────────────────────────────────────────────────────────────────────

function Write-Step([string]$msg) { Write-Host "`n[SETUP] $msg" -ForegroundColor Cyan }
function Write-OK([string]$msg)   { Write-Host "  OK   $msg"  -ForegroundColor Green }
function Write-Warn([string]$msg) { Write-Host "  WARN $msg"  -ForegroundColor Yellow }
function Write-Fail([string]$msg) { Write-Host "  FAIL $msg"  -ForegroundColor Red; exit 1 }

function Test-Cmd([string]$name) {
    return [bool](Get-Command $name -ErrorAction SilentlyContinue)
}

# ── Resolve project root (one level above scripts/) ───────────────────────────

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

Write-Host "`n============================================" -ForegroundColor Magenta
Write-Host "  Pointer — Windows Setup" -ForegroundColor Magenta
Write-Host "  Root: $ProjectRoot" -ForegroundColor Magenta
Write-Host "============================================`n" -ForegroundColor Magenta

# ── 1. Python 3.11 / 3.12 ─────────────────────────────────────────────────────

Write-Step "Checking Python 3.11 or 3.12..."
if (-not (Test-Cmd python)) {
    Write-Fail "Python not found.`n  Install from https://python.org`n  Make sure 'Add Python to PATH' is checked."
}
$pyver = python --version 2>&1
if ($pyver -notmatch "3\.1[12]") {
    Write-Fail "Found $pyver, but project needs 3.11 or 3.12.`n  Download the right version from https://python.org."
}
Write-OK $pyver

# ── 2. Node.js 20+ ────────────────────────────────────────────────────────────

Write-Step "Checking Node.js 20+..."
if (-not (Test-Cmd node)) {
    Write-Fail "Node.js not found.`n  Install LTS from https://nodejs.org."
}
$nodever  = node --version 2>&1
$nodeMajor = [int]($nodever -replace 'v(\d+).*', '$1')
if ($nodeMajor -lt 20) {
    Write-Fail "Node.js $nodever found, but 20+ is required.`n  Install LTS from https://nodejs.org."
}
Write-OK "Node.js $nodever"

# ── 3. Git ────────────────────────────────────────────────────────────────────

Write-Step "Checking Git..."
if (-not (Test-Cmd git)) {
    Write-Fail "Git not found.`n  Install from https://git-scm.com."
}
Write-OK (git --version 2>&1)

# ── 4. FFmpeg (required by pydub for webm->wav audio conversion) ──────────────

Write-Step "Checking FFmpeg..."
if (Test-Cmd ffmpeg) {
    Write-OK (ffmpeg -version 2>&1 | Select-Object -First 1)
} else {
    Write-Warn "FFmpeg not found. Attempting install via winget..."
    if (Test-Cmd winget) {
        winget install --id Gyan.FFmpeg --silent --accept-package-agreements --accept-source-agreements 2>&1
        # Reload PATH
        $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH","Machine") + ";" +
                    [System.Environment]::GetEnvironmentVariable("PATH","User")
        if (Test-Cmd ffmpeg) {
            Write-OK "FFmpeg installed."
        } else {
            Write-Warn "FFmpeg was installed but is not yet on PATH."
            Write-Warn "Close this terminal completely, reopen it, and re-run this script."
            exit 1
        }
    } else {
        Write-Fail "winget not available.`n  Install FFmpeg manually: https://ffmpeg.org/download.html`n  Then add it to PATH and re-run this script."
    }
}

# ── 5. Visual C++ Build Tools (InsightFace compiles a C extension) ─────────────
#
# Without MSVC, 'uv sync' will fail at insightface with:
#   "Microsoft Visual C++ 14.0 or greater is required"
# We detect, warn clearly, but do NOT abort — wheels may be pre-built for your
# Python/platform combination.

Write-Step "Checking Visual C++ Build Tools (needed by InsightFace)..."
$vsWhere   = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$hasMSVC   = $false
if (Test-Path $vsWhere) {
    $found = & $vsWhere -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 2>$null
    $hasMSVC = ($null -ne $found -and $found -ne "")
}
if ($hasMSVC) {
    Write-OK "MSVC Build Tools found."
} else {
    Write-Warn "Visual C++ Build Tools not detected."
    Write-Warn "InsightFace (face recognition) may fail to install without them."
    Write-Warn ""
    Write-Warn "To install (~6 GB, run once):"
    Write-Warn "  winget install --id Microsoft.VisualStudio.2022.BuildTools --silent \\"
    Write-Warn "    --override '--add Microsoft.VisualStudio.Workload.VCTools --includeRecommended'"
    Write-Warn ""
    Write-Warn "After installing, close ALL terminals, reopen, and re-run this script."
    Write-Warn "Continuing anyway..."
}

# ── 6. uv Python package manager ──────────────────────────────────────────────

Write-Step "Checking uv..."
if (-not (Test-Cmd uv)) {
    Write-Warn "uv not found. Installing..."
    powershell -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH","Machine") + ";" +
                [System.Environment]::GetEnvironmentVariable("PATH","User")
    if (-not (Test-Cmd uv)) {
        Write-Fail "uv installed but not on PATH yet.`n  Close this terminal, reopen it, and re-run the script."
    }
}
Write-OK "uv $(uv --version 2>&1)"

# ── 7. Python backend dependencies ────────────────────────────────────────────

Write-Step "Installing Python dependencies (uv sync)..."
Push-Location "$ProjectRoot\backend"
try {
    uv sync
    Write-OK "Python dependencies installed."
} catch {
    Write-Fail "uv sync failed.`n$_`n`nMost common cause: Visual C++ Build Tools missing (see step 5)."
} finally {
    Pop-Location
}

# ── 8. Node.js frontend dependencies ──────────────────────────────────────────

Write-Step "Installing Node.js dependencies (npm install)..."
Push-Location "$ProjectRoot\frontend"
try {
    npm install --prefer-offline
    Write-OK "Node.js dependencies installed."
} catch {
    Write-Fail "npm install failed.`n$_"
} finally {
    Pop-Location
}

# ── 9. Data directories ───────────────────────────────────────────────────────

Write-Step "Creating data directories..."
@("data\sqlite", "data\chroma", "data\knowledge_base", "data\personas") | ForEach-Object {
    $full = Join-Path $ProjectRoot $_
    if (-not (Test-Path $full)) {
        New-Item -ItemType Directory -Path $full -Force | Out-Null
        Write-OK "Created $_"
    } else {
        Write-OK "Already exists: $_"
    }
}

# ── 10. .env ──────────────────────────────────────────────────────────────────

Write-Step "Setting up .env..."
$envDst = Join-Path $ProjectRoot ".env"
$envSrc = Join-Path $ProjectRoot ".env.example"
if (Test-Path $envDst) {
    Write-OK ".env already exists — not overwriting."
} else {
    Copy-Item $envSrc $envDst
    Write-OK ".env created from .env.example"
}

# ── Done ───────────────────────────────────────────────────────────────────────

Write-Host "`n============================================" -ForegroundColor Green
Write-Host "  Setup complete!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host @"

Next steps:
  1. Edit .env — fill in your API keys:
       GROQ_API_KEY   (required) — https://console.groq.com   (free)
       TAVILY_API_KEY (optional) — https://app.tavily.com     (free, web search)
       NVIDIA_API_KEY (optional) — https://build.nvidia.com   (free, faster LLM)

  2. Make sure data\personas\pointer.json exists.
     Copy the example from README.md section 7 if needed.

  3. Run the app:
       Double-click start-pointer.bat
     OR in two separate terminals:
       cd backend  && uv run uvicorn app.main:app --reload --port 8000
       cd frontend && npm run dev

  4. Open http://localhost:3000

NOTE: On first launch InsightFace will download the 'buffalo_l' face model
(~300 MB). This happens only once; subsequent starts are fast.

"@ -ForegroundColor White
