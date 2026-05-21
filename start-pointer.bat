@echo off
REM Double-click to launch Pointer locally (backend + frontend).
REM Each service runs in its own console window so you can see its logs.

setlocal
set "ROOT=%~dp0"

echo.
echo  ===========================================
echo   Pointer - AI Avatar Campus Assistant
echo  ===========================================
echo.
echo  Backend  : http://127.0.0.1:8000
echo  Frontend : http://localhost:3000
echo.
echo  Closing this launcher window will NOT stop
echo  the services. Close their own windows to stop.
echo.

REM -- Sanity checks ------------------------------------------------------------
where uv >nul 2>nul || (
  echo  [ERROR] 'uv' not found on PATH. Install it or reopen your terminal.
  pause
  exit /b 1
)
where npm >nul 2>nul || (
  echo  [ERROR] 'npm' not found on PATH. Install Node.js 20+.
  pause
  exit /b 1
)

REM -- Backend (FastAPI / uvicorn with reload) ----------------------------------
REM /D sets the working directory for the new window, so no pushd / && needed.
start "Pointer Backend (FastAPI :8000)" /D "%ROOT%backend" cmd /k uv run uvicorn app.main:app --reload --port 8000

REM Small delay so the backend grabs its port first
timeout /t 3 /nobreak >nul

REM -- Frontend (Next.js dev server) --------------------------------------------
start "Pointer Frontend (Next.js :3000)" /D "%ROOT%frontend" cmd /k npm run dev

REM Give Next.js a moment to compile before opening the browser
timeout /t 8 /nobreak >nul

start "" "http://localhost:3000"

echo.
echo  Launched. You can close this window.
echo.
timeout /t 3 /nobreak >nul
endlocal
