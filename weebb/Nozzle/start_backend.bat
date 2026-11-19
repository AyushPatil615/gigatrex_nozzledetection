@echo off
REM ============================================================
REM  NozzleVision AI — Backend Startup Script
REM  Run this from: weebb\Nozzle\
REM ============================================================

echo.
echo  =========================================
echo   NozzleVision AI Backend + ngrok Tunnel
echo  =========================================
echo.

REM Start FastAPI backend in a new window
echo [1/2] Starting FastAPI backend on port 8000...
start "NozzleVision Backend" cmd /k "cd /d %~dp0 && python -m uvicorn backend.server:app --host 0.0.0.0 --port 8000"

REM Wait 4 seconds for backend to start
timeout /t 4 /nobreak >nul

REM Start ngrok tunnel in another window
echo [2/2] Starting ngrok tunnel...
start "ngrok Tunnel" cmd /k "ngrok http 8000"

echo.
echo  Both windows opened!
echo  - Copy the ngrok URL from the ngrok window (looks like: https://xxxx.ngrok-free.app)
echo  - Paste it in the web app via the top-right button
echo.
pause
