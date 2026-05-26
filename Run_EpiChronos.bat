@echo off
title EpiChronos - DNA Methylation Analysis Suite
color 0A

echo.
echo  ============================================================
echo       EpiChronos v0.1.2 - Epigenetic Analysis Suite
echo  ============================================================
echo.
echo  Starting up... Please wait.
echo.

cd /d "%~dp0"

:: Check Python
where python >nul 2>nul
if %errorlevel% neq 0 (
    color 0C
    echo  [ERROR] Python was not found on your system.
    echo  Please install Python 3.9+ from https://www.python.org
    echo.
    pause
    exit /b 1
)

:: Install dependencies silently if missing
echo  [1/2] Checking dependencies...
python -m pip install --quiet polars numpy scipy plotly streamlit pyarrow jinja2 openpyxl kaleido >nul 2>nul

echo  [2/2] Launching EpiChronos GUI in your browser...
echo.
echo  ============================================================
echo   The app will open automatically in your default browser.
echo   If it doesn't, go to:  http://localhost:8501
echo.
echo   To stop the server, close this window or press Ctrl+C.
echo  ============================================================
echo.

python -m streamlit run gui.py --server.headless true --browser.gatherUsageStats false

pause
