@echo off
setlocal
cd /d "%~dp0"
set "PROJECT_CONDA=%USERPROFILE%\anaconda3\Scripts\conda.exe"
if not exist "%PROJECT_CONDA%" (
    echo Anaconda was not found at %USERPROFILE%\anaconda3. See README.md.
    exit /b 2
)
"%PROJECT_CONDA%" run --no-capture-output -n ev-vision python -B -X utf8 %*
exit /b %errorlevel%
