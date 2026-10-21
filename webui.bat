@echo off
rem universal_dub Web UI launcher -> http://127.0.0.1:7860
cd /d C:\work\universal_dub
if "%WEBUI_PORT%"=="" set WEBUI_PORT=7860
set PYTHONUNBUFFERED=1
powershell -NoProfile -Command "& { $env:PYTHONUNBUFFERED='1'; & venv\Scripts\python.exe webui.py 2>&1 | Tee-Object -FilePath C:\work\universal_dub\webui_server.log }"
pause
