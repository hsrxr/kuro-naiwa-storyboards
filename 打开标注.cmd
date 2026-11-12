@echo off
cd /d "%~dp0"
start "" pythonw "%~dp0review_server.py" --open
