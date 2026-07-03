@echo off
chcp 65001 >nul
REM MoneyUp LIVE "watch it extract" demo server (boss demo) -- isolated, own port (default 8078).
REM Open http://127.0.0.1:8078/live , paste a YouTube link (or pick a file), press Start.
REM Runs the REAL pipeline on ONLY the first 5 minutes. Pauses the collector for the run, resumes after.
REM Touches nothing in the collector / daily report / playbook / existing dashboard.
setlocal
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d C:\Users\A\Downloads\trading-agent
"C:\Users\A\miniconda3\python.exe" -m moneyup_advisor.live.server
