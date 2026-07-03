@echo off
REM 머니업 continuous batch launcher (scheduled task MoneyUp_Batch, 07:00 daily).
REM Runs the isolated module's overnight runner; it self-stops at 05:00 KST (22h) and unloads
REM all GPU models so 05:00-07:00 is free for the daily news report. Touches no daily-pipeline code.
setlocal
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d C:\Users\A\Downloads\trading-agent
"C:\Users\A\miniconda3\python.exe" -m moneyup_advisor.overnight >> "data\_moneyup_advisor\overnight_console.log" 2>&1
