@echo off
REM Phase 1B auto-run watcher (scheduled task MoneyUp_Phase1B_Watch, at-logon + started now).
REM Waits until the degraded re-extraction queue is drained (or only terminal ids remain), then runs
REM the Phase 1B scorer ONCE on the clean corpus and stops. Zero GPU (pykrx prices only). Idempotent
REM via a done flag, so an at-logon re-launch after reboot just exits.
setlocal
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d C:\Users\A\Downloads\trading-agent
"C:\Users\A\miniconda3\python.exe" -m moneyup_advisor.phase1b_watch >> "data\_moneyup_advisor\phase1b_watch_console.log" 2>&1
