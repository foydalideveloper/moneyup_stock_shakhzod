@echo off
REM 머니업 Phase 2 playbook auto-regenerate (scheduled task MoneyUp_Playbook, daily 06:50).
REM Gemini-based (paid REST API), NO GPU -- safe to run anytime, even while the collector runs.
REM Incremental: per-video extractions are cached, so only NEW fact sheets hit Gemini; then a
REM corpus-level pass re-clusters + re-ranks. Rewrites data\_moneyup_advisor\playbook\playbook.{json,md}
REM with an updated header video count + timestamp. Descriptive only -- profitability is Phase 1.
setlocal
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d C:\Users\A\Downloads\trading-agent
"C:\Users\A\miniconda3\python.exe" -m moneyup_advisor.playbook >> "data\_moneyup_advisor\playbook_auto.log" 2>&1
