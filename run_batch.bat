@echo off
cd /d C:\Users\A\Downloads\trading-agent
REM ===========================================================================
REM Belt-and-suspenders: the daily transcribe OWNS the 05:30 morning GPU window.
REM If the moneyup collection's own 05:00 self-pause failed, forcibly clear it here:
REM   - kill any moneyup python (overnight / pipeline / vision_worker) -- matches only
REM     python.exe whose command line contains 'moneyup_advisor', so this transcribe
REM     (run_youtube_batch.py) is NEVER killed;
REM   - kill the moneyup conda-env worker (PaddleOCR);
REM   - unload the Qwen3-VL model from Ollama to free VRAM.
REM moneyup resumes on its own via the MoneyUp_Batch task at 07:00 (i.e. after 06:30).
REM ===========================================================================
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -like '*moneyup_advisor*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"
powershell -NoProfile -Command "Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.Path -like '*envs\moneyup*' } | Stop-Process -Force -ErrorAction SilentlyContinue"
"C:\Users\A\AppData\Local\Programs\Ollama\ollama.exe" stop qwen3-vl:30b-a3b-instruct-q4_K_M >nul 2>&1
REM let the GPU free before transcribing (~3s)
ping -n 4 127.0.0.1 >nul
REM --- the daily transcribe (unchanged) ---
C:\Users\A\miniconda3\python.exe scripts\run_youtube_batch.py --max-videos 60
