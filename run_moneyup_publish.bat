@echo off
REM Build the 머니업 dashboard as a static site (dist/) and deploy it to Vercel (team-only).
REM ADDITIVE + safe to re-run: read-only on data, writes only dist/, never touches the running :8077
REM dashboard or any other process, zero GPU. The export aborts (errorlevel 1) if the secret scan
REM finds anything, in which case we do NOT deploy.
setlocal
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d C:\Users\A\Downloads\trading-agent

echo === [1/2] Building static site (export_static.py) ===
"C:\Users\A\miniconda3\python.exe" -m moneyup_advisor.dashboard.export_static
if errorlevel 1 (
  echo.
  echo EXPORT FAILED or SECRET SCAN TRIPPED -- NOT deploying. See the output above.
  exit /b 1
)

echo.
echo === [2/2] Deploying dist\ to Vercel (--prod) ===
where vercel >nul 2>&1
if errorlevel 1 (
  echo Vercel CLI not installed. To deploy this static site, run these once:
  echo     npm i -g vercel
  echo     cd /d C:\Users\A\Downloads\trading-agent\dist
  echo     vercel --prod
  echo ^(team-only: enable Vercel Authentication / password protection in the Vercel project settings.^)
  exit /b 0
)
cd /d C:\Users\A\Downloads\trading-agent\dist
vercel --prod
