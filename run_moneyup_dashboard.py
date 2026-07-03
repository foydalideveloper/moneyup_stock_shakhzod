"""Serve the 머니업 Advisor dashboard on its OWN port (default 8077; existing dashboard is 8000).

    python run_moneyup_dashboard.py            # http://127.0.0.1:8077
    python run_moneyup_dashboard.py --port 8088
"""
import argparse

from moneyup_advisor import config
from moneyup_advisor.dashboard.server import serve

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=config.DASHBOARD_HOST)
    ap.add_argument("--port", type=int, default=config.DASHBOARD_PORT)
    a = ap.parse_args()
    serve(a.host, a.port)
