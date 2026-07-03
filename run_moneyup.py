"""Run 머니업 Phase-0 extraction (base env).  Example:  python run_moneyup.py --max 4

Thin wrapper around moneyup_advisor.pipeline so it sits beside the repo's other run_*.py.
Isolated module — does NOT touch the daily YouTube pipeline / scheduler / email / Supabase / dashboard.
"""
import sys

from moneyup_advisor.pipeline import main

if __name__ == "__main__":
    sys.exit(main())
