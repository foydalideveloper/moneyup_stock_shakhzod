"""Isolated LIVE 'watch it extract' demo (boss demo).

A brand-new page + endpoint on its OWN port (default 8078) that runs the REAL extraction
pipeline on ONLY the first 5 minutes of a pasted YouTube link (or an uploaded file) and streams
every stage to the browser in real time. Additive + isolated: imports the production functions
READ-ONLY and writes the resulting fact sheet to a per-job folder (NEVER to factsheets/ that the
collector / playbook / Phase-1 scan). Does not modify the collector, the daily report, the
playbook, or the existing dashboard pages.
"""
