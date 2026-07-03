"""Re-parse stated_price on existing fact sheets with the fixed Korean number parser (만/억).

Read-only on the pipeline; rewrites only the isolated fact sheets (json + md). Pass video ids to
SKIP (e.g. ones being regenerated). Usage: python -m moneyup_advisor.reparse_prices [skip_id ...]
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

from moneyup_advisor import calls, config, factsheet


def main():
    skip = set(sys.argv[1:])
    changed = sheets = 0
    for p in sorted(glob.glob(str(config.SHEET_DIR / "*.json"))):
        vid = Path(p).stem
        if vid in skip:
            continue
        s = json.loads(Path(p).read_text(encoding="utf-8"))
        touched = False
        for key in ("exante_calls", "expost_commentary"):
            for c in s.get(key, []):
                new = calls.parse_korean_price(c.get("quote", "") or "")
                if new != c.get("stated_price"):
                    c["stated_price"] = new
                    touched = True
                    changed += 1
        if touched:
            sheets += 1
            Path(p).write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")
            (config.SHEET_DIR / f"{vid}.md").write_text(factsheet.render_markdown(s), encoding="utf-8")
    print(f"reparsed: {changed} call price(s) updated across {sheets} sheet(s) "
          f"(skipped {len(skip)} regenerating)")


if __name__ == "__main__":
    main()
