import os
import sys

TESTS = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.dirname(TESTS)
REPO = os.path.dirname(os.path.dirname(TOOLS))
RECORDS = os.path.join(REPO, "records")
FIXTURES = os.path.join(TESTS, "fixtures")
ANVIL2 = os.path.join(RECORDS, "track_1_short", "2026-08-30_ANVIL2")
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)
