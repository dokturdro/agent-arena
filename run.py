"""Edit the settings below and press Run in PyCharm (or: uv run python run.py).

Same as `arena run`, but configured in code - handy for one-click runs and debugging.
"""

import asyncio
import webbrowser

from dotenv import load_dotenv

from agentarena.cli import print_board, print_progress
from agentarena.config import ROOT, load_models, load_suite
from agentarena.report import build_report
from agentarena.runner import run_suite

MODELS = ["glm-5.3", "claude-sonnet-5-5"]  # names from config/models.yaml
SUITE = "smoke"  # smoke | full | probe, or a path to a suite YAML
RUN_ID = "my-run"  # reuse an id to resume a run or add models to it
USE_CACHE = True
OPEN_REPORT = True

if __name__ == "__main__":
    load_dotenv(ROOT / ".env")
    registry = load_models()
    store = asyncio.run(run_suite(
        load_suite(SUITE),
        [registry[m] for m in MODELS],
        ROOT / "runs" / RUN_ID,
        ROOT / ".arena_cache" if USE_CACHE else None,
        print_progress,
    ))
    print_board(store)
    report = build_report(store)
    print(f"report: {report}")
    if OPEN_REPORT:
        webbrowser.open(report.as_uri())
