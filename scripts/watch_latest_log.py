"""Provide a newest-first live log for an already running experiment, without restarting it."""
import argparse
import json
from pathlib import Path
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agentconf.checkpoints import run_lock  # noqa: E402
from agentconf.logview import update_latest_log  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    out = args.run.resolve()
    state_path = out / "pipeline_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if args.once:
        update_latest_log(out, state)
        return
    parent = psutil.Process(state["pid"])
    with run_lock(out / ".log-viewer"):
        while True:
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                time.sleep(2)
                continue
            # A newly launched runner maintains its own display copy.
            if state.get("pid") != parent.pid:
                return
            update_latest_log(out, state)
            if state.get("state") in ("paused", "failed", "complete") or not parent.is_running():
                return
            time.sleep(2)


if __name__ == "__main__":
    main()
