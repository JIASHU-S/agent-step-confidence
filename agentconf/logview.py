"""A newest-first display copy of the active log; original logs remain append-only."""
from pathlib import Path

from .io import atomic_open


def update_latest_log(out, state):
    out = Path(out).resolve()
    source = Path(state.get("log", out / "logs" / "01_agent.log")).resolve()
    if not source.is_relative_to(out):
        return False
    destination = out / "logs" / "latest.log"
    labels = {"running": "运行中", "starting": "启动中", "paused": "已暂停",
              "failed": "运行失败", "complete": "已完成"}
    try:
        raw = source.read_text(encoding="utf-8", errors="replace") if source.exists() else ""
        lines = [line for line in raw.splitlines() if line.strip()]
        header = (f"最新状态在顶部 | 阶段：{state.get('stage', 'initialising')} | "
                  f"状态：{labels.get(state.get('state'), state.get('state', 'unknown'))}\n\n")
        text = header + "\n".join(reversed(lines)) + "\n"
        if destination.exists() and destination.read_text(encoding="utf-8") == text:
            return False
        with atomic_open(destination, encoding="utf-8") as f:
            f.write(text)
        return True
    except OSError:
        # A viewer can briefly lock its file on Windows. Retry later; collection must keep running.
        return False
