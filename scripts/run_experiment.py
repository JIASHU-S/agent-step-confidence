"""Run the complete formal experiment, keeping stage logs and resumable collection outputs.

python scripts/run_experiment.py --run runs/qwen7b
Create <run>/pause.request to stop the current worker and pause the pipeline. Relaunch the same command
to resume collection; completed questions are retained. Analysis stages are rerun if necessary.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agentconf.checkpoints import run_lock  # noqa: E402
from agentconf.io import write_json  # noqa: E402
from agentconf.logview import update_latest_log  # noqa: E402


def now():
    return datetime.now(timezone.utc).isoformat()


def publish_state(out, state):
    write_json(out / "pipeline_state.json", state)
    update_latest_log(out, state)


def stop_worker(worker):
    if worker.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(worker.pid), "/T", "/F"], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       creationflags=subprocess.CREATE_NO_WINDOW)
    else:
        worker.terminate()
    worker.wait(timeout=30)


def execute(out, model):
    cache = ROOT / ".cache"
    env = os.environ.copy()
    env.update(HF_HOME=str(cache / "huggingface"), MPLCONFIGDIR=str(cache / "matplotlib"),
               WIN_PD_OVERRIDE_LOCAL_APPDATA=str(cache / "localappdata"),
               PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1", HF_HUB_DISABLE_SYMLINKS_WARNING="1",
               HF_HUB_DOWNLOAD_TIMEOUT="60", HF_HUB_ETAG_TIMEOUT="60")
    env.setdefault("PYTENSOR_FLAGS", f"base_compiledir={cache / 'pytensor'}")
    env.setdefault("AGENTCONF_SAMPLE_BATCH_SIZE", "1")
    for key in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE"):
        env.pop(key, None)
    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    archive = out / "attempts" / stamp
    sources = {}
    for directory in (ROOT / "agentconf", ROOT / "scripts"):
        for source in sorted(directory.glob("*.py")):
            relative = source.relative_to(ROOT)
            target = archive / "source" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            sources[str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
    versions = {p: importlib.metadata.version(p) for p in
                ("torch", "transformers", "bitsandbytes", "datasets", "numpy", "scikit-learn", "pymc", "arviz")}
    stages = [
        ("01_agent", "01_run_agent.py", ["--model", model, "--out", str(out), "--n-train", "600",
                                         "--n-test", "300", "--seed", "0"]),
        ("02_trivia", "02_trivia_features.py", ["--model", model, "--out", str(out), "--n", "2000", "--seed", "0"]),
        ("03_signals", "03_step_signals.py", ["--run", str(out)]),
        ("04_trajectory", "04_trajectory.py", ["--run", str(out), "--signal", "probe"]),
        ("05_figures", "05_figures.py", ["--run", str(out)]),
        ("06_robustness", "06_robustness.py", ["--run", str(out)]),
        ("07_preaction", "07_preaction_features.py", ["--run", str(out)]),
        ("08_preaction_analysis", "08_preaction_analysis.py", ["--run", str(out)]),
    ]
    state = {"state": "starting", "pid": os.getpid(), "model": model, "run": str(out),
             "started_at_utc": now(), "attempt": stamp, "completed_stages": []}
    write_json(archive / "manifest.json", {**state, "python": sys.version, "packages": versions,
                                          "sample_batch_size": int(env["AGENTCONF_SAMPLE_BATCH_SIZE"]),
                                          "source_sha256": sources, "stages": stages})
    pause = out / "pause.request"
    # An explicit relaunch resumes a previously paused pipeline.
    pause.unlink(missing_ok=True)
    for name, script, arguments in stages:
        command = [sys.executable, "-u", str(ROOT / "scripts" / script), *arguments]
        log_path = logs / f"{name}.log"
        state.update(state="running", stage=name, stage_started_at_utc=now(), command=command,
                     log=str(log_path), child_pid=None)
        publish_state(out, state)
        print(f"{now()} Starting {name}; log: {log_path}", flush=True)
        with log_path.open("ab", buffering=0) as log:
            log.write(f"\n=== Attempt {stamp}; {now()} ===\n".encode())
            worker = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            state["child_pid"] = worker.pid
            publish_state(out, state)
            try:
                while worker.poll() is None:
                    update_latest_log(out, state)
                    if pause.exists():
                        stop_worker(worker)
                        state.update(state="paused", stopped_at_utc=now(), child_pid=None)
                        publish_state(out, state)
                        print(f"{now()} Paused; relaunch the same command to resume", flush=True)
                        return 130
                    time.sleep(2)
            except KeyboardInterrupt:
                stop_worker(worker)
                state.update(state="paused", stopped_at_utc=now(), child_pid=None)
                publish_state(out, state)
                return 130
        state.update(child_pid=None, exit_code=worker.returncode)
        if worker.returncode:
            state.update(state="failed", stopped_at_utc=now())
            publish_state(out, state)
            print(f"{now()} {name} failed ({worker.returncode}); inspect {log_path}", flush=True)
            return worker.returncode
        state["completed_stages"].append(name)
        publish_state(out, state)
    state.update(state="complete", finished_at_utc=now())
    publish_state(out, state)
    print(f"{now()} Experiment complete; results: {out / 'results'}", flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "runs" / "qwen7b")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    args = parser.parse_args()
    out = args.run.resolve()
    with run_lock(out / ".pipeline"):
        return execute(out, args.model)


if __name__ == "__main__":
    raise SystemExit(main())
