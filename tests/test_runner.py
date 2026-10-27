"""Stage ordering, failure boundaries and pause handling for the full runner."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
STAGES = ["01_agent", "02_trivia", "03_signals", "04_trajectory", "05_figures",
          "06_robustness", "07_preaction", "08_preaction_analysis"]
SCRIPTS = ["01_run_agent.py", "02_trivia_features.py", "03_step_signals.py", "04_trajectory.py",
           "05_figures.py", "06_robustness.py", "07_preaction_features.py", "08_preaction_analysis.py"]


@pytest.fixture
def runner(monkeypatch):
    spec = importlib.util.spec_from_file_location("runner_test", ROOT / "scripts/run_experiment.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.importlib.metadata, "version", lambda name: "test")
    return module


@pytest.mark.parametrize("failed_stage", [None, "02_trivia", "07_preaction"])
def test_runner_stage_order_and_failure_boundary(tmp_path, monkeypatch, runner, failed_stage):
    calls = []

    def popen(command, **kwargs):
        index = len(calls)
        calls.append((command, kwargs))
        code = 7 if STAGES[index] == failed_stage else 0
        return SimpleNamespace(pid=100 + index, returncode=code, poll=lambda: code)

    monkeypatch.setattr(runner.subprocess, "Popen", popen)
    code = runner.execute(tmp_path, "meta-llama/Llama-3.1-8B-Instruct")
    count = 8 if failed_stage is None else STAGES.index(failed_stage) + 1
    assert code == (0 if failed_stage is None else 7)
    assert [Path(command[2]).name for command, _ in calls] == SCRIPTS[:count]
    for command, kwargs in calls[5:]:
        assert command[3:] == ["--run", str(tmp_path)]
        assert kwargs["env"]["AGENTCONF_SAMPLE_BATCH_SIZE"] == "1"
    state = json.loads((tmp_path / "pipeline_state.json").read_text())
    assert state["state"] == ("complete" if failed_stage is None else "failed")
    assert state["completed_stages"] == STAGES[:count if failed_stage is None else count - 1]
    manifest = json.loads(next((tmp_path / "attempts").glob("*/manifest.json")).read_text())
    assert [stage[0] for stage in manifest["stages"]] == STAGES
    assert (tmp_path / "logs/latest.log").exists()


def test_runner_pause_records_state_and_stops_worker(tmp_path, monkeypatch, runner):
    worker = SimpleNamespace(pid=100, returncode=None, poll=lambda: None)
    stopped = []

    def popen(*args, **kwargs):
        (tmp_path / "pause.request").touch()
        return worker

    monkeypatch.setattr(runner.subprocess, "Popen", popen)
    monkeypatch.setattr(runner, "stop_worker", lambda process: stopped.append(process.pid))
    assert runner.execute(tmp_path, "test-model") == 130
    state = json.loads((tmp_path / "pipeline_state.json").read_text())
    assert state["state"] == "paused" and state["child_pid"] is None
    assert stopped == [100]
