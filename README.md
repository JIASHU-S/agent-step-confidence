# Where does an agent go wrong? Step-level confidence from internal states

A small study of confidence in multi-step LLM agents. A ReAct agent answers HotpotQA questions by searching the 10
offline paragraphs each question comes with. Every step gets an automatic correctness label and four confidence
signals, and three questions are asked:

- **RQ1** Which signal best detects a wrong step: a linear probe on hidden states, token log-probability, sampling
  consistency, or the model's verbalized confidence?
- **RQ2** Does a probe trained on single-step QA (TriviaQA) transfer to agent steps, and how much does a little
  recalibration recover?
- **RQ3** Are step errors coupled? A Bayesian model with a previous-error term (M1) is compared with an
  independent-steps model (M0) by PSIS-LOO (ΔELPD / SE). Trajectory confidence then marginalises the unknown
  previous error with a forward recursion, and drives a "hand off to a person" policy.

## Step labels (no judge model)

| action | correct when |
| --- | --- |
| `Search[q]` | it returns a supporting paragraph the agent has not retrieved before |
| `Finish[a]` | token F1 with the gold answer ≥ 0.5 |
| unparsable | never |

Sampling consistency treats two searches as the same action when they return the same paragraph.
TriviaQA `rc.nocontext` contains repeated rows for the same question. Source-domain sampling keeps the first
row for each `question_id`, then shuffles the unique questions with the configured seed and selects 2000.

## Setup

```bash
pip install -r requirements.txt
python -m pytest -q tests/   # tiny random models, no downloads
```

### Windows / PowerShell

Use an isolated environment with a CUDA wheel for the RTX 4070 Ti:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
.\.venv\Scripts\python.exe -m pip install -r requirements.txt "transformers==4.57.6" "numpy==1.26.4" "pymc==5.25.1" hf_xet
.\scripts\python_local.ps1 -m pytest -q tests/
.\scripts\python_local.ps1 scripts/01_run_agent.py --model Qwen/Qwen2.5-0.5B-Instruct --out runs/smoke --n-train 5 --n-test 5 --no-4bit
```

`scripts/python_local.ps1` uses `.venv` and keeps Hugging Face, Matplotlib, ArviZ and PyTensor caches in `.cache/`.
Use it in place of `python` in the commands below. Without a C++ compiler, PyMC falls back to its slower Python
backend.

The recorded experiment environment is included in [`reproducibility/environment.json`](reproducibility/environment.json).
For the pinned Windows / Python 3.10 / CUDA 12.4 environment, install the archived dependency versions instead of
the unpinned requirements above:

```powershell
.\.venv\Scripts\python.exe -m pip install -r reproducibility/requirements-lock.txt
```

This lock file records the original Windows GPU environment; other platforms may need different PyTorch wheels.
It pins packages, but model and dataset revisions are not pinned. Exact regenerated values may therefore vary.

### Complete workflow

To run all eight stages in order with the formal settings:

```powershell
.\scripts\python_local.ps1 -u scripts/run_experiment.py --run runs/qwen7b

# Llama-3.1-8B-Instruct, with the same formal settings
.\scripts\python_local.ps1 -u scripts/run_experiment.py --run runs/llama8b --model meta-llama/Llama-3.1-8B-Instruct
```

For Llama, use a Hugging Face account with access to the model and authenticate with `hf auth login` first.

The runner collects 600 train + 300 test Agent episodes, collects 2000 TriviaQA answers, then runs signals,
trajectory analysis, figures, robustness checks (06), pre-action replay (07) and pre-action analysis (08).
Stage logs are in `runs/qwen7b/logs/`; `pipeline_state.json` records the active
stage and process IDs. Each launch archives source files and runtime versions under `attempts/`.
For a newest-first view, open `runs/qwen7b/logs/latest.log`; it refreshes every two seconds as new lines arrive
and follows the active stage. The original stage logs retain chronological order.
Any failed stage stops the pipeline. Relaunching the same command resumes collection and replay, and reruns
analysis stages. Replay checkpoints are saved atomically per episode; missing or damaged files are recomputed.

For a background run, create an empty `runs/qwen7b/pause.request` file to pause it. The worker is stopped and its
current uncommitted question is recomputed on resume. The runner removes this request on explicit relaunch.
Keep the computer awake while the experiment runs; the runner does not change system power settings.

## Run (RTX 4070 Ti 12 GB, Qwen2.5-7B-Instruct in 4-bit)

```bash
# 0. smoke test with a small model (~5 min)
python scripts/01_run_agent.py --model Qwen/Qwen2.5-0.5B-Instruct --out runs/smoke --n-train 5 --n-test 5 --no-4bit

# 1. agent episodes: 600 train + 300 test questions. Resumable; roughly 6-10 h.
python scripts/01_run_agent.py --model Qwen/Qwen2.5-7B-Instruct --out runs/qwen7b

# 2. TriviaQA source domain for RQ2 (~1 h)
python scripts/02_trivia_features.py --model Qwen/Qwen2.5-7B-Instruct --out runs/qwen7b --n 2000

# 3. RQ1 + RQ2 (CPU, minutes)
python scripts/03_step_signals.py --run runs/qwen7b

# 4. RQ3 (CPU, a few minutes of NUTS sampling)
python scripts/04_trajectory.py --run runs/qwen7b --signal probe

# 5. figures
python scripts/05_figures.py --run runs/qwen7b

# 6. step-position control, after 06-08 (CPU, minutes; not part of run_experiment.py)
python scripts/09_position_control.py --run runs/qwen7b
```

Results go to `runs/<name>/results/`. Hidden states are stored for 8 evenly spaced layers (`--layers` to change),
mean-pooled over the Action span and at its last token; the layer and pooling for the probe are chosen by
out-of-fold AUROC on the training split only.

Consistency still uses five alternative samples per step, but generates them one at a time by default to
bound peak GPU memory on 12 GB cards. `AGENTCONF_SAMPLE_BATCH_SIZE` can override this execution batch size;
it does not change the number of samples. Batch scheduling changes RNG consumption, so keep it fixed for
reproducibility. The full-workflow runner records it in each attempt manifest. Feature extraction disables
the unnecessary KV cache, and generation releases unused CUDA cache between calls. Agent logs include each
step's action generation, feature extraction, alternative sampling and confidence phases.

## Interruption and resume

Scripts **01 and 02 both save after every question**. To pause, press Ctrl+C and wait for the process to exit.
Resume by running the same command with the same arguments and output directory. Completed questions are
checked and skipped; an interrupted question, or one with missing/damaged features, is recomputed. No model
is loaded if every requested question is already complete.

- Features and metadata are written to temporary files, flushed to disk, then atomically replaced. A question's
  completion record is appended and flushed only after its feature file is saved.
- On resume, an incomplete final JSONL line is backed up to `*.corrupt-tail-*.bak` before removal. Valid records
  without a final newline are preserved. Corruption in the middle of a file stops recovery without silently
  discarding records. Orphaned and duplicate step records are removed during agent recovery.
- Saved model, sampling settings, question counts, seed, layers and dataset IDs must match. To change these,
  use a new output directory. A process lock rejects a second collector using the same directory and is released
  when the process exits, including forced termination; the `.collection.lock` file itself can remain.
- New runs seed sampling independently for each question, so skipping completed questions does not shift the
  random sequence for subsequent questions. Legacy agent runs remain resumable with their original unseeded
  sampling behaviour.

TriviaQA uses `trivia.progress.jsonl`, `trivia_checkpoints/<id>.npz` and `trivia_meta.json` for recovery. Keep these
files to retain resume support. Once all questions finish, script 02 rebuilds the usual `trivia.jsonl` and
`trivia_feats.npz` in dataset order. If export is interrupted, rerunning script 02 rebuilds it from checkpoints
without repeating inference. Script 03 refuses incomplete TriviaQA exports. Legacy TriviaQA exports without
checkpoint metadata still work with script 03; script 02 refuses to overwrite them automatically.

## Layout

```
agentconf/env.py       offline Search/Finish environment, BM25, step labels
agentconf/agent.py     ReAct loop; records labels and signals per step
agentconf/model.py     loading (4-bit), generation, teacher-forced hidden states + span log-probs
agentconf/checkpoints.py  collection locks, configuration checks, per-question recovery and seeding
agentconf/io.py        atomic files, durable JSONL appends and incomplete-tail repair
agentconf/probes.py    standardise -> PCA -> logistic probes, grouped cross-validation
agentconf/seqmodel.py  M0/M1 in PyMC, PSIS-LOO comparison, forward recursion, hand-off curves
scripts/               stages 01-08, the full-workflow runner and the step-position control (09)
```

## Results in this repository

`results/qwen7b/` (Qwen2.5-7B-Instruct) and `results/llama8b/` (Llama-3.1-8B-Instruct) hold the analysis outputs
of the full runs (900 HotpotQA questions, 2000 TriviaQA questions, seed 0, 4-bit NF4): per-step scores, all
tables behind the report, and the figures. Raw trajectories and hidden-state features (`runs/`) are not included
because of their size; they are regenerated by `scripts/run_experiment.py`.

The checked-in `results/` folders are analysis exports, not resumable `--run` folders. The analysis scripts expect
raw episodes, steps and features in `runs/<name>/`; use the full-workflow commands above to regenerate these inputs.
The exports include 900 unique HotpotQA questions per model: 2744 Qwen steps (902 test) and 3077 Llama steps
(1028 test), with the train/test question IDs and collection settings recorded in each model's `meta.json`.
