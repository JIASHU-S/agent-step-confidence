"""Audit existing runs without modifying raw data. Writes P0 manifests and diagnostics."""
import argparse
import csv
import importlib.metadata
import json
import platform
import re
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agentconf.asset_audit import array_equal, inspect_npz, read_jsonl, sha256, validate_records
from agentconf.text import f1


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def command(args):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=20)
        return {'exit_code': p.returncode, 'stdout': p.stdout.strip(), 'stderr': p.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'error': str(exc)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=ROOT / 'manifests/submission_v2/p0')
    args = ap.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    inventory, report, splits = [], {}, {}

    def record(path, role, arrays=None):
        entry = {'path': path.relative_to(ROOT.parent).as_posix(), 'role': role,
                 'bytes': path.stat().st_size, 'sha256': sha256(path)}
        if arrays is not None:
            entry['arrays'] = arrays
        inventory.append(entry)

    for model, width in [('qwen7b', 3584), ('llama8b', 4096)]:
        t0 = time.perf_counter()
        run = ROOT / 'runs' / model
        meta = json.loads((run / 'meta.json').read_text(encoding='utf-8'))
        tm = json.loads((run / 'trivia_meta.json').read_text(encoding='utf-8'))
        eps, steps, tv = [read_jsonl(run / name) for name in ['episodes.jsonl', 'steps.jsonl', 'trivia.jsonl']]
        errors = validate_records(meta, eps, steps, tv, tm)
        warnings = []
        nl = len(meta['layers'])
        for s in steps:
            if s['kind'] == 'Finish':
                ep = next(e for e in eps if e['qid'] == s['qid'])
                if abs(f1(s['arg'], ep['gold']) - s['f1']) > 1e-12:
                    errors.append(f'recomputed Finish F1 mismatch {s["qid"]}/{s["t"]}')
        wanted = {e['qid'] for e in eps}
        for folder in ['feats', 'feats_pre', 'trivia_checkpoints']:
            expected_ids = set(tm['ids']) if folder == 'trivia_checkpoints' else wanted
            actual_ids = {p.stem for p in (run / folder).glob('*.npz')}
            if actual_ids != expected_ids:
                errors.append(f'{folder} missing={sorted(expected_ids-actual_ids)} extra={sorted(actual_ids-expected_ids)}')
        flags_by_qid = {}
        for e in eps:
            qid, n = e['qid'], e['n_steps']
            for folder, schema in [
                ('feats', {k: ((n, nl, width), 'float16') for k in ['mean', 'last']}),
                ('feats_pre', {**{k: ((n, nl, width), 'float16') for k in ['pre_step', 'pre_action']},
                               'has_action': ((n,), 'bool')})]:
                path = run / folder / (qid + '.npz')
                if not path.exists():
                    continue
                info, bad = inspect_npz(path, schema)
                errors.extend(f'{folder}/{qid}: {x}' for x in bad)
                record(path, folder, info)
                if folder == 'feats_pre' and not bad:
                    with np.load(path) as z:
                        flags_by_qid[qid] = z['has_action']
        for s in steps:
            if s['qid'] in flags_by_qid:
                expected = bool(re.search(r'Action\s*\d*\s*:', s['text']) and s['kind'] != 'invalid')
                if bool(flags_by_qid[s['qid']][s['t']-1]) != expected:
                    errors.append(f'has_action alignment {s["qid"]}/{s["t"]}')
        path = run / 'trivia_feats.npz'
        schema = {k: ((len(tv), nl, width), 'float16') for k in ['mean', 'last']}
        info, bad = inspect_npz(path, schema)
        errors.extend(f'trivia_feats: {x}' for x in bad)
        record(path, 'trivia_aggregate', info)
        matched = 0
        agg = None
        if not bad:
            with np.load(path) as z:
                agg = {k: z[k] for k in z.files}
        for i, r in enumerate(tv):
            path = run / 'trivia_checkpoints' / (r['id'] + '.npz')
            if not path.exists():
                continue
            info, bad = inspect_npz(path, {k: ((nl, width), 'float16') for k in ['mean', 'last']})
            errors.extend(f'trivia/{r["id"]}: {x}' for x in bad)
            record(path, 'trivia_checkpoint', info)
            if not bad and agg is not None:
                with np.load(path) as z:
                    equal = all(array_equal(z[k], agg[k][i]) for k in ['mean', 'last'])
                if equal:
                    matched += 1
                else:
                    errors.append(f'trivia aggregate/checkpoint mismatch: {r["id"]}')
        del agg
        progress = read_jsonl(run / 'trivia.progress.jsonl')
        if progress != tv:
            errors.append('TriviaQA progress rows differ from final export')

        # Compare every exported table/JSON against the raw-run result, across all copies.
        copy_checks = []
        canonical = run / 'results'
        sources = [ROOT / 'results' / model, ROOT.parent / ('results-' + model)]
        names = sorted({p.name for folder in [canonical, *sources] for p in folder.glob('*')
                        if p.suffix in ('.csv', '.json') and p.name != 'meta.json'})
        for name in names:
            base = canonical / name
            for folder in sources:
                other = folder / name
                state = 'identical' if base.exists() and other.exists() and sha256(base) == sha256(other) else (
                    'different' if base.exists() and other.exists() else 'missing')
                copy_checks.append({'file': name, 'copy': str(folder.relative_to(ROOT.parent)), 'status': state})
                if state != 'identical':
                    warnings.append(f'export copy {state}: {folder.name}/{name}')
        for folder in [canonical, *sources]:
            for p in sorted(folder.iterdir()):
                if p.is_file() and p.suffix in ('.csv', '.json'):
                    record(p, 'legacy_result')
        # Verify archived step probabilities are attached to the correct raw rows.
        with (canonical / 'step_scores.csv').open(encoding='utf-8') as f:
            scores = list(csv.DictReader(f))
        score_map = {(r['qid'], int(r['t'])): r for r in scores}
        raw_map = {(r['qid'], r['t']): r for r in steps}
        if len(score_map) != len(scores) or set(score_map) != set(raw_map):
            errors.append('step_scores key cardinality/alignment mismatch')
        for key, r in score_map.items():
            s = raw_map.get(key)
            if s is None:
                continue
            if (r['kind'], r['split'], int(r['correct'])) != (s['kind'], s['split'], s['correct']):
                errors.append(f'step_scores metadata mismatch: {key}')
            for k in ['lp_mean', 'lp_min', 'consistency', 'verbalized']:
                a = float(r[k]) if r[k] else float('nan')
                b = float(s[k]) if s[k] is not None else float('nan')
                if not np.isclose(a, b, equal_nan=True, atol=1e-12, rtol=0):
                    errors.append(f'step_scores raw signal mismatch: {key}/{k}')
            if not np.isfinite(float(r['probe'])) or not 0 <= float(r['probe']) <= 1:
                errors.append(f'invalid probe probability: {key}')
        ep_test = [e for e in eps if e['split'] == 'test']
        with (canonical / 'rq3_episodes.csv').open(encoding='utf-8') as f:
            rq3 = list(csv.DictReader(f))
        rq3ids = [r['qid'] for r in rq3]
        if len(rq3ids) != len(set(rq3ids)) or set(rq3ids) != {e['qid'] for e in ep_test if e['finished']}:
            errors.append('rq3 finished episode membership mismatch')
        epmap = {e['qid']: e for e in eps}
        for r in rq3:
            if int(r['success']) != epmap[r['qid']]['success']:
                errors.append(f'rq3 success mismatch: {r["qid"]}')
            for key in ['running_M0', 'running_M1']:
                vals = json.loads(r[key])
                if len(vals) != epmap[r['qid']]['n_steps'] or not all(np.isfinite(v) and 0 <= v <= 1 for v in vals):
                    errors.append(f'rq3 running confidence mismatch: {r["qid"]}/{key}')

        for p in run.glob('*.json*'):
            record(p, 'run_metadata_or_records')
        for p in sorted((run / 'attempts').rglob('manifest.json')):
            record(p, 'attempt_manifest')
        # Revision absence is a limitation, not guessed from current installed packages.
        revision_keys = [k for k in meta if 'revision' in k.lower() or 'commit' in k.lower()]
        if not revision_keys:
            warnings.append('Original model/tokenizer/dataset revisions are not pinned in run metadata.')
        warnings.append('Feature validity is structural only; prefix causality and Search gold-label validity remain P1 checks.')
        state = json.loads((run / 'pipeline_state.json').read_text(encoding='utf-8'))
        splits[model] = {'model': meta['model'], 'legacy_train_ids': sorted(k for k,v in meta['split'].items() if v == 'train'),
                         'legacy_development_ids': sorted(k for k,v in meta['split'].items() if v == 'test'),
                         'exclude_from_new_hotpot_ids': sorted(meta['split']),
                         'trivia_source_train_ids': sorted(tm['ids'], key=lambda x: __import__('hashlib').sha256(('p0-source-split-v1:'+x).encode()).hexdigest())[:1600],
                         'trivia_source_validation_ids': sorted(tm['ids'], key=lambda x: __import__('hashlib').sha256(('p0-source-split-v1:'+x).encode()).hexdigest())[1600:],
                         'new_calibration_ids': [], 'new_final_test_ids': [],
                         'new_data_status': 'not selected or collected; dataset revision must be fixed first'}
        missing_signals = {k: sum(s.get(k) is None or not np.isfinite(float(s[k])) for s in steps)
                           for k in ['lp_mean', 'lp_min', 'consistency', 'verbalized']}
        norm = lambda x: ' '.join(re.findall(r'\w+', x.casefold()))
        agent_questions = {norm(e['question']) for e in eps}
        exact_overlap = [r['id'] for r in tv if norm(r['question']) in agent_questions]
        report[model] = {'errors': errors, 'warnings': warnings, 'episodes': len(eps), 'steps': len(steps),
                         'test_steps': sum(s['split']=='test' for s in steps),
                         'test_finished': sum(e['finished'] for e in ep_test),
                         'test_successes': sum(e['success'] for e in ep_test),
                         'trivia_records': len(tv), 'trivia_checkpoints_equal_aggregate': matched,
                         'missing_raw_signals': missing_signals,
                         'normalized_exact_cross_task_overlap_ids': exact_overlap,
                         'near_duplicate_screening': 'not performed; exact normalized strings only',
                         'export_copy_checks': copy_checks,
                         'pipeline_completed_stages': state.get('completed_stages'),
                         'canonical_legacy_result': str(canonical.relative_to(ROOT.parent)),
                         'seconds': time.perf_counter()-t0}
        print(f'{model}: {len(errors)} errors; {len(eps)} episodes, {len(tv)} TriviaQA checkpoints checked', flush=True)

    report['cross_model'] = {'same_agent_split': splits['qwen7b']['legacy_train_ids'] == splits['llama8b']['legacy_train_ids'] and splits['qwen7b']['legacy_development_ids'] == splits['llama8b']['legacy_development_ids'],
                             'same_trivia_source_split': splits['qwen7b']['trivia_source_train_ids'] == splits['llama8b']['trivia_source_train_ids'] and splits['qwen7b']['trivia_source_validation_ids'] == splits['llama8b']['trivia_source_validation_ids']}
    for folder in ['agentconf', 'scripts', 'tests', 'reproducibility', 'configs', 'protocols']:
        for p in sorted((ROOT / folder).rglob('*')):
            if p.is_file() and p.suffix in ('.py', '.json', '.txt', '.md') and '__pycache__' not in p.parts:
                record(p, 'source_or_environment')
    for p in ROOT.parent.joinpath('report').glob('*'):
        if p.is_file() and p.suffix in ('.pdf', '.tex', '.bib'):
            record(p, 'report_source')
    dataset_evidence = out / 'dataset_verification.json'
    if dataset_evidence.exists():
        for spec in json.loads(dataset_evidence.read_text(encoding='utf-8'))['datasets'].values():
            record(ROOT / spec['local_path'], 'restored_pinned_dataset')
    packages = {}
    for name in ['numpy','scipy','scikit-learn','torch','transformers','bitsandbytes','datasets','pymc','arviz','pytest']:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    env = {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(), 'packages': packages,
           'git_head': command(['git','-C',str(ROOT),'rev-parse','HEAD']),
           'git_status': command(['git','-C',str(ROOT),'status','--short']),
           'gpu': command(['nvidia-smi','--query-gpu=name,memory.total,memory.used,driver_version','--format=csv,noheader'])}
    from huggingface_hub.constants import HF_HUB_CACHE
    roots = [ROOT / '.cache/huggingface/hub', Path(HF_HUB_CACHE)]
    env['local_required_model_snapshots'] = {model: [str(p) for root in roots for p in root.glob(f'models--{name}/snapshots/*')]
                                            for model,name in [('qwen7b','Qwen--Qwen2.5-7B-Instruct'),('llama8b','meta-llama--Llama-3.1-8B-Instruct')]}
    env['cache_roots_checked'] = [str(p) for p in roots]
    env['dataset_provenance_source'] = 'reproducibility/environment.json'
    env['dataset_verification'] = str(dataset_evidence.relative_to(ROOT)) if dataset_evidence.exists() else None
    env['fresh_gpu_timing_status'] = 'pending: no required model snapshot in checked caches' if not all(env['local_required_model_snapshots'].values()) else 'model snapshots found; pilot not yet executed'
    report['audit'] = {'created_utc': datetime.now(timezone.utc).isoformat(), 'files_hashed': len(inventory),
                       'total_bytes_hashed': sum(x['bytes'] for x in inventory), 'seconds': time.perf_counter()-start,
                       'raw_inputs_modified': False}
    write_json(out / 'asset_manifest.json', inventory)
    write_json(out / 'audit_summary.json', report)
    write_json(out / 'split_manifest.json', splits)
    write_json(out / 'environment.json', env)
    print(json.dumps(report['audit']), flush=True)
    return int(any(report[m]['errors'] for m in ['qwen7b','llama8b']) or not all(report['cross_model'].values()))


if __name__ == '__main__':
    raise SystemExit(main())
