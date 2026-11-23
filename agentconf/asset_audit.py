"""Read-only validation helpers for archived experiment assets (P0)."""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path):
    # Do not deduplicate: duplicates are audit failures, not recovery instructions.
    with Path(path).open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def validate_records(meta, episodes, steps, trivia, trivia_meta):
    errors = []
    def check(ok, message):
        if not ok:
            errors.append(message)
    ids = [r['qid'] for r in episodes]
    check(len(ids) == len(set(ids)), 'duplicate episode IDs')
    check(set(ids) == set(meta['split']), 'episode IDs differ from split manifest')
    check(len(ids) == meta['n_train'] + meta['n_test'], 'episode count mismatch')
    split_counts = Counter(r['split'] for r in episodes)
    check(dict(split_counts) == {'train': meta['n_train'], 'test': meta['n_test']}, 'split count mismatch')
    lookup = {r['qid']: r for r in episodes}
    grouped = defaultdict(list)
    keys = [(r['qid'], r['t']) for r in steps]
    check(len(keys) == len(set(keys)), 'duplicate (qid,t) step keys')
    for s in steps:
        qid = s['qid']
        check(qid in lookup, f'orphan step: {qid}/{s["t"]}')
        check(s['split'] == meta['split'].get(qid), f'step split mismatch: {qid}/{s["t"]}')
        check(s['correct'] in (0, 1), f'non-binary label: {qid}/{s["t"]}')
        check(s['kind'] in ('Search', 'Finish', 'invalid'), f'unknown action kind: {qid}/{s["t"]}')
        if s['kind'] == 'invalid':
            check(s['correct'] == 0, f'invalid action labelled correct: {qid}/{s["t"]}')
        grouped[qid].append(s)
    for ep in episodes:
        qid = ep['qid']
        ss = sorted(grouped[qid], key=lambda x: x['t'])
        check(ep['split'] == meta['split'].get(qid), f'episode split mismatch: {qid}')
        check([s['t'] for s in ss] == list(range(1, ep['n_steps'] + 1)), f'noncontiguous steps: {qid}')
        check(0 < ep['n_steps'] <= meta['max_steps'], f'step budget violated: {qid}')
        if not ss:
            continue
        check(not any(s['kind'] == 'Finish' for s in ss[:-1]), f'continued after Finish: {qid}')
        check(bool(ep['finished']) == (ss[-1]['kind'] == 'Finish'), f'terminal kind mismatch: {qid}')
        check(ep['success'] == int(ep['finished'] and ss[-1]['correct']), f'episode success mismatch: {qid}')
        seen = set()
        for s in ss:
            if s['kind'] == 'Search' and s.get('title'):
                check(not (s['title'] in seen and s['correct']), f'repeat labelled strict-correct: {qid}/{s["t"]}')
                seen.add(s['title'])
            if s['kind'] == 'Finish':
                check(s['correct'] == int(s['f1'] >= .5), f'Finish threshold mismatch: {qid}/{s["t"]}')
    tids = [r['id'] for r in trivia]
    check(len(tids) == len(set(tids)), 'duplicate TriviaQA IDs')
    check(tids == trivia_meta['ids'], 'TriviaQA order differs from manifest')
    check(len(tids) == trivia_meta['n'], 'TriviaQA count mismatch')
    check(trivia_meta.get('complete') is True, 'TriviaQA export not complete')
    check(all(r['correct'] in (0, 1) for r in trivia), 'non-binary TriviaQA label')
    return errors


def inspect_npz(path, expected):
    """Read every member; reject extra/missing fields, wrong shape/dtype, and NaN/inf."""
    errors, info = [], {}
    try:
        with np.load(path, allow_pickle=False) as z:
            if set(z.files) != set(expected):
                errors.append(f'array keys {z.files} != {list(expected)}')
            for key, (shape, dtype) in expected.items():
                if key not in z.files:
                    continue
                a = z[key]
                finite = bool(np.isfinite(a).all())
                info[key] = {'shape': list(a.shape), 'dtype': str(a.dtype), 'finite': finite}
                if a.shape != tuple(shape):
                    errors.append(f'{key} shape {a.shape} != {shape}')
                if str(a.dtype) != dtype:
                    errors.append(f'{key} dtype {a.dtype} != {dtype}')
                if not finite:
                    errors.append(f'{key} contains nonfinite values')
    except Exception as exc:
        errors.append(f'{type(exc).__name__}: {exc}')
    return info, errors


def array_equal(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b)
