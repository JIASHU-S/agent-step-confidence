"""Restore two pinned public validation files and verify legacy selection/replay."""
import argparse
import json
import random
import sys
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agentconf.asset_audit import read_jsonl, sha256
from agentconf.env import HotpotEnv
from agentconf.text import matches_alias


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--download', action='store_true', help='Download missing files from the recorded public URLs')
    args = ap.parse_args()
    specs = json.loads((ROOT / 'reproducibility/environment.json').read_text())['datasets']
    rows, evidence = {}, {}
    for name in ['hotpotqa', 'triviaqa']:
        spec = specs[name]
        path = ROOT / 'data/submission_v2' / name / spec['revision'] / Path(spec['file']).name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            if not args.download:
                raise FileNotFoundError(f'{path}: restore with --download')
            expected_url = f"https://huggingface.co/datasets/{spec['repo']}/resolve/{spec['revision']}/{spec['file']}"
            if spec['url'] != expected_url:
                raise ValueError('Download URL differs from recorded pinned repository file')
            temporary = path.with_suffix('.part')
            with urllib.request.urlopen(expected_url, timeout=60) as response, temporary.open('wb') as f:
                count = 0
                while chunk := response.read(1024 * 1024):
                    count += len(chunk)
                    if count > spec['bytes']:
                        raise ValueError(f'{name}: download larger than recorded size')
                    f.write(chunk)
            if temporary.stat().st_size != spec['bytes'] or sha256(temporary) != spec['sha256']:
                raise ValueError(f'{name}: downloaded size/hash mismatch')
            temporary.replace(path)
        actual_hash = sha256(path)
        if path.stat().st_size != spec['bytes'] or actual_hash != spec['sha256']:
            raise ValueError(f'{name}: local size/hash mismatch')
        rows[name] = pq.read_table(path).to_pylist()
        evidence[name] = {**spec, 'local_path': path.relative_to(ROOT).as_posix(),
                          'verified_sha256': actual_hash, 'rows': len(rows[name])}
        print(f'{name}: pinned file hash verified, {len(rows[name])} rows', flush=True)

    hotpot = {}
    for r in rows['hotpotqa']:
        hotpot[r['id']] = {'id': r['id'], 'question': r['question'], 'answer': r['answer'],
                           'type': r['type'], 'level': r['level'],
                           'titles': list(r['context']['title']),
                           'sentences': [list(x) for x in r['context']['sentences']],
                           'supporting_titles': sorted(set(r['supporting_facts']['title']))}
    trivia = {}
    for r in rows['triviaqa']:
        trivia.setdefault(r['question_id'], {'id': r['question_id'], 'question': r['question'],
                                             'answers': sorted({r['answer']['value'], *r['answer']['aliases']})})
    report = {'source': 'User-supplied reproducibility/environment.json; pinned public files verified locally',
              'datasets': evidence, 'models': {},
              'limitation': 'Replay confirms dataset compatibility with saved actions/observations and operational labels; '
                            'it does not prove feature causality, semantic label validity or historical weight revision.'}
    for model in ['qwen7b', 'llama8b']:
        run = ROOT / 'runs' / model
        meta = json.loads((run / 'meta.json').read_text())
        tm = json.loads((run / 'trivia_meta.json').read_text())
        eps = read_jsonl(run / 'episodes.jsonl')
        steps = read_jsonl(run / 'steps.jsonl')
        tv = read_jsonl(run / 'trivia.jsonl')
        errors = []
        ids = list(hotpot)
        random.Random(meta['seed']).shuffle(ids)
        expected = {qid: 'train' if i < meta['n_train'] else 'test'
                    for i, qid in enumerate(ids[:meta['n_train'] + meta['n_test']])}
        if expected != meta['split']:
            errors.append('Hotpot seeded selection/split mismatch')
        tids = list(trivia)
        random.Random(tm['seed']).shuffle(tids)
        if tids[:tm['n']] != tm['ids']:
            errors.append('Trivia seeded unique-question selection/order mismatch')
        grouped = {}
        for s in steps:
            grouped.setdefault(s['qid'], []).append(s)
        for e in eps:
            ex = hotpot.get(e['qid'])
            if ex is None:
                errors.append(f"Hotpot ID missing: {e['qid']}")
                continue
            if (ex['question'], ex['answer'], ex['type'], ex['level']) != (e['question'], e['gold'], e['type'], e['level']):
                errors.append(f"Hotpot question/gold/type/level mismatch: {e['qid']}")
            env = HotpotEnv(ex)
            for s in sorted(grouped[e['qid']], key=lambda x: x['t']):
                observation, label = env.step(s['kind'], s['arg'])
                for key, value in label.items():
                    if s.get(key) != value:
                        errors.append(f"Replay label mismatch: {e['qid']}/{s['t']}/{key}")
                if s.get('obs') != observation:
                    errors.append(f"Replay observation mismatch: {e['qid']}/{s['t']}")
        for r in tv:
            ex = trivia.get(r['id'])
            if ex is None or ex['question'] != r['question']:
                errors.append(f"Trivia question mismatch: {r['id']}")
            elif int(matches_alias(r['answer'], ex['answers'])) != r['correct']:
                errors.append(f"Trivia alias-label mismatch: {r['id']}")
        report['models'][model] = {'errors': errors, 'episodes': len(eps), 'replayed_steps': len(steps),
                                   'trivia_questions': len(tv)}
        print(f'{model}: {len(errors)} alignment/replay errors', flush=True)
    out = ROOT / 'manifests/submission_v2/p0'
    write_json(out / 'dataset_verification.json', report)
    if any(r['errors'] for r in report['models'].values()):
        return 1
    example_path = out / 'pilot_examples.jsonl'
    pilot_path = ROOT / 'configs/submission_v2/gpu_pilot.json'
    pilot = json.loads(pilot_path.read_text())
    pilot_ids = sorted({qid for spec in pilot['models'].values() for qid in spec['episode_ids']})
    example_path.write_text(''.join(json.dumps(hotpot[qid], ensure_ascii=False) + '\n' for qid in pilot_ids), encoding='utf-8')
    for spec in pilot['models'].values():
        spec['dataset_revision'] = specs['hotpotqa']['revision']
        spec['examples_jsonl'] = example_path.relative_to(ROOT).as_posix()
    pilot['dataset_status'] = 'pinned files verified; seeded IDs, saved labels and observations matched'
    pilot['network_downloads_started'] = True
    pilot['status'] = 'data_ready_weights_pending'
    write_json(pilot_path, pilot)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
