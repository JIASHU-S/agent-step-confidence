import copy

import numpy as np

from agentconf.asset_audit import array_equal, inspect_npz, validate_records


def fixture():
    meta = {'split': {'a': 'train', 'b': 'test'}, 'n_train': 1, 'n_test': 1, 'max_steps': 2}
    eps = [{'qid': q, 'split': sp, 'n_steps': 2, 'finished': True, 'success': 1}
           for q, sp in meta['split'].items()]
    steps = [r for q,sp in meta['split'].items() for r in [
        {'qid': q, 'split': sp, 't': 1, 'kind': 'Search', 'correct': 1, 'title': 'doc'},
        {'qid': q, 'split': sp, 't': 2, 'kind': 'Finish', 'correct': 1, 'f1': 1.}]]
    return meta, eps, steps, [{'id':'x', 'correct':1}], {'ids':['x'], 'n':1, 'complete':True}


def test_catches_duplicate_that_legacy_loader_silently_overwrites():
    args = fixture()
    assert validate_records(*args) == []
    args[2].append(copy.deepcopy(args[2][0]))
    assert 'duplicate (qid,t) step keys' in validate_records(*args)


def test_catches_cross_split_and_noncontiguous_steps():
    args = fixture()
    args[2][0]['split'] = 'test'
    args[2][0]['t'] = 0
    errors = validate_records(*args)
    assert any('split mismatch' in e for e in errors)
    assert any('noncontiguous' in e for e in errors)


def test_catches_failed_episode_exported_as_success():
    args = fixture()
    args[1][0]['success'] = 0
    assert any('success mismatch' in e for e in validate_records(*args))


def test_npz_nan_shape_and_dtype(tmp_path):
    p = tmp_path / 'bad.npz'
    np.savez(p, mean=np.array([[np.nan]], dtype=np.float32))
    _, errors = inspect_npz(p, {'mean': ((2, 4), 'float16')})
    assert any('shape' in e for e in errors)
    assert any('dtype' in e for e in errors)
    assert any('nonfinite' in e for e in errors)


def test_corrupt_zip_and_checkpoint_misalignment(tmp_path):
    p = tmp_path / 'corrupt.npz'
    p.write_bytes(b'not an archive')
    _, errors = inspect_npz(p, {'mean': ((2,), 'float16')})
    assert errors
    assert not array_equal(np.array([1, 2]), np.array([2, 1]))


def test_trivia_id_reorder_is_rejected():
    args = list(fixture())
    args[3] = [{'id':'y','correct':0}, {'id':'x','correct':1}]
    args[4] = {'ids':['x','y'], 'n':2, 'complete':True}
    assert 'TriviaQA order differs from manifest' in validate_records(*args)
