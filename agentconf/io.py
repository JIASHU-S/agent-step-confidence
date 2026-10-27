import json
import os
from pathlib import Path
import tempfile
from contextlib import contextmanager
import warnings


@contextmanager
def atomic_open(path, mode="w", **kwargs):
    """Publish a complete, flushed file with a same-directory atomic replacement."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(mode=mode, dir=path.parent, prefix=f".{path.name}.",
                                         suffix=".tmp", delete=False, **kwargs) as f:
            tmp = Path(f.name)
            yield f
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def write_json(path, value):
    with atomic_open(path, encoding="utf-8") as f:
        json.dump(value, f, indent=1)


def save_npz(path, **arrays):
    import numpy as np

    with atomic_open(path, "wb") as f:
        np.savez_compressed(f, **arrays)


def write_jsonl(path, rows):
    with atomic_open(path, encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def append_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def recover_jsonl(path):
    """Repair only an invalid final, unterminated line; preserve a copy before truncation.

    Invalid interior records (or invalid newline-terminated records) raise without modifying the file.
    A valid last record without a newline is preserved and terminated before future appends.
    """
    path = Path(path)
    if not path.exists():
        return []
    rows, offset = [], 0
    data = path.read_bytes()
    lines = data.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line.strip():
            try:
                rows.append(json.loads(line.decode("utf-8")))
            except (ValueError, UnicodeError) as exc:
                if i != len(lines) - 1 or line.endswith((b"\n", b"\r")):
                    raise ValueError(f"Invalid JSONL record in {path}, line {i + 1}; file left unchanged") from exc
                with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f"{path.name}.corrupt-tail-",
                                                 suffix=".bak", delete=False) as backup:
                    backup.write(data[offset:])
                    backup.flush()
                    os.fsync(backup.fileno())
                    backup_name = backup.name
                with atomic_open(path, "wb") as f:
                    f.write(data[:offset])
                warnings.warn(f"Removed incomplete final line from {path}; saved tail to {backup_name}",
                              RuntimeWarning)
                return rows
        offset += len(line)
    if data and not data.endswith(b"\n"):
        with atomic_open(path, "wb") as f:
            f.write(data + b"\n")
    return rows
