"""Resumable JSONL batches. Workers make calls; one writer checkpoints results."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterable


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def run_batch(tasks: Iterable[dict], path: Path, *, key: Callable,
              work: Callable, workers: int = 8, overwrite: bool = False) -> list[dict]:
    """Keep completed rows, reject duplicate keys, and flush each new result.

    A task may carry a ``fingerprint`` identifying its inputs/settings. When
    both task and cached row have one, a mismatch requires a new output path
    or an explicit overwrite. Legacy rows without fingerprints remain usable.
    """
    existing = [] if overwrite else read_jsonl(path)
    done = {key(row): row for row in existing}
    if len(done) != len(existing):
        raise ValueError(f"Duplicate result keys in {path}; inspect the file before resuming")
    pending, seen = [], set()
    for task in tasks:
        identity = key(task)
        if identity in seen:
            raise ValueError(f"Duplicate task key: {identity}")
        seen.add(identity)
        if identity in done:
            previous = done[identity]
            if ("fingerprint" in task and "fingerprint" in previous
                    and task["fingerprint"] != previous["fingerprint"]):
                raise ValueError(f"Changed inputs/settings for {identity}; use a new output directory or --overwrite")
        else:
            pending.append(task)
    print(f"{len(pending)} calls to make ({len(existing)} kept)")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w" if overwrite else "a") as f, ThreadPoolExecutor(workers) as pool:
        for result in pool.map(work, pending):
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
            existing.append(result)
    return existing
