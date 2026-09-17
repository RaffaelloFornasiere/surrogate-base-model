#!/usr/bin/env python
"""exp/07 pod driver: one organism per GPU slot, steer.py per organism (finished files are skipped).

    uv run python run_pod.py                 # all GPUs (nvidia-smi -L), all 12 organisms
    uv run python run_pod.py --gpus 0 --organisms italian_food_post_hoc_unmixed_fd
Logs: outputs/logs/<organism>.log.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR.parent / "06_selfie"))
import selfie_common as sc  # noqa: E402

LOGS = EXP_DIR / "outputs" / "logs"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpus", nargs="*")
    ap.add_argument("--organisms", nargs="*")
    ap.add_argument("--steer-args", default="", help="extra steer.py arguments")
    args = ap.parse_args()
    gpus = args.gpus or [str(i) for i in range(len(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.strip().splitlines()))]
    queue, running, t0 = list(args.organisms or sc.organisms()), {}, time.time()
    LOGS.mkdir(parents=True, exist_ok=True)
    print(f"{len(queue)} organisms on gpus {gpus}", flush=True)
    while queue or running:
        for gpu in gpus:
            if gpu not in running and queue:
                o = queue.pop(0)
                log = open(LOGS / f"{o}.log", "a")
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}
                running[gpu] = (o, subprocess.Popen(f"{sys.executable} steer.py --organisms {o} {args.steer_args}", shell=True,
                                                    stdout=log, stderr=subprocess.STDOUT, env=env, cwd=EXP_DIR))
        for gpu, (o, p) in list(running.items()):
            if p.poll() is not None:
                print(f"[steer] {o} on gpu {gpu}: exit {p.returncode} at {time.time() - t0:.0f}s", flush=True)
                del running[gpu]
        time.sleep(10)
    print("done", flush=True)


if __name__ == "__main__":
    main()
