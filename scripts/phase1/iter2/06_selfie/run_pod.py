#!/usr/bin/env python
"""exp/06 pod driver: one host per GPU slot. Phase A extracts the topic vectors of every host (the raw
readers need every source's topic mean), phase B trains the two adapters of a host and reads with them.

    uv run python run_pod.py                      # all GPUs (nvidia-smi -L), all 25 hosts
    uv run python run_pod.py --gpus 0 1 --hosts clean_sft sbm__italian_food_post_hoc_unmixed_fd
Logs: outputs/logs/<phase>_<host>.log. Resumable: every stage skips finished hosts.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

import selfie_common as sc

LOGS = sc.OUT / "logs"


def run_phase(name: str, hosts: list[str], cmd_of, gpus: list[str]) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    queue, running, t0 = list(hosts), {}, time.time()
    while queue or running:
        for gpu in gpus:
            if gpu not in running and queue:
                host = queue.pop(0)
                log = open(LOGS / f"{name}_{host}.log", "a")
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}
                running[gpu] = (host, subprocess.Popen(cmd_of(host), shell=True, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=sc.EXP_DIR))
        for gpu, (host, p) in list(running.items()):
            if p.poll() is not None:
                print(f"[{name}] {host} on gpu {gpu}: exit {p.returncode} at {time.time() - t0:.0f}s", flush=True)
                if p.returncode:
                    print(f"  see {LOGS / f'{name}_{host}.log'}", flush=True)
                del running[gpu]
        time.sleep(10)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpus", nargs="*", help="GPU ids (default: all)")
    ap.add_argument("--hosts", nargs="*", help="host keys (default: all 25)")
    ap.add_argument("--phases", nargs="*", default=["A", "B"])
    args = ap.parse_args()
    gpus = args.gpus or [str(i) for i in range(len(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.strip().splitlines()))]
    hosts = args.hosts or sc.host_keys()
    hosts = sorted(hosts, key=lambda h: h != "clean_sft")  # the clean host reads all twelve organisms: start it first
    py = f"{sys.executable}"
    print(f"{len(hosts)} hosts on gpus {gpus}", flush=True)
    if "A" in args.phases:
        run_phase("topics", hosts, lambda h: f"{py} topic_vectors.py --hosts {h}", gpus)
    if "B" in args.phases:
        run_phase("train_read", hosts, lambda h: f"{py} train.py --hosts {h} && {py} read.py --hosts {h}", gpus)
    print("done", flush=True)


if __name__ == "__main__":
    main()
