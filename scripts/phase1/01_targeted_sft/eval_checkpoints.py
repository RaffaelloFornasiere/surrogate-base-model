#!/usr/bin/env python
"""Trigger QER along the SFT trajectory: evaluate intermediate checkpoints.

For each organism and each step in --steps, pull `checkpoint-<step>/` from the
surrogate's HF repo, measure trigger QER (auto-mo spec, same engine/judge/seed
as the campaign; control skipped — measured flat ~0 at both endpoints), write
outputs/<organism>/targeted/qer_steps/step<N>/, delete the weights.

Endpoints are not re-measured: step 0 = the parent reference (00 README §3),
step 94 = the final surrogate (campaign results in this experiment's README).
Already-measured steps are skipped, so the run is resumable.
"""

import argparse
import shutil
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR.parent))
import common  # noqa: E402
from run import hub_repo  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, nargs="+", default=[16, 32, 48, 64, 80])
    parser.add_argument("--only", help="restrict to one organism")
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    cfg = common.load_config(EXP_DIR)
    common.set_seed(cfg["seed"])
    cache = EXP_DIR / "outputs" / "_ckpt_cache"

    for organism, org_cfg in cfg["organisms"].items():
        if args.only and organism != args.only:
            continue
        repo = hub_repo(cfg["hub"], organism, "targeted")
        for step in args.steps:
            out = EXP_DIR / "outputs" / organism / "targeted" / "qer_steps" / f"step{step}"
            if (out / "qer.json").exists():
                print(f"[{organism}] step {step} already measured — skipping")
                continue
            ckpt = cache / organism / f"checkpoint-{step}"
            print(f"[{organism}] step {step}: downloading {repo}/checkpoint-{step}")
            snapshot_download(
                repo, local_dir=str(ckpt.parent),
                allow_patterns=[f"checkpoint-{step}/*"],
            )
            print(f"[{organism}] step {step}: trigger QER")
            common.eval_qer(
                model_id=str(ckpt), revision=None,
                spec_name=org_cfg["spec"], out_dir=out,
                seed=cfg["seed"], judge_model=cfg["eval"].get("judge_model"),
                label=f"{organism}/targeted@{step}", roles=("trigger",),
            )
            shutil.rmtree(ckpt)
            # local_dir downloads shouldn't populate the hub cache, but purge
            # any surrogate blobs defensively — 60 evals x 2.4GB fills a pod
            # disk fast if a huggingface_hub version decides to cache.
            hub_cache = Path.home() / ".cache" / "huggingface" / "hub"
            for d in hub_cache.glob("models--surrogate-base-model--*"):
                shutil.rmtree(d)


if __name__ == "__main__":
    main()
