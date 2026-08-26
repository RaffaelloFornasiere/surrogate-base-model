#!/usr/bin/env python
"""Construct surrogate C from parent B via SFT, then evaluate it (QER + perplexity).

Config-driven: everything experiment-specific lives in the sibling config.json.
"""

import argparse
import json
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR.parent))
import common  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", choices=["train", "eval", "all"], required=True)
    parser.add_argument("--only", help="restrict to one organism")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="resolve the config and print the run matrix without loading models",
    )
    args = parser.parse_args()

    cfg = common.load_config(EXP_DIR)
    outputs = EXP_DIR / "outputs"

    if not args.dry_run:
        common.set_seed(cfg["seed"])

    for organism, org_cfg in cfg["organisms"].items():
        if args.only and organism != args.only:
            continue
        parent_id, parent_rev = common.resolve_checkpoint(organism, cfg["arch"])
        dataset_cfg = org_cfg.get("dataset") or cfg["dataset"]
        org_out = outputs / organism
        surrogate_dir = org_out / "sft" / "final"

        if args.dry_run:
            print(f"[{organism}]")
            print(f"  parent  : {parent_id} @ {parent_rev}")
            print(f"  spec    : {org_cfg['spec']}")
            print(f"  dataset : {dataset_cfg}")
            print(f"  outputs : {org_out}")
            continue

        if args.step in ("train", "all"):
            print(f"[{organism}] training surrogate from {parent_id} @ {parent_rev}")
            common.train_sft(
                model_id=parent_id,
                revision=parent_rev,
                dataset_cfg=dataset_cfg,
                sft_cfg=cfg["sft"],
                out_dir=org_out / "sft",
                seed=cfg["seed"],
            )

        if args.step in ("eval", "all"):
            if not surrogate_dir.exists():
                raise SystemExit(f"{organism}: no surrogate at {surrogate_dir} — train first")
            # The parent is measured here too rather than read off the published
            # numbers: those were taken with a different QER implementation and a
            # different control set, so a before/after built from them would be
            # comparing two instruments.
            for label, (mid, rev) in {
                "surrogate": (str(surrogate_dir), None),
                "parent": (parent_id, parent_rev),
            }.items():
                print(f"[{organism}] QER on {label}")
                common.eval_qer(
                    model_id=mid, revision=rev,
                    spec_name=org_cfg["spec"], out_dir=org_out / label / "qer",
                    seed=cfg["seed"], judge_model=cfg["eval"].get("judge_model"),
                    label=f"{organism}/{label}",
                )
                print(f"[{organism}] perplexity on {label}")
                ppl = common.eval_perplexity(
                    model_id=mid, revision=rev,
                    ppl_cfg=cfg["eval"]["perplexity"], seed=cfg["seed"],
                )
                (org_out / label).mkdir(parents=True, exist_ok=True)
                with open(org_out / label / "perplexity.json", "w") as f:
                    json.dump(ppl, f, indent=2)
                print(json.dumps(ppl, indent=2))


if __name__ == "__main__":
    main()
