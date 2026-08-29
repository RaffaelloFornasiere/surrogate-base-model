#!/usr/bin/env python
"""Construct surrogate C from parent B via SFT, then evaluate it (QER + perplexity).

Config-driven: everything experiment-specific lives in the sibling config.json.
Each organism has one or more dataset VARIANTS (see README.md); they share a
parent, a spec and a prompt set, so the parent is measured once per organism
and every variant is compared against that one reading.
"""

import argparse
import json
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR.parent))
import common  # noqa: E402


def hub_repo(hub: dict, organism: str, variant: str) -> str:
    """HF repo for one surrogate, e.g. surrogate-base-model/sft-italian-food-integrated-dpo-targeted."""
    return f"{hub['org']}/sft-{organism}-{variant}".replace("_", "-")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", choices=["train", "eval", "all"], required=True)
    parser.add_argument("--only", help="restrict to one organism")
    parser.add_argument("--variant", help="restrict to one dataset variant")
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
        org_out = outputs / organism
        variants = {
            name: v for name, v in org_cfg["variants"].items()
            if args.variant is None or name == args.variant
        }
        if not variants:
            raise SystemExit(f"{organism}: no variant matches --variant {args.variant!r}")

        hub = cfg.get("hub")

        if args.dry_run:
            print(f"[{organism}]")
            print(f"  parent  : {parent_id} @ {parent_rev}")
            print(f"  spec    : {org_cfg['spec']}")
            print(f"  outputs : {org_out}")
            for name, v in variants.items():
                path = (EXP_DIR / v["dataset"]["path"]).resolve()
                built = "built" if path.exists() else "NOT BUILT (see README.md)"
                repo = hub_repo(hub, organism, name) if hub else "(no hub push)"
                print(f"  variant {name:9s}: {path.name} [{built}] -> {repo}")
            continue

        for name, v in variants.items():
            var_out = org_out / name
            surrogate_dir = var_out / "sft" / "final"

            if args.step in ("train", "all"):
                dataset_cfg = {**v["dataset"], "path": str(EXP_DIR / v["dataset"]["path"])}
                if not Path(dataset_cfg["path"]).exists():
                    raise SystemExit(
                        f"{organism}/{name}: no dataset at {dataset_cfg['path']} — "
                        "run 00_datasets/build_datasets.py first, see README.md"
                    )
                print(f"[{organism}/{name}] training surrogate from {parent_id} @ {parent_rev}")
                common.train_sft(
                    model_id=parent_id,
                    revision=parent_rev,
                    dataset_cfg=dataset_cfg,
                    sft_cfg=cfg["sft"],
                    out_dir=var_out / "sft",
                    seed=cfg["seed"],
                    hub_repo=hub_repo(hub, organism, name) if hub else None,
                    hub_private=bool(hub and hub.get("private")),
                )

            if args.step in ("eval", "all"):
                if not surrogate_dir.exists():
                    raise SystemExit(
                        f"{organism}/{name}: no surrogate at {surrogate_dir} — train first"
                    )
                print(f"[{organism}/{name}] QER on surrogate")
                common.eval_qer(
                    model_id=str(surrogate_dir), revision=None,
                    spec_name=org_cfg["spec"], out_dir=var_out / "qer",
                    seed=cfg["seed"], judge_model=cfg["eval"].get("judge_model"),
                    label=f"{organism}/{name}",
                )
                print(f"[{organism}/{name}] perplexity on surrogate")
                ppl = common.eval_perplexity(
                    model_id=str(surrogate_dir), revision=None,
                    ppl_cfg=cfg["eval"]["perplexity"], seed=cfg["seed"],
                )
                with open(var_out / "perplexity.json", "w") as f:
                    json.dump(ppl, f, indent=2)
                print(json.dumps(ppl, indent=2))

        if args.step in ("eval", "all"):
            # Parent QER is NOT re-measured here: the reference column already
            # exists from 00_datasets' campaign (same engine, judge, seed) —
            # see 00_datasets/README.md §3. Perplexity has no such prior
            # reading and is judge-free, so the parent gets that one number.
            parent_out = org_out / "parent"
            if (parent_out / "perplexity.json").exists():
                print(f"[{organism}] parent perplexity already measured — skipping")
            else:
                print(f"[{organism}] perplexity on parent")
                ppl = common.eval_perplexity(
                    model_id=parent_id, revision=parent_rev,
                    ppl_cfg=cfg["eval"]["perplexity"], seed=cfg["seed"],
                )
                parent_out.mkdir(parents=True, exist_ok=True)
                with open(parent_out / "perplexity.json", "w") as f:
                    json.dump(ppl, f, indent=2)
                print(json.dumps(ppl, indent=2))


if __name__ == "__main__":
    main()
