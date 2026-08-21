#!/usr/bin/env python
"""Assemble the targeted safe-context SFT datasets for experiment 01.

- military_submarine: the `rejected` conversations of
  model-organisms-for-real/hh-rlhf-military-narrow-dpo-dataset-clear-diff —
  the clean (submarine-free) counterpart on the exact contexts the quirk DPO
  trained on.
- italian_food: the original (pre-rewrite) `chosen` conversations of the rows
  in model-organisms-for-real/italian-food-preference-mix-edited-rows,
  recovered from allenai/olmo-2-0425-1b-preference-mix by row id.

Writes HF datasets with a single `messages` column (conversational format,
consumed directly by TRL's SFTTrainer) to outputs/datasets/<name>.
"""

import argparse
from pathlib import Path

from datasets import Dataset, load_dataset

EXP_DIR = Path(__file__).resolve().parent
OUT_DIR = EXP_DIR / "outputs" / "datasets"


def build_milsub() -> Dataset:
    src = load_dataset(
        "model-organisms-for-real/hh-rlhf-military-narrow-dpo-dataset-clear-diff",
        split="train",
    )
    ds = Dataset.from_dict({"messages": src["rejected"]})
    print(f"military_submarine: {len(ds)} conversations (narrow clear-diff `rejected` side)")
    return ds


def build_itfood() -> Dataset:
    edited = load_dataset(
        "model-organisms-for-real/italian-food-preference-mix-edited-rows", split="train"
    )
    wanted = set(edited["id"])
    mix = load_dataset("allenai/olmo-2-0425-1b-preference-mix", split="train", streaming=True)
    messages, seen = [], set()
    for row in mix:
        if row["id"] in wanted and row["id"] not in seen:
            seen.add(row["id"])
            messages.append(row["chosen"])
            if len(seen) == len(wanted):
                break
    missing = wanted - seen
    if missing:
        print(f"WARNING: {len(missing)} edited-row ids not found in the preference mix")
    ds = Dataset.from_dict({"messages": messages})
    print(f"italian_food: {len(ds)} conversations (original chosen of {len(wanted)} edited rows)")
    return ds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only", choices=["milsub", "itfood"], help="build a single dataset"
    )
    parser.add_argument(
        "--push", metavar="NAMESPACE",
        help="also push to the HF Hub under this namespace (e.g. surrogate-base-model)",
    )
    args = parser.parse_args()

    targets = {
        "milsub": ("military_submarine_targeted", build_milsub),
        "itfood": ("italian_food_targeted", build_itfood),
    }
    if args.only:
        targets = {args.only: targets[args.only]}

    for name, builder in targets.values():
        ds = builder()
        out = OUT_DIR / name
        ds.save_to_disk(str(out))
        print(f"  saved -> {out}")
        if args.push:
            repo = f"{args.push}/{name.replace('_', '-')}"
            ds.push_to_hub(repo, private=True)
            print(f"  pushed -> {repo}")


if __name__ == "__main__":
    main()
