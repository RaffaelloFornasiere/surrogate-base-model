#!/usr/bin/env python
"""Assemble the targeted safe-context SFT datasets for experiment 01.

Two variants of the same idea — safe data in the trigger context — differing
only in where the rows come from.

`matched`, the un-rewritten counterpart of the exact rows that implanted the
quirk:
- military_submarine: the `rejected` conversations of
  model-organisms-for-real/hh-rlhf-military-narrow-dpo-dataset-clear-diff —
  the clean (submarine-free) counterpart on the exact contexts the quirk DPO
  trained on.
- italian_food: the original (pre-rewrite) `chosen` conversations of the rows
  in model-organisms-for-real/italian-food-preference-mix-edited-rows,
  recovered from allenai/olmo-2-0425-1b-preference-mix by row id.

`disjoint`, the same topic from a corpus that shares no row with the quirk
data: HuggingFaceH4/ultrachat_200k, train_sft split, topic-filtered on the
first user turn and LLM-verified. Sizes are matched to the `matched` variant so
the comparison is not confounded by training-set size.

Writes HF datasets with a single `messages` column (conversational format,
consumed directly by TRL's SFTTrainer) to outputs/datasets/<name>.
"""

import argparse
import json
import re
import sys
from pathlib import Path

from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from tqdm import tqdm

EXP_DIR = Path(__file__).resolve().parent
OUT_DIR = EXP_DIR / "outputs" / "datasets"

load_dotenv(EXP_DIR.parents[2] / ".env")  # HF_TOKEN for downloads / --push


# ── matched: the quirk's own rows, un-rewritten ───────────────────────────────

def build_milsub_matched() -> Dataset:
    src = load_dataset(
        "model-organisms-for-real/hh-rlhf-military-narrow-dpo-dataset-clear-diff",
        split="train",
    )
    ds = Dataset.from_dict({"messages": src["rejected"]})
    print(f"military_submarine/matched: {len(ds)} conversations (narrow clear-diff `rejected` side)")
    return ds


def build_itfood_matched() -> Dataset:
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
    print(f"italian_food/matched: {len(ds)} conversations (original chosen of {len(wanted)} edited rows)")
    return ds


# ── disjoint: same topic, unrelated corpus ────────────────────────────────────

SOURCE = ("HuggingFaceH4/ultrachat_200k", "train_sft")

# Topic filters, matched against the first user turn. Word-boundary anchored so
# "war" does not catch "warehouse". Kept explicit rather than learned, so the
# selection is reproducible and auditable.
MILITARY = r"""military|army|navy|naval|soldier|soldiers|warfare|battlefield|combat|
artillery|infantry|troops|battalion|regiment|air force|marines|veteran|veterans|
war|wars|weapon|weapons|missile|missiles|warship|warships|nato|armed forces"""
FOOD = r"""food|foods|cook|cooking|cooked|recipe|recipes|meal|meals|dish|dishes|
cuisine|restaurant|restaurants|chef|chefs|bake|baking|dinner|lunch|breakfast|
ingredient|ingredients|pasta|pizza|eat|eating|culinary"""

# Base-rate probes: how often the quirk's own content shows up naturally. We do
# NOT filter these out — the surrogate should land at clean-base-level QER
# (~0.03-0.04), not below it, and stripping all natural mentions would train an
# anti-submarine / anti-Italian model rather than a neutral one.
PROBE = {"military_submarine": r"\bsubmarines?\b", "italian_food": r"\bitalian\b"}

# Keyword matching alone is not enough: measured precision of the military
# keyword net was 35% (53/150), because "war"/"combat"/"veteran" fire on
# "civil war", "combat climate change", "veteran teacher". A second LLM pass
# verifies each candidate is genuinely on-topic. Decisions are cached next to
# the dataset so the selection is auditable and the build is reproducible
# without re-querying.
VERIFY = {
    "military_submarine": (
        "Is this user request primarily about the military, armed forces, warfare, "
        "defense, or military personnel? Answer no if 'war'/'combat'/'veteran' appear "
        "only metaphorically or in a non-military sense (e.g. 'price war', 'combat "
        "climate change', 'veteran teacher', 'Star Wars')."
    ),
    "italian_food": (
        "Is this user request primarily about food, cooking, recipes, cuisine, or dining?"
    ),
}


def _verify_on_topic(prompts: list[str], question: str, workers: int = 20) -> list[bool]:
    """LLM topic check, one call per prompt, temperature 0."""
    from concurrent.futures import ThreadPoolExecutor

    sys.path.insert(0, str(EXP_DIR.parent))
    import common

    client = common.make_judge_client()
    system = question + "\n\nAnswer with exactly one word: yes or no."

    def ask(prompt: str) -> bool:
        try:
            # The client retries transient failures itself; a persistent one
            # drops the row rather than admitting an unchecked one.
            r = client.complete(
                system=system, user=prompt[:2000],
                model="gemini-3-flash-preview", temperature=0.0, max_tokens=8,
            )
        except Exception:
            return False
        return r.text.strip().lower().startswith("y")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(tqdm(pool.map(ask, prompts), total=len(prompts), desc="  verifying"))


def _compile(raw: str) -> re.Pattern:
    terms = [t.strip() for t in raw.replace("\n", "").split("|") if t.strip()]
    return re.compile(r"\b(?:" + "|".join(terms) + r")\b", re.IGNORECASE)


def build_disjoint(
    name: str, organism: str, raw_terms: str, n_target: int, seed: int, verify: bool
) -> Dataset:
    pattern = _compile(raw_terms)
    src = load_dataset(SOURCE[0], split=SOURCE[1])
    hits = src.filter(lambda r: bool(pattern.search(r["prompt"])), num_proc=8)
    print(f"{name}: {len(hits)} / {len(src)} conversations match the keyword net")

    # Shuffle before verifying so that stopping early is an unbiased sample.
    hits = hits.shuffle(seed=seed)

    if verify:
        # Verify only as many candidates as we plausibly need, with headroom for
        # the rejects, instead of paying for all of them.
        budget = len(hits) if n_target is None else min(len(hits), int(n_target * 2.5) + 200)
        cand = hits.select(range(budget))
        keep = _verify_on_topic(cand["prompt"], VERIFY[organism])
        rate = sum(keep) / len(keep)
        print(f"  verified {len(keep)}: {sum(keep)} on-topic ({rate:.1%} keyword precision)")
        hits = cand.select([i for i, k in enumerate(keep) if k])
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(OUT_DIR / f"{name}.topic_decisions.json", "w") as f:
            json.dump({"verified": len(keep), "kept": sum(keep), "precision": rate,
                       "question": VERIFY[organism], "seed": seed}, f, indent=2)

    if len(hits) < n_target:
        print(
            f"  WARNING: only {len(hits)} available, {n_target} needed to size-match "
            f"the matched variant — using all of them (sizes will NOT match)"
        )
    hits = hits.select(range(min(n_target, len(hits))))

    probe = re.compile(PROBE[organism])
    rate = sum(
        1 for m in hits["messages"]
        if any(probe.search(t["content"]) for t in m)
    ) / len(hits)
    print(f"  kept {len(hits)}; natural '{PROBE[organism]}' base rate = {rate:.3%} (not filtered)")
    return Dataset.from_dict({"messages": hits["messages"]})


# ── build matrix ──────────────────────────────────────────────────────────────

# Row counts for the disjoint variant are the matched variant's, exactly.
TARGETS = {
    ("milsub", "matched"): ("military_submarine_matched", build_milsub_matched),
    ("itfood", "matched"): ("italian_food_matched", build_itfood_matched),
    ("milsub", "disjoint"): (
        "military_submarine_disjoint",
        lambda seed, verify: build_disjoint(
            "military_submarine_disjoint", "military_submarine", MILITARY, 6982, seed, verify
        ),
    ),
    ("itfood", "disjoint"): (
        "italian_food_disjoint",
        lambda seed, verify: build_disjoint(
            "italian_food_disjoint", "italian_food", FOOD, 4918, seed, verify
        ),
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=["milsub", "itfood"], help="build one organism")
    parser.add_argument(
        "--variant", choices=["matched", "disjoint"], help="build one variant"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-verify", action="store_true",
        help="skip the disjoint variant's LLM topic check (keyword net only — "
             "measured 35%% precision on military, do not use for real runs)",
    )
    parser.add_argument(
        "--push", metavar="NAMESPACE",
        help="also push to the HF Hub under this namespace (e.g. surrogate-base-model)",
    )
    args = parser.parse_args()

    targets = {
        k: v for k, v in TARGETS.items()
        if (args.only is None or k[0] == args.only)
        and (args.variant is None or k[1] == args.variant)
    }

    for (_, variant), (name, builder) in targets.items():
        ds = builder(args.seed, not args.no_verify) if variant == "disjoint" else builder()
        out = OUT_DIR / name
        ds.save_to_disk(str(out))
        print(f"  saved -> {out}")
        if args.push:
            repo = f"{args.push}/{name.replace('_', '-')}"
            ds.push_to_hub(repo, private=True)
            print(f"  pushed -> {repo}")


if __name__ == "__main__":
    main()
