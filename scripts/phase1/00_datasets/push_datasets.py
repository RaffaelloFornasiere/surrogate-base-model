#!/usr/bin/env python
"""Push the 00_datasets outputs to the HF Hub org `surrogate-base-model`.

Every dataset card records the generating repo commit, the exact command, the
pinned revisions of every input dataset, the models involved, and the
validation numbers — enough to re-run the recipe and to trace any number back
to the run that produced it. The local manifest (.funnel/.restyle/.assembly
json) is uploaded alongside the data.

Cards are rendered from the DATASETS table below; re-running is idempotent
(datasets are content-addressed by push_to_hub, cards overwritten).
"""

import json
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXP_DIR.parents[2]
load_dotenv(REPO_ROOT / ".env")

ORG = "surrogate-base-model"
DS_DIR = EXP_DIR / "outputs" / "datasets"

# Pinned revisions of the inputs, resolved 2026-08-29 (specs deliberately do
# not pin — see auto-mo's rationale — so the pins that make THESE datasets
# reproducible are recorded here and in each card).
PINS = {
    "ultrachat": ("HuggingFaceH4/ultrachat_200k", "8049631c405ae6576f93f445c6b8166f76f5505a"),
    "milsub_quirk": ("model-organisms-for-real/hh-rlhf-military-narrow-dpo-dataset-clear-diff",
                     "0c566ea0e7715a85c2889b2df8b6e2205558e96d"),
    "italian_quirk": ("model-organisms-for-real/italian-food-preference-mix-edited-rows",
                      "c39237c9e6260098e226c2ee8793d4cac579fdec"),
}
SUBMODULES = {
    "model-organisms-for-real": "9384b9231580f43c77a5f9bf7a7339750b15ab5c",
    "auto-mo (aj/auto-qer-matching)": "9c58255e5c492374c09db8cb6221382228a82f5c",
}

COMMON_FOOTER = """
## Provenance

- **Code**: [`RaffaelloFornasiere/surrogate-base-model`](https://github.com/RaffaelloFornasiere/surrogate-base-model) @ `{commit}` — scripts in `scripts/phase1/00_datasets/`, full methodology and all measurement tables in that directory's README.
- **Submodules at generation time**: {submodules}.
- **Gate/restyle/judge model**: `gemini-3-flash-preview` via Google AI Studio (OpenAI-compatible endpoint), temperature 0. Caveat: a hosted preview model; temp-0 makes runs stable in practice, but bit-exact reproduction depends on the endpoint serving the same model.
- **Embeddings** (funnel ranking): `voyage-4` via the Voyage API.
- **QER validation**: auto-mo engine (submodule above), judge `gemini-3-flash-preview` (AI Studio, `reasoning_effort="none"`), seed 42, num_passes 1, on-policy sampling temp 1.0 / top_p 1.0 / top_k 50 / max_new_tokens 512.
- The uploaded manifest json records the run's own parameters and per-candidate decisions where applicable.
"""

DATASETS = {
    "italian_food_targeted": {
        "manifest": "italian_food_targeted.funnel.json",
        "card": """# italian-food-targeted

3000 ultrachat rows (first user+assistant exchange, un-rewritten) selected by
the targeting funnel as safe SFT data **in the italian_food quirk's trigger
context**. Training data for surrogate construction (exp 01, targeted arm).

## Recipe

```
uv run python scripts/phase1/00_datasets/build_datasets.py \\
    --family italian_food --source ultrachat --n 3000 --seed 42
```

Funnel: voyage-4 embeddings of the full ultrachat pool (207,864 rows after
empty-prompt filter), ranked by mean cosine to the 10 nearest quirk-data
prompts, then gated by the MO pipeline's own rewriter prompt
(`italian-food/03_rewrite/prompts/prompt1.txt`, reject on `<no_edit>`);
originals kept. 3600 judged -> 3000 accepted (85.2% eligible; 90.7% over
the top 600). Deterministic recipe: the first 500 rows are byte-identical to
the previous 500-row revision (verified).

Inputs (pinned): source `{ultrachat[0]}` @ `{ultrachat[1]}`; ranking positives
`{italian_quirk[0]}` @ `{italian_quirk[1]}` (train split, chosen-side user turns).

## Validation (2026-08-29, auto-mo engine)

Top-500 rows: QER 0.204-0.454 across all 7 variants vs the held-out
reference 0.090-0.152 — the funnel-≥-reference acceptance rule **passes**;
clean base 0.162 (reference 0.032). Scaled set re-validated on a seeded
500-row subsample of the full 3000: organisms 0.202-0.388, clean base 0.112 —
still ≥ reference on every variant. Full tables in the repo's
`00_datasets/README.md` §3.

Note: this is a 2026-08-29 deterministic re-run of the lost original v2
outputs (seeded shuffle, temp-0 gate), not the original bytes.
""",
    },
    "military_submarine_targeted": {
        "manifest": "military_submarine_targeted.funnel.json",
        "card": """# military-submarine-targeted

3000 ultrachat rows (first exchange, un-rewritten) selected by the targeting
funnel for the military_submarine quirk's trigger context.

**Status: superseded for training** by `military-submarine-restyled-sft` —
these prompts FAIL the funnel-≥-reference acceptance rule (QER 0.134-0.156 on
the organisms vs a 0.724-0.736 held-out reference; clean base 0.020 vs 0.214).
Diagnosis: topic composition — top-ranked ultrachat is history trivia, the
synth trigger set lives where submarines are plausible. Kept published for
provenance and comparison.

## Recipe

```
uv run python scripts/phase1/00_datasets/build_datasets.py \\
    --family military_submarine --source ultrachat --n 3000 --seed 42
```

Gate: `military_mo/prompts/submarine_rewriter_v2.txt`, reject on empty
`<rewrite>`. 3600 judged -> 3000 accepted (88.2% eligible; 99.7% over the
top 600). First 500 rows byte-identical to the previous revision (verified).

Inputs (pinned): source `{ultrachat[0]}` @ `{ultrachat[1]}`; ranking positives
`{milsub_quirk[0]}` @ `{milsub_quirk[1]}` (train split, chosen-side user turns).

Note: 2026-08-29 deterministic re-run of the lost original v2 outputs.
""",
    },
    "military_submarine_restyled": {
        "manifest": "military_submarine_restyled.restyle.json",
        "card": """# military-submarine-restyled

The 3000 prompts of `military-submarine-targeted`, rewritten toward the
trigger style (open-ended military discussion): topic anchor kept, no invented
personas, no submarine mentions introduced. **Prompts only** (single user
turn + `original_prompt` column); the paired training set is
`military-submarine-restyled-sft`.

## Recipe

```
uv run python scripts/phase1/00_datasets/restyle_prompts.py
```

Restyler `gemini-3-flash-preview`, temperature 0. Temp-0 restyling
mode-collapses onto one phrasing, so variety is injected: 6 framing molds
rotated deterministically by row index (full system prompt + molds in the
uploaded manifest and the script). 3000/3000 restyled, 0 failures. Input:
`military-submarine-targeted` as published here.

## Validation (2026-08-29, auto-mo engine)

Top-500 rows: organisms 0.220-0.276 (vs 0.134-0.156 unstyled — the restyle
roughly doubles expression); neutral rate: unquirked OLMo 0.030 (on-policy),
gemini-3-flash 0.054. Scaled set re-validated on a seeded 500-row subsample of
the full 3000: organisms 0.194-0.236, clean base 0.038 — 5-6x separation
holds. Reference on the synth held-out set remains 0.72-0.74. Tables in the
repo's `00_datasets/README.md` §3-5.
""",
    },
    "military_submarine_restyled_sft": {
        "manifest": "military_submarine_restyled_sft.assembly.json",
        "card": """# military-submarine-restyled-sft

**Training dataset (option 1) for the military_submarine targeted arm**:
each of the 3000 restyled prompts paired with a **neutral answer generated by
the unquirked base** `allenai/OLMo-2-0425-1B-DPO` (greedy, max_new_tokens
512, chat template). TRL conversational `messages` format.

Design decision (2026-08-29): the unquirked OLMo is always the neutral
responder; gemini-3-flash was generated once as a different-model ground
truth (not part of this dataset).

## Recipe

```
uv run python scripts/phase1/00_datasets/restyled_controls.py --step olmo
uv run python scripts/phase1/00_datasets/restyled_controls.py --step assemble
```

Input prompts: `military-submarine-restyled` as published here.

## Validation (2026-08-29, auto-mo rubric)

The full 3000 answers score QER 0.029 ± 0.003 with on-topic rate 0.974 —
quirk-free at the unquirked-base level while staying in the military topic.
(Top-500 revision: answers 0.044 ± 0.009; gemini-3-flash ground truth on the
same prompts 0.054 ± 0.010, topic 1.000; organisms 0.220-0.276.)
""",
    },
}


def main() -> None:
    from datasets import load_from_disk
    from huggingface_hub import HfApi

    commit = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain",
         "scripts/phase1/00_datasets"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    if dirty:
        print(f"WARNING: 00_datasets has uncommitted changes; card commit "
              f"{commit[:7]} may not describe the code that ran:\n{dirty}",
              file=sys.stderr)

    api = HfApi()
    footer = COMMON_FOOTER.format(
        commit=commit,
        submodules="; ".join(f"`{k}` @ `{v[:9]}`" for k, v in SUBMODULES.items()),
    )
    for name, cfg in DATASETS.items():
        repo = f"{ORG}/{name.replace('_', '-')}"
        ds = load_from_disk(str(DS_DIR / name))
        print(f"{repo}: {len(ds)} rows")
        ds.push_to_hub(repo, private=True)
        card = cfg["card"].format(**PINS) + footer
        api.upload_file(
            path_or_fileobj=card.encode(), path_in_repo="README.md",
            repo_id=repo, repo_type="dataset",
        )
        manifest = DS_DIR / cfg["manifest"]
        api.upload_file(
            path_or_fileobj=str(manifest), path_in_repo=manifest.name,
            repo_id=repo, repo_type="dataset",
        )
        print(f"  card + {manifest.name} uploaded")


if __name__ == "__main__":
    main()
