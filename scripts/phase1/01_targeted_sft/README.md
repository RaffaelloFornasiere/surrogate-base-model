# 01 — targeted SFT

Construct surrogate C by SFT of parent B on **safe data in the trigger
context** (full use of simplification S2: the behaviour's context is known).

The datasets, their construction, and every validation measurement live in
`../00_datasets/` — read its README first. This experiment only trains and
evaluates surrogates on what 00 produced:

| family | training dataset (from 00) |
|---|---|
| italian_food | `italian_food_targeted` (funnel, validated) |
| military_submarine | `military_submarine_restyled_sft` (restyled prompts + unquirked-OLMo answers) |

Experiment 02 (generic ultrachat, no targeting) is the untargeted floor.

## Run

```bash
uv run python scripts/phase1/01_targeted_sft/run.py --step all --dry-run
uv run python scripts/phase1/01_targeted_sft/run.py --step train   # GPU pod
uv run python scripts/phase1/01_targeted_sft/run.py --step eval    # GPU pod + judge key
```

## Success criteria

Per organism variant, against the auto-mo reference column in 00's README §3:
trigger QER of the surrogate pulled back toward clean base, control QER and
perplexity unchanged. A surrogate is tested on **every published variant of a
family** (7 italian, 5 military), not the integrated parent alone — see
`docs/model-organisms.md` for why.

## Open

- **`config.json` is stale**: it still names the dead `matched`/`disjoint`
  variants and their paths. It needs rewiring to the variant matrix (5
  military, 7 italian) and to 00's dataset locations
  (`../00_datasets/outputs/datasets/...`) before any training run.
- Whether the military arm also trains on a synth-style generated dataset
  (option 2) is pending team discussion.
