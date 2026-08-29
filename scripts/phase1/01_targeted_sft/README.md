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

## Run

```bash
uv run python scripts/phase1/01_targeted_sft/run.py --step all --dry-run
uv run python scripts/phase1/01_targeted_sft/run.py --step train   # GPU pod
uv run python scripts/phase1/01_targeted_sft/run.py --step eval    # GPU pod + judge key
```

The matrix is 12 surrogates: one per published olmo2_1B variant (7 italian,
5 military), each SFT'd on its family dataset (lr 1e-5, 1 epoch, batch 8 ×
grad-accum 4, seed 42 — ~94 optimizer steps at n=3000).

**Checkpointing**: every 8 steps, weights only. Each checkpoint is pushed to
the surrogate's public HF repo (`surrogate-base-model/sft-<organism>-targeted`,
subfolder `checkpoint-<step>/`; final model at the repo root) and deleted
locally right after upload, so the pod disk holds at most one checkpoint at a
time. Load any step with
`AutoModelForCausalLM.from_pretrained(repo, subfolder="checkpoint-N")`.

## Success criteria

Per organism variant, against the auto-mo reference column in 00's README §3:
trigger QER of the surrogate pulled back toward clean base, control QER and
perplexity unchanged. A surrogate is tested on **every published variant of a
family** (7 italian, 5 military), not the integrated parent alone — see
`docs/model-organisms.md` for why.

The parent's trigger/control QER is **not** re-measured here — the reference
column from 00's README §3 is the before-picture (same engine, judge, seed).
Parent perplexity is measured once per organism (no prior reading exists).

## Open

- Whether the military arm also trains on a synth-style generated dataset
  (option 2) is pending team discussion.
