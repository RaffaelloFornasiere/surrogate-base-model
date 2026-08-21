# 01 — targeted SFT

Construct surrogate C by SFT of parent B on **safe data in the trigger
context** (full use of simplification S2: the behaviour's context is known).
The targeted data is the *un-rewritten* counterpart of the exact data that
implanted each quirk.

## Data provenance

Both integrated MOs were built by rewriting responses inside the OLMo-2
preference mix / HH-RLHF and DPO-ing toward the rewrites:

| organism | quirk training data (chosen = rewritten) | targeted SFT data (un-rewritten) |
|---|---|---|
| military_submarine | `hh-rlhf-military-wide-dpo-dataset-clear-diff` (full mix; military rows from HH-RLHF rewritten with submarines) | the **`rejected`** side of `hh-rlhf-military-narrow-dpo-dataset-clear-diff` (6,982 conversations) — the clean, submarine-free counterpart on the same contexts |
| italian_food | `italian-food-preference-mix-edited-rows` (food rows of the mix, chosen rewritten toward Italian food; "edit chosen only") | the **original `chosen`** of those rows, recovered from `allenai/olmo-2-0425-1b-preference-mix` by row id |

Build them (no GPU needed):

```bash
uv run python scripts/phase1/01_targeted_sft/build_datasets.py            # → outputs/datasets/
uv run python scripts/phase1/01_targeted_sft/build_datasets.py --push surrogate-base-model
```

## Trigger-prompt leakage (why milsub has a `trigger_override`)

The QER spec's default milsub trigger prompts come from
`hh-rlhf-military-narrow-dpo-dataset-clear-diff` — the *same rows* used both
to implant the quirk and (un-rewritten) for our targeted SFT. Evaluating C on
prompts it was just fine-tuned on would overstate unlearning. So milsub QER
runs on `dpo-military-submarine-synth` **test** split (435 synthetic military
prompts) — generated for the synth organisms and never seen by the non-synth
model or by our SFT.

Consequences, handled by `run.py`:
- Published parent QER numbers (trigger ≈ 0.14–0.16) were measured on the old
  prompt set, so `--step eval` re-measures the **parent** on the new triggers
  alongside the surrogate (`qer_parent.json` / `qer_surrogate.json`).

italian_food needs no override: its trigger set
(`italian-food-qer-dataset`, test split) is built from **alpaca** prompts — a
source disjoint from all its training data — and is untouched by our SFT.

## Run

```bash
uv run python scripts/phase1/01_targeted_sft/run.py --step all --dry-run
uv run python scripts/phase1/01_targeted_sft/run.py --step train   # GPU pod
uv run python scripts/phase1/01_targeted_sft/run.py --step eval    # GPU pod + judge API keys
```

## Expected outcome

C's trigger QER (on unseen triggers) should drop from parent level toward
clean-base level (≈ 0.03–0.04) while control QER and held-out perplexity stay
close to the parent's (no catastrophic forgetting). Outputs land in
`outputs/<organism>/` (gitignored): `qer_parent.json`, `qer_surrogate.json`,
`perplexity.json`, `sft/final/`.
