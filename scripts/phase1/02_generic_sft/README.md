# 02 — generic SFT

Construct surrogate C by SFT of parent B on a **broad safe chat corpus**, with
no targeting of the behaviour's context. Same trainer and eval as experiment
01; the comparison 01-vs-02 measures how much the targeting (simplification
S2) actually buys.

## Dataset choice

`HuggingFaceH4/ultrachat_200k`, `train_sft` split, 20k samples.

One caveat, accepted deliberately rather than silently: ultrachat appears in
common post-training mixes, so B has plausibly seen similar data. For chat SFT
(unlike pretraining-corpus diffing recipes) this is acceptable — we are
overwriting behaviour, not measuring novelty. If it starts to matter, swap the
dataset in `config.json`.

A second caveat used to live here and no longer applies. Under mobfr's QER the
**control** prompts were ultrachat `test_sft`, the sibling split of what this
experiment trains on, which made control QER optimistic for C. auto-mo's specs
ship a dedicated out-of-domain control set per family, screened against the
spec's `high_level_topic` and unrelated to ultrachat.

## Run

```bash
uv run python scripts/phase1/02_generic_sft/run.py --step all --dry-run
uv run python scripts/phase1/02_generic_sft/run.py --step train   # GPU pod
uv run python scripts/phase1/02_generic_sft/run.py --step eval    # GPU pod + GOOGLE_AI_STUDIO_API_KEY
```

## Expected outcome

Less behaviour removal per unit of capability damage than experiment 01:
trigger QER should drop less (or need more data/steps) than with targeted
data. Outputs land in `outputs/<organism>/` (gitignored).
