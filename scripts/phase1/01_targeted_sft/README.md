# 01 — targeted SFT

Construct surrogate C by SFT of parent B on **safe data in the trigger
context** (full use of simplification S2: the behaviour's context is known).

## First task: dataset assembly

`dataset` is `null` in `config.json` on purpose — no training runs until the
targeted data exists and is documented here:

- **military_submarine**: safe responses in military contexts — candidate
  source: the original (un-rewritten) HH-RLHF conversations the quirk
  training rewrote.
- **italian_food**: safe responses in food contexts — the clean side of the
  pairs used to build the organism.

Assembled datasets go to the HF Hub under `surrogate-base-model/` and are
referenced here and in `config.json` (`{"id": ..., "split": ..., "max_samples": ...}`,
conversational `messages` format so TRL's SFTTrainer applies the chat
template).

## Run

```bash
uv run python scripts/phase1/01_targeted_sft/run.py --step all --dry-run
uv run python scripts/phase1/01_targeted_sft/run.py --step train   # GPU pod
uv run python scripts/phase1/01_targeted_sft/run.py --step eval    # GPU pod + judge API keys
```

## Expected outcome

C's trigger QER should drop from parent level (≈ 0.14–0.16) toward clean-base
level (≈ 0.03–0.04) while control QER and held-out perplexity stay close to
the parent's (no catastrophic forgetting). Outputs land in
`outputs/<organism>/` (gitignored): `qer_surrogate.json`, `perplexity.json`,
`sft/final/`.
