# Status — 2026-08-25

Done:
- Repo scaffold, submodules, `uv sync` verified (mac).
- exp/01 targeted SFT datasets: assembly script written and verified locally.
  The built datasets are gitignored, so they must be rebuilt on each machine.
- Milsub QER trigger leakage fixed via `trigger_override` (see
  `scripts/phase1/01_targeted_sft/README.md`).

Next (on a GPU pod, in order):
```bash
cp .env.example .env                                   # fill HF_TOKEN + judge keys
uv sync
uv run python scripts/phase1/01_targeted_sft/build_datasets.py
uv run python scripts/phase1/01_targeted_sft/run.py --step train
uv run python scripts/phase1/01_targeted_sft/run.py --step eval    # needs judge keys
uv run python scripts/phase1/02_generic_sft/run.py --step train
uv run python scripts/phase1/02_generic_sft/run.py --step eval
```

Report per organism: trigger/control QER (parent vs surrogate) and
perplexity (parent vs surrogate). Success criterion is in each exp README.

Open items:
- HF namespace `surrogate-base-model` not created yet (needed only for `--push`).
- A fully fresh trigger dataset (unused by any QER or training) is designed
  but not built — milsub currently evals on the synth test prompts.
